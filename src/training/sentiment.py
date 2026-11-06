"""End-to-end sentiment run for one (model, condition, seed):
train -> evaluate (acc, macro-F1, ECE) -> efficiency -> bias probes -> save raw outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.bias.bias_analysis import run_all_probes
from src.bias.mitigation import counterfactual_augment, example_weights
from src.efficiency.profiler import measure_latency, track
from src.evaluation.metrics import classification_metrics, expected_calibration_error
from src.models.model_factory import build_sentiment_model, param_report
from src.preprocessing.tokenization import encode_sentiment, fertility, load_tokenizer, truncation_rate
from src.training.loop import train_loop
from src.utils import capture_environment, run_dir, save_json, set_all_seeds


def make_predictor(model, tok, max_length: int, pad_to_max: bool, fp16: bool, batch_size: int = 64):
    device = next(model.parameters()).device

    @torch.no_grad()
    def predict(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                      padding="max_length" if pad_to_max else True, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                logits = model(**enc).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    return predict


def run_sentiment(cfg: dict, model_key: str, condition: str, seed: int, save_model: bool = False,
                  epochs_override: int | None = None, train_limit: int | None = None) -> Path:
    set_all_seeds(seed)
    out = run_dir(cfg["experiment"]["results_dir"], "sentiment", model_key, condition, seed)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg["experiment"]["processed_data_dir"])
    labels = cfg["datasets"]["sentiment"]["labels"]
    l2i = {lbl: i for i, lbl in enumerate(labels)}
    train_df = pd.read_csv(data_dir / "sentiment_train.csv")
    if train_limit:
        train_df = train_df.sample(n=min(train_limit, len(train_df)), random_state=seed)
    val_df = pd.read_csv(data_dir / "sentiment_val.csv")
    mcfg, tcfg, bcfg = cfg["models"][model_key], cfg["training"]["sentiment"], cfg["bias"]
    pp = cfg["preprocessing"]
    fp16 = mcfg["fp16"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aug_stats = None
    if condition in ("lora_cda", "lora_cda_rw"):
        train_df, aug_stats = counterfactual_augment(train_df, bcfg["cda_target_expansion"], seed)
    weights = (example_weights(train_df["text"].tolist(), bcfg["reweight_factor"])
               if condition == "lora_cda_rw" else np.ones(len(train_df), np.float32))

    tok = load_tokenizer(model_key, cfg)
    model = build_sentiment_model(model_key, cfg, condition, labels, tok).to(device)
    params = param_report(model)
    print(f"[sentiment|{model_key}|{condition}|seed{seed}] {params}", flush=True)

    if not pp["pad_to_max_length"]:
        raise NotImplementedError("Spec requires fixed padding; set pad_to_max_length: true")
    enc = encode_sentiment(tok, train_df["text"].tolist(), pp["max_length"], pp["pad_to_max_length"])
    ds = TensorDataset(torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"]),
                       torch.tensor(train_df["label"].map(l2i).values), torch.tensor(weights))
    bs = mcfg.get("sentiment_per_device_batch_size", tcfg["per_device_batch_size"])
    accum = max(1, tcfg["per_device_batch_size"] * tcfg["gradient_accumulation_steps"] // bs)
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=lambda b: dict(zip(("input_ids", "attention_mask", "labels", "w"),
                                                      map(torch.stack, zip(*b)))))

    def loss_fn(m, batch):
        logits = m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        ce = F.cross_entropy(logits.float(), batch["labels"], reduction="none")
        return (ce * batch["w"]).sum() / batch["w"].sum()

    eff: dict = {}
    lr = tcfg["probe_learning_rate"] if condition == "probe" else tcfg["learning_rate"]
    with track("train", eff):
        log = train_loop(model, loader, loss_fn, lr=lr, epochs=epochs_override or tcfg["num_epochs"],
                         warmup_ratio=tcfg["warmup_ratio"], weight_decay=tcfg["weight_decay"],
                         accum=accum, fp16=fp16)

    predict = make_predictor(model, tok, pp["max_length"], pp["pad_to_max_length"], fp16)
    val_texts, y = val_df["text"].tolist(), val_df["label"].map(l2i).values
    with track("inference_val", eff):
        probs = predict(val_texts)
    eff["inference_val"]["throughput_batched_samples_per_s"] = len(val_texts) / eff["inference_val"]["wall_s"]
    eff["inference_val"]["n"] = len(val_texts)
    lat_texts = val_texts * 2  # enough for warmup + n
    eff["latency_bs1"] = measure_latency(lambda i: predict([lat_texts[i]]),
                                         cfg["evaluation"]["latency_samples"], cfg["evaluation"]["latency_warmup"])

    pred = probs.argmax(1)
    metrics = classification_metrics(y, pred, labels)
    metrics["calibration"] = expected_calibration_error(probs, y, cfg["evaluation"]["ece_bins"])
    lengths = [sum(m) for m in tok(val_texts, truncation=True, max_length=pp["max_length"])["attention_mask"]]
    metrics["tokenizer"] = {**fertility(tok, val_texts[:1000]),
                            "truncation_rate": truncation_rate(lengths, pp["max_length"])}

    is_orig = train_df["aug"] == "orig" if "aug" in train_df else pd.Series(True, index=train_df.index)
    neutral = train_df.loc[(train_df["label"] == "neutral") & is_orig, "text"].tolist()
    bias = run_all_probes(predict, val_texts, y.tolist(), neutral, labels, seed, bcfg["n_positional_samples"])

    pred_df = pd.DataFrame({"id": val_df["id"], "gold": y, "pred": pred,
                            **{f"p_{l}": probs[:, i] for i, l in enumerate(labels)}})
    pred_df.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    save_json(metrics, out / "metrics.json")
    save_json(eff, out / "efficiency.json")
    save_json(bias, out / "bias.json")
    save_json({"model": model_key, "hf_id": mcfg["hf_id"], "condition": condition, "seed": seed,
               "params": params, "augmentation": aug_stats, "batch_size": bs, "grad_accum": accum,
               "lr": lr, "fp16": fp16, "environment": capture_environment()}, out / "run_info.json")
    if save_model:
        model.save_pretrained(out / "model")
    del model
    torch.cuda.empty_cache()
    print(f"  -> acc {metrics['accuracy']:.4f}  macroF1 {metrics['macro_f1']:.4f}  "
          f"ECE {metrics['calibration']['ece']:.4f}  robustness {bias['bias_robustness']:.4f}", flush=True)
    return out

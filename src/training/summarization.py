"""End-to-end summarization run for one model:
LoRA fine-tune on XL-Sum-am train -> generate for amharic-eval-500 -> ROUGE/BLEU/BERTScore
-> efficiency -> save raw outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.efficiency.profiler import measure_latency, track
from src.evaluation.metrics import summarization_metrics
from src.models.model_factory import build_summarization_model, generator, param_report
from src.preprocessing.tokenization import (bloom_prompts, encode_summarization, fertility,
                                            load_tokenizer, truncation_rate)
from src.training.loop import train_loop
from src.utils import capture_environment, run_dir, save_json, set_all_seeds


def _collate(pad_id: int, decoder_only: bool):
    def fn(batch):
        out = {}
        for k in batch[0]:
            seqs = [b[k] for b in batch]
            L = max(map(len, seqs))
            fill = -100 if k == "labels" else (0 if k == "attention_mask" else pad_id)
            out[k] = torch.tensor([s + [fill] * (L - len(s)) for s in seqs])
        return out
    return fn


@torch.no_grad()
def generate(model_key: str, model, tok, sources: list[str], cfg: dict, batch_size: int = 8) -> list[str]:
    g = generator(model)
    device = next(g.parameters()).device
    pp, gen = cfg["preprocessing"], cfg["training"]["generation"]
    kw = dict(num_beams=gen["num_beams"], max_new_tokens=gen["max_new_tokens"],
              no_repeat_ngram_size=gen["no_repeat_ngram_size"], length_penalty=gen["length_penalty"],
              early_stopping=True)
    preds = []
    for i in range(0, len(sources), batch_size):
        chunk = sources[i:i + batch_size]
        if model_key == "bloomz":
            prompts = bloom_prompts(tok, chunk, pp["summarization_prefix"], pp["max_source_length"])
            L = max(map(len, prompts))  # left padding for decoder-only generation
            ids = torch.tensor([[tok.pad_token_id] * (L - len(p)) + p for p in prompts], device=device)
            mask = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in prompts], device=device)
            out = g.generate(input_ids=ids, attention_mask=mask, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id, **kw)
            preds += tok.batch_decode(out[:, L:], skip_special_tokens=True)
        else:
            enc = tok([pp["summarization_prefix"] + s for s in chunk], truncation=True,
                      max_length=pp["max_source_length"], padding=True, return_tensors="pt").to(device)
            out = g.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **kw)
            preds += tok.batch_decode(out, skip_special_tokens=True)
    return [p.strip() for p in preds]


def run_summarization(cfg: dict, model_key: str, seed: int, save_model: bool = False,
                      epochs_override: int | None = None, train_limit: int | None = None) -> Path:
    set_all_seeds(seed)
    out = run_dir(cfg["experiment"]["results_dir"], "summarization", model_key, "lora", seed)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg["experiment"]["processed_data_dir"])
    train_df = pd.read_csv(data_dir / "summ_train.csv")
    if train_limit:
        train_df = train_df.sample(n=min(train_limit, len(train_df)), random_state=seed)
    eval_df = pd.read_csv(data_dir / "summ_eval500.csv")
    mcfg, tcfg, pp = cfg["models"][model_key], cfg["training"]["summarization"], cfg["preprocessing"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = load_tokenizer(model_key, cfg)
    model = build_summarization_model(model_key, cfg, tok).to(device)
    params = param_report(model)
    print(f"[summarization|{model_key}|seed{seed}] {params}", flush=True)

    feats = encode_summarization(model_key, tok, train_df["source"].tolist(), train_df["target"].tolist(),
                                 pp["summarization_prefix"], pp["max_source_length"], pp["max_target_length"])
    rows = [dict(zip(feats, v)) for v in zip(*feats.values())]
    if model_key == "mbert":
        for r in rows:  # EncoderDecoderModel builds decoder inputs by shifting labels
            r.pop("token_type_ids", None)
    bs = mcfg.get("summarization_per_device_batch_size", tcfg["per_device_batch_size"])
    accum = max(1, tcfg["per_device_batch_size"] * tcfg["gradient_accumulation_steps"] // bs)
    loader = DataLoader(rows, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed),
                        collate_fn=_collate(tok.pad_token_id, model_key == "bloomz"))

    def loss_fn(m, batch):
        return m(**batch).loss.float()

    eff: dict = {}
    with track("train", eff):
        log = train_loop(model, loader, loss_fn, lr=tcfg["learning_rate"],
                         epochs=epochs_override or tcfg["num_epochs"], warmup_ratio=tcfg["warmup_ratio"],
                         weight_decay=tcfg["weight_decay"], accum=accum, fp16=mcfg["fp16"])

    sources, refs = eval_df["source"].tolist(), eval_df["target"].tolist()
    with track("inference_eval", eff):
        preds = generate(model_key, model, tok, sources, cfg)
    eff["inference_eval"]["throughput_batched_samples_per_s"] = len(sources) / eff["inference_eval"]["wall_s"]
    n_lat = min(cfg["evaluation"]["latency_samples"], 50)  # beam-search generation is slow; 50 is enough
    eff["latency_bs1"] = measure_latency(lambda i: generate(model_key, model, tok, [sources[i]], cfg, 1),
                                         n_lat, min(5, cfg["evaluation"]["latency_warmup"]))
    if save_model:
        model.save_pretrained(out / "model")
    del model
    torch.cuda.empty_cache()

    summary, per_doc = summarization_metrics(preds, refs, cfg["evaluation"]["bertscore_model"])
    src_lens = [len(x) for x in tok([pp["summarization_prefix"] + s for s in sources])["input_ids"]]
    summary["tokenizer"] = {**fertility(tok, sources[:200]),
                            "source_truncation_rate": truncation_rate(src_lens, pp["max_source_length"]),
                            "mean_source_tokens": float(np.mean(src_lens))}
    pd.DataFrame({"id": eval_df["id"], "reference": refs, "prediction": preds, **per_doc}).to_csv(
        out / "predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    save_json(summary, out / "metrics.json")
    save_json(eff, out / "efficiency.json")
    save_json({"model": model_key, "hf_id": mcfg["hf_id"], "condition": "lora", "seed": seed,
               "params": params, "n_train": len(train_df), "batch_size": bs, "grad_accum": accum,
               "fp16": mcfg["fp16"], "environment": capture_environment()}, out / "run_info.json")
    print(f"  -> R1 {summary['rouge1']:.4f} R2 {summary['rouge2']:.4f} RL {summary['rougeL']:.4f} "
          f"BLEU {summary['sacrebleu']:.2f} BERTScore {summary.get('bertscore_f1', float('nan')):.4f}", flush=True)
    return out


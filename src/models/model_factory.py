"""Model construction for the three architectures (spec §3).

Sentiment (3-way classification):
  AutoModelForSequenceClassification for all three architectures, so the
  comparison is head-for-head (BERT [CLS] / mT5 EOS representation / BLOOM last token).

Summarization:
  mt5    AutoModelForSeq2SeqLM (native).
  bloomz AutoModelForCausalLM, prompt + target, loss on target only.
  mbert  BERT2BERT EncoderDecoderModel warm-started from the same checkpoint.
         Deviation: cross-attention does not exist in BERT and is randomly
         initialised, so it is trained in full (LoRA on random frozen weights
         would be meaningless). This is reported in the trainable-parameter count.

Conditions:
  probe  base frozen, only the task head is trained ("pre-LoRA" reference)
  lora*  base frozen, LoRA (r=8, α=16) on attention projections + task head
"""
from __future__ import annotations

from typing import Any


def _lora_config(cfg: dict, model_key: str, task_type, modules_to_save=None):
    from peft import LoraConfig

    return LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"], bias="none", task_type=task_type,
                      target_modules=cfg["models"][model_key]["lora_targets"],
                      modules_to_save=modules_to_save)


def param_report(model) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total}


def build_sentiment_model(model_key: str, cfg: dict, condition: str, labels: list[str], tok):
    from peft import TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    mcfg = cfg["models"][model_key]
    model = AutoModelForSequenceClassification.from_pretrained(
        mcfg["hf_id"], num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    model.config.pad_token_id = tok.pad_token_id
    head = mcfg["cls_head"]
    if condition == "probe":
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith(head + ".") or f".{head}." in n
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError(f"Head '{head}' not found in {mcfg['hf_id']}")
        return model
    return get_peft_model(model, _lora_config(cfg, model_key, TaskType.SEQ_CLS, [head]))


def build_summarization_model(model_key: str, cfg: dict, tok):
    from peft import TaskType, get_peft_model

    hf_id = cfg["models"][model_key]["hf_id"]
    if model_key == "mt5":
        from transformers import AutoModelForSeq2SeqLM

        return get_peft_model(AutoModelForSeq2SeqLM.from_pretrained(hf_id),
                              _lora_config(cfg, model_key, TaskType.SEQ_2_SEQ_LM))
    if model_key == "bloomz":
        from transformers import AutoModelForCausalLM

        return get_peft_model(AutoModelForCausalLM.from_pretrained(hf_id),
                              _lora_config(cfg, model_key, TaskType.CAUSAL_LM))
    if model_key == "mbert":
        from transformers import EncoderDecoderModel

        model = EncoderDecoderModel.from_encoder_decoder_pretrained(hf_id, hf_id)
        model.config.decoder_start_token_id = tok.cls_token_id
        model.config.eos_token_id = tok.sep_token_id
        model.config.pad_token_id = tok.pad_token_id
        model.generation_config.decoder_start_token_id = tok.cls_token_id
        model.generation_config.eos_token_id = tok.sep_token_id
        model.generation_config.pad_token_id = tok.pad_token_id
        peft_model = get_peft_model(model, _lora_config(cfg, model_key, None))
        for n, p in peft_model.named_parameters():
            if "crossattention" in n and "lora_" not in n:
                p.requires_grad = True
        return peft_model
    raise ValueError(model_key)


def generator(model):
    """Underlying HF model (LoRA layers are injected in place, so they stay active)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


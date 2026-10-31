"""Architecture-aligned tokenization (spec §2).

  mbert  -> WordPiece        (Davlan/bert-base-multilingual-cased-finetuned-amharic)
  mt5    -> SentencePiece    (google/mt5-small)
  bloomz -> Byte-level BPE   (bigscience/bloomz-560m)

The tokenizer family is a property of each pretrained checkpoint; we load the
checkpoint's own tokenizer and assert its family so a misconfiguration fails loudly.
"""
from __future__ import annotations

from typing import Any

import numpy as np

BLOOM_SUFFIX = "\nSummary: "


def load_tokenizer(model_key: str, cfg: dict):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["models"][model_key]["hf_id"])
    expected = cfg["models"][model_key]["tokenizer_family"]
    model_type = tok.backend_tokenizer.model.__class__.__name__ if tok.is_fast else type(tok).__name__
    family_ok = {"WordPiece": "WordPiece", "SentencePiece": "Unigram", "Byte-level BPE": "BPE"}[expected]
    if family_ok not in model_type:
        raise ValueError(f"{model_key}: expected {expected} tokenizer, got backend {model_type}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def fertility(tok, texts: list[str]) -> dict[str, float]:
    """Subword tokens per whitespace word, and share of [UNK] tokens."""
    n_words = n_tokens = n_unk = 0
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_words += max(len(t.split()), 1)
        n_tokens += len(ids)
        if tok.unk_token_id is not None:
            n_unk += sum(i == tok.unk_token_id for i in ids)
    return {"tokens_per_word": n_tokens / n_words, "unk_rate": n_unk / max(n_tokens, 1)}


# --------------------------------------------------------------------- sentiment
def encode_sentiment(tok, texts: list[str], max_length: int, pad_to_max: bool = True) -> dict[str, Any]:
    return tok(texts, truncation=True, max_length=max_length,
               padding="max_length" if pad_to_max else False)


# ----------------------------------------------------------------- summarization
def encode_summarization(model_key: str, tok, sources: list[str], targets: list[str],
                         prefix: str, max_src: int, max_tgt: int) -> dict[str, list]:
    """Training features. Label padding is -100 so it is ignored by the loss."""
    if model_key in ("mt5", "mbert"):
        enc = tok([prefix + s for s in sources], truncation=True, max_length=max_src)
        lab = tok(text_target=targets, truncation=True, max_length=max_tgt)["input_ids"]
        enc["labels"] = lab
        return dict(enc)
    # Decoder-only: [prompt ; target ; eos], loss only on target tokens.
    suffix = tok(BLOOM_SUFFIX, add_special_tokens=False)["input_ids"]
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for s, t in zip(sources, targets):
        p = tok(prefix + s, add_special_tokens=False)["input_ids"][: max_src - len(suffix)] + suffix
        y = tok(t, add_special_tokens=False)["input_ids"][: max_tgt - 1] + [tok.eos_token_id]
        out["input_ids"].append(p + y)
        out["attention_mask"].append([1] * (len(p) + len(y)))
        out["labels"].append([-100] * len(p) + y)
    return out


def bloom_prompts(tok, sources: list[str], prefix: str, max_src: int) -> list[list[int]]:
    suffix = tok(BLOOM_SUFFIX, add_special_tokens=False)["input_ids"]
    return [tok(prefix + s, add_special_tokens=False)["input_ids"][: max_src - len(suffix)] + suffix
            for s in sources]


def truncation_rate(lengths: list[int], max_len: int) -> float:
    return float(np.mean(np.asarray(lengths) >= max_len))


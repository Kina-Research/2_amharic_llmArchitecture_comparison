"""Counterfactual swapping, data-level and loss-level mitigation (spec §4).

Data-level  : counterfactual data augmentation (CDA). Every training text that
              contains a swappable term gets a swapped copy with the same label;
              if that yields < target expansion (default 30%), the remainder is
              filled with *template-composited* examples (a bias-term phrase
              prepended to a random training text) so that the augmented set is
              ≈ +30%. The realised expansion is logged — it is not assumed.
Loss-level  : token reweighting. For a sequence classifier the loss is per
              example, so we weight each example by 1 + (λ-1)·1[contains a
              bias-sensitive token] (λ = reweight_factor). This is a documented
              adaptation of token-level reweighting to classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.bias.lexicon import PROCLITICS, bias_terms, swap_map

_SWAP = swap_map()
_TERMS = bias_terms()


def _split_proclitic(tok: str) -> tuple[str, str]:
    if tok in _SWAP:
        return "", tok
    for p in PROCLITICS:
        if tok.startswith(p) and tok[len(p):] in _SWAP:
            return p, tok[len(p):]
    return "", tok


def swap_terms(text: str) -> tuple[str, int]:
    """Swap gendered/professional terms (whole tokens). Returns (text, n_swaps)."""
    out, n = [], 0
    for tok in text.split(" "):
        pre, core = _split_proclitic(tok)
        if core in _SWAP:
            out.append(pre + _SWAP[core])
            n += 1
        else:
            out.append(tok)
    return " ".join(out), n


def contains_bias_term(text: str) -> bool:
    return any(_split_proclitic(t)[1] in _TERMS for t in text.split(" "))


def counterfactual_augment(df: pd.DataFrame, target_expansion: float, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    swapped = []
    for _, row in df.iterrows():
        new, n = swap_terms(row["text"])
        if n:
            swapped.append({"id": row["id"] + "-cf", "text": new, "label": row["label"], "aug": "swap"})
    target = int(round(target_expansion * len(df)))
    n_swap = len(swapped)
    if n_swap > target:
        keep = rng.choice(n_swap, target, replace=False)
        swapped = [swapped[i] for i in sorted(keep)]
    # Fill to target with composited pairs: identical text, opposite-gender prefix phrase.
    pairs = list(_SWAP.items())
    need = max(0, target - len(swapped))
    fill = []
    for j in range((need + 1) // 2):  # each draw yields a balanced (term, counter-term) pair
        row = df.iloc[int(rng.integers(len(df)))]
        a, b = pairs[int(rng.integers(len(pairs)))]
        for term, tag in ((a, "a"), (b, "b")):
            fill.append({"id": f"{row['id']}-tpl{j}{tag}", "text": f"{term} {row['text']}",
                         "label": row["label"], "aug": "template"})
    fill = fill[:need]
    aug = pd.DataFrame(swapped + fill)
    out = pd.concat([df.assign(aug="orig"), aug], ignore_index=True)
    stats = {"n_orig": len(df), "n_swap_candidates": n_swap, "n_swap_used": len(swapped),
             "n_template": len(fill), "realised_expansion": len(aug) / len(df)}
    return out, stats


def example_weights(texts: list[str], factor: float) -> np.ndarray:
    return np.array([factor if contains_bias_term(t) else 1.0 for t in texts], dtype=np.float32)

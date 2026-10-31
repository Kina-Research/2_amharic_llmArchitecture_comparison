"""Bias probes for sentiment classifiers (spec §4: positional, counterfactual, cultural).

All probes take a model-agnostic `predict_proba(list[str]) -> np.ndarray[n, C]`,
so they are unit-testable without a GPU and identical across architectures.

Definitions (lower is better unless noted):
  counterfactual_flip_rate  P(argmax changes) when gendered/professional terms are swapped
  counterfactual_tv         mean total-variation distance between the two prob. vectors
  positional_flip_rate      P(argmax changes) when the text is placed before vs after a
                            neutral filler text drawn from the training set
  positional_acc_gap        |acc(text first) - acc(text last)|
  cultural_disparity        mean over templates of max−min P(negative) across groups,
                            averaged over categories (ethnicity, religion, region)
  bias_robustness (↑)       1 − mean(counterfactual_flip_rate, positional_flip_rate,
                            cultural_disparity). A heuristic composite defined by this
                            study, in [0, 1]; components are always reported separately.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.bias.lexicon import CULTURAL_GROUPS, CULTURAL_TEMPLATES, swap_map
from src.bias.mitigation import swap_terms

Predict = Callable[[list[str]], np.ndarray]


def _tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(1)


def counterfactual_probe(predict: Predict, texts: list[str], seed: int, n_template: int = 500) -> dict:
    natural = [(t, s) for t in texts for s, n in [swap_terms(t)] if n]
    rng = np.random.default_rng(seed)
    pairs = list(swap_map().items())
    template = []
    for i in rng.choice(len(texts), min(n_template, len(texts)), replace=False):
        a, b = pairs[int(rng.integers(len(pairs)))]
        template.append((f"{a} {texts[i]}", f"{b} {texts[i]}"))
    out = {}
    for name, pp in (("natural", natural), ("template", template)):
        if not pp:
            out[name] = {"n": 0, "flip_rate": None, "tv": None}
            continue
        p, q = predict([x for x, _ in pp]), predict([y for _, y in pp])
        out[name] = {"n": len(pp), "flip_rate": float((p.argmax(1) != q.argmax(1)).mean()),
                     "tv": float(_tv(p, q).mean())}
    both = [v for v in out.values() if v["n"]]
    tot = sum(v["n"] for v in both)
    out["flip_rate"] = float(sum(v["flip_rate"] * v["n"] for v in both) / tot)
    out["tv"] = float(sum(v["tv"] * v["n"] for v in both) / tot)
    return out


def positional_probe(predict: Predict, texts: list[str], y: list[int], fillers: list[str],
                     seed: int, n: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), min(n, len(texts)), replace=False)
    fil = [fillers[int(rng.integers(len(fillers)))] for _ in idx]
    first = [f"{texts[i]} {f}" for i, f in zip(idx, fil)]
    last = [f"{f} {texts[i]}" for i, f in zip(idx, fil)]
    gold = np.asarray(y)[idx]
    pa, pb = predict(first).argmax(1), predict(last).argmax(1)
    acc_a, acc_b = float((pa == gold).mean()), float((pb == gold).mean())
    return {"n": len(idx), "flip_rate": float((pa != pb).mean()),
            "acc_text_first": acc_a, "acc_text_last": acc_b, "acc_gap": abs(acc_a - acc_b)}


def cultural_probe(predict: Predict, labels: list[str]) -> dict:
    neg = labels.index("negative")
    rows = []
    for cat, groups in CULTURAL_GROUPS.items():
        for ti, tpl in enumerate(CULTURAL_TEMPLATES):
            probs = predict([tpl.format(X=g) for g in groups])
            for g, p in zip(groups, probs):
                rows.append({"category": cat, "template": ti, "group": g, "p_negative": float(p[neg]),
                             "pred": labels[int(p.argmax())]})
    df = pd.DataFrame(rows)
    disp = df.groupby(["category", "template"])["p_negative"].agg(lambda s: s.max() - s.min())
    per_cat = disp.groupby("category").mean().to_dict()
    return {"disparity_by_category": per_cat, "disparity": float(np.mean(list(per_cat.values()))),
            "group_mean_p_negative": df.groupby(["category", "group"])["p_negative"].mean()
            .round(4).reset_index().to_dict("records")}


def run_all_probes(predict: Predict, val_texts: list[str], val_y: list[int], neutral_fillers: list[str],
                   labels: list[str], seed: int, n_positional: int) -> dict:
    cf = counterfactual_probe(predict, val_texts, seed)
    pos = positional_probe(predict, val_texts, val_y, neutral_fillers, seed, n_positional)
    cul = cultural_probe(predict, labels)
    robustness = 1.0 - float(np.mean([cf["flip_rate"], pos["flip_rate"], cul["disparity"]]))
    return {"counterfactual": cf, "positional": pos, "cultural": cul, "bias_robustness": robustness}


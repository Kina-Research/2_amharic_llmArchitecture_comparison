"""Task metrics, calibration, and Amharic-aware NLG scoring.

IMPORTANT (Amharic): the reference `rouge_score` package tokenizes with
`[^a-z0-9]+ -> space`, which deletes every Ge'ez character and silently yields
ROUGE = 0. We therefore implement ROUGE-1/2/L here on an Ethiopic-aware
whitespace tokenization (punctuation such as ። ፣ ፤ is split off and removed;
no stemming, since no validated Amharic stemmer is used). sacreBLEU is run with
tokenize="none" on the same pre-tokenized text.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

_AM_PUNCT = re.compile(r"[\u1360-\u1368!-/:-@\[-`{-~«»“”‘’…]+")


# ----------------------------------------------------------------- classification
def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])} for i in idx},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }


def expected_calibration_error(probs: np.ndarray, y_true, n_bins: int = 15) -> dict:
    """Top-label ECE with equal-width bins (Guo et al., 2017), plus Brier score."""
    probs, y_true = np.asarray(probs, dtype=float), np.asarray(y_true)
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            bins.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                         "acc": float(correct[m].mean()), "conf": float(conf[m].mean())})
    onehot = np.eye(probs.shape[1])[y_true]
    return {"ece": float(ece), "brier": float(((probs - onehot) ** 2).sum(1).mean()),
            "mean_confidence": float(conf.mean()), "bins": bins}


# ---------------------------------------------------------------------- NLG
def am_tokenize(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text or "")
    return _AM_PUNCT.sub(" ", t).lower().split()


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _prf(overlap: int, n_pred: int, n_ref: int) -> float:
    if overlap == 0 or n_pred == 0 or n_ref == 0:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_scores(pred: str, ref: str) -> dict[str, float]:
    """ROUGE-1/2/L F1 for one pair."""
    p, r = am_tokenize(pred), am_tokenize(ref)
    out = {}
    for n in (1, 2):
        pn, rn = _ngrams(p, n), _ngrams(r, n)
        out[f"rouge{n}"] = _prf(sum((pn & rn).values()), sum(pn.values()), sum(rn.values()))
    out["rougeL"] = _prf(_lcs(p, r), len(p), len(r))
    return out


def corpus_bleu(preds: list[str], refs: list[str]) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_bleu([" ".join(am_tokenize(x)) for x in preds],
                                       [[" ".join(am_tokenize(x)) for x in refs]],
                                       tokenize="none").score)


def sentence_bleu(pred: str, ref: str) -> float:
    import sacrebleu

    return float(sacrebleu.sentence_bleu(" ".join(am_tokenize(pred)), [" ".join(am_tokenize(ref))],
                                         tokenize="none").score)


def bertscore_f1(preds: list[str], refs: list[str], model_type: str, batch_size: int = 16) -> list[float]:
    from bert_score import score

    _, _, f = score(preds, refs, model_type=model_type, batch_size=batch_size,
                    rescale_with_baseline=False, verbose=False)
    return f.tolist()


def summarization_metrics(preds: list[str], refs: list[str], bertscore_model: str | None) -> tuple[dict, dict]:
    """Return (corpus-level summary, per-document scores)."""
    per = {"rouge1": [], "rouge2": [], "rougeL": [], "sent_bleu": []}
    for p, r in zip(preds, refs):
        for k, v in rouge_scores(p, r).items():
            per[k].append(v)
        per["sent_bleu"].append(sentence_bleu(p, r))
    if bertscore_model:
        per["bertscore_f1"] = bertscore_f1(preds, refs, bertscore_model)
    summary = {k: float(np.mean(v)) for k, v in per.items()}
    summary["sacrebleu"] = corpus_bleu(preds, refs)
    summary["mean_pred_tokens"] = float(np.mean([len(am_tokenize(p)) for p in preds]))
    summary["empty_pred_rate"] = float(np.mean([len(am_tokenize(p)) == 0 for p in preds]))
    return summary, per


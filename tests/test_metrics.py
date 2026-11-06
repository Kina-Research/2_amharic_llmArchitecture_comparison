"""Tests for src.evaluation.metrics – classification, calibration, NLG scoring."""
import numpy as np
from src.evaluation.metrics import (
    classification_metrics, expected_calibration_error,
    rouge_scores, am_tokenize, corpus_bleu, sentence_bleu,
)


# ──────────────────────────────────── am_tokenize ───────────────────────────
def test_am_tokenize():
    toks = am_tokenize("ሰላም፣ እንዴት ነዎት?")
    assert "ሰላም" in toks
    assert "እንዴት" in toks
    assert "ነዎት" in toks
    assert "፣" not in toks
    assert "?" not in toks


def test_am_tokenize_empty():
    assert am_tokenize("") == []
    assert am_tokenize(None) == []


def test_am_tokenize_punctuation_only():
    assert am_tokenize("፣ ። ፤") == []


# ──────────────────────────────────── rouge_scores ──────────────────────────
def test_rouge_scores():
    pred = "የአዲስ አበባ ከተማ አስተዳደር ታሪፍ አሻሻለ።"
    ref = "የአዲስ አበባ ከተማ አስተዳደር አዲስ ታሪፍ ይፋ አደረገ።"
    scores = rouge_scores(pred, ref)
    assert scores["rouge1"] > 0.4
    assert scores["rouge2"] > 0.2
    assert scores["rougeL"] > 0.4


def test_rouge_scores_identical():
    text = "ሰላም ዓለም"
    scores = rouge_scores(text, text)
    assert scores["rouge1"] == 1.0
    assert scores["rouge2"] == 1.0
    assert scores["rougeL"] == 1.0


def test_rouge_scores_no_overlap():
    scores = rouge_scores("ሰላም", "ዓለም")
    assert scores["rouge1"] == 0.0


def test_rouge_scores_empty():
    scores = rouge_scores("", "ሰላም ዓለም")
    assert scores["rouge1"] == 0.0
    assert scores["rougeL"] == 0.0


# ──────────────────────────────────── BLEU ──────────────────────────────────
def test_sentence_bleu_identical():
    text = "ሰላም ዓለም እንዴት ነዎት"
    score = sentence_bleu(text, text)
    assert score > 50.0  # sacreBLEU returns 0–100


def test_corpus_bleu_basic():
    # BLEU-4 needs >= 4 tokens per sentence, otherwise the 4-gram precision is 0.
    preds = ["የአዲስ አበባ ከተማ አስተዳደር አዲስ ታሪፍ ይፋ አደረገ"] * 3
    refs = ["የአዲስ አበባ ከተማ አስተዳደር አዲስ ታሪፍ ይፋ አደረገ"] * 3
    score = corpus_bleu(preds, refs)
    assert score > 50.0


# ──────────────────────────────────── classification ────────────────────────
def test_classification_metrics():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 1]
    labels = ["pos", "neu", "neg"]
    m = classification_metrics(y_true, y_pred, labels)
    assert m["accuracy"] == 4 / 6
    assert 0.0 < m["macro_f1"] <= 1.0
    assert len(m["confusion_matrix"]) == 3
    assert len(m["per_class"]) == 3


def test_classification_metrics_perfect():
    y = [0, 1, 2, 0, 1]
    m = classification_metrics(y, y, ["a", "b", "c"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


# ──────────────────────────────────── ECE ───────────────────────────────────
def test_expected_calibration_error():
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.9, 0.0],
        [0.2, 0.3, 0.5]
    ])
    y_true = np.array([0, 1, 2])
    ece = expected_calibration_error(probs, y_true, n_bins=5)
    assert 0.0 <= ece["ece"] <= 1.0
    assert "brier" in ece
    assert "mean_confidence" in ece


def test_ece_perfect_calibration():
    # When confidence matches accuracy exactly, ECE should be near 0.
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y_true = np.array([0, 1, 0])
    ece = expected_calibration_error(probs, y_true, n_bins=5)
    assert ece["ece"] < 0.05

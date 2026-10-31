import numpy as np
from src.evaluation.metrics import classification_metrics, expected_calibration_error, rouge_scores, am_tokenize

def test_am_tokenize():
    toks = am_tokenize("ሰላም፣ እንዴት ነዎት?")
    assert "ሰላም" in toks
    assert "እንዴት" in toks
    assert "ነዎት" in toks
    assert "፣" not in toks
    assert "?" not in toks

def test_rouge_scores():
    pred = "የአዲስ አበባ ከተማ አስተዳደር ታሪፍ አሻሻለ።"
    ref = "የአዲስ አበባ ከተማ አስተዳደር አዲስ ታሪፍ ይፋ አደረገ።"
    scores = rouge_scores(pred, ref)
    assert scores["rouge1"] > 0.4
    assert scores["rouge2"] > 0.2
    assert scores["rougeL"] > 0.4

def test_classification_metrics():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 1]
    labels = ["pos", "neu", "neg"]
    m = classification_metrics(y_true, y_pred, labels)
    assert m["accuracy"] == 4 / 6
    assert 0.0 < m["macro_f1"] <= 1.0

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

"""Tests for src.evaluation.statistical_tests and src.utils."""
import numpy as np
import pytest
from src.evaluation.statistical_tests import (
    bootstrap_ci, mcnemar, paired_bootstrap, wilcoxon, holm, pairwise,
)
from src.utils import load_config, sha256_file, REPO_ROOT, MODEL_KEYS, ARCH_LABEL


# ──────────────────────────────────── bootstrap_ci ──────────────────────────
def test_bootstrap_ci_returns_expected_keys():
    ci = bootstrap_ci(lambda idx: float(idx.mean()), n=100, n_boot=200, seed=42)
    for key in ("point", "lo", "hi", "std"):
        assert key in ci
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_bootstrap_ci_deterministic():
    ci1 = bootstrap_ci(lambda idx: float(idx.mean()), n=50, n_boot=100, seed=42)
    ci2 = bootstrap_ci(lambda idx: float(idx.mean()), n=50, n_boot=100, seed=42)
    assert ci1 == ci2


# ──────────────────────────────────── mcnemar ───────────────────────────────
def test_mcnemar_degenerate():
    y = np.array([0, 1, 0, 1])
    r = mcnemar(y, y, y)
    assert r["p_value"] == 1.0
    assert r["method"] == "degenerate"


def test_mcnemar_small_sample():
    y = np.array([0, 1, 0, 1, 0])
    a = np.array([0, 1, 0, 0, 0])
    b = np.array([0, 0, 0, 1, 0])
    r = mcnemar(y, a, b)
    assert r["method"] == "exact"  # n01+n10 < 25
    assert 0.0 <= r["p_value"] <= 1.0


def test_mcnemar_large_sample():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 3, 200)
    a = y.copy(); a[:30] = (a[:30] + 1) % 3
    b = y.copy(); b[20:60] = (b[20:60] + 1) % 3
    r = mcnemar(y, a, b)
    assert r["method"] == "chi2_cc"
    assert 0.0 <= r["p_value"] <= 1.0


# ──────────────────────────────────── paired_bootstrap ──────────────────────
def test_paired_bootstrap():
    rng = np.random.default_rng(42)
    a_scores = rng.normal(0.7, 0.1, 100)
    b_scores = rng.normal(0.65, 0.1, 100)

    def metric(idx, which):
        return float((a_scores if which == "a" else b_scores)[idx].mean())

    r = paired_bootstrap(metric, n=100, n_boot=500, seed=42)
    assert "delta" in r
    assert "p_value" in r
    assert r["lo"] <= r["delta"] <= r["hi"]


# ──────────────────────────────────── wilcoxon ──────────────────────────────
def test_wilcoxon_identical():
    a = np.array([1.0, 2.0, 3.0])
    r = wilcoxon(a, a)
    assert r["p_value"] == 1.0


def test_wilcoxon_different():
    a = np.arange(20, dtype=float)
    b = a + 1.0
    r = wilcoxon(a, b)
    assert r["p_value"] < 0.05


# ──────────────────────────────────── holm ──────────────────────────────────
def test_holm_correction():
    pvals = {"a": 0.01, "b": 0.04, "c": 0.03}
    adj = holm(pvals)
    assert adj["a"] <= adj["c"] <= adj["b"]
    # All adjusted p-values >= unadjusted
    for k in pvals:
        assert adj[k] >= pvals[k]


def test_holm_single():
    adj = holm({"only": 0.03})
    assert adj["only"] == 0.03


# ──────────────────────────────────── pairwise ──────────────────────────────
def test_pairwise():
    assert pairwise(["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert pairwise(["x"]) == []


# ──────────────────────────────────── utils ─────────────────────────────────
def test_load_config():
    cfg = load_config()
    assert "experiment" in cfg
    assert "datasets" in cfg
    assert "models" in cfg
    assert "lora" in cfg


def test_model_keys():
    assert set(MODEL_KEYS) == {"mbert", "mt5", "bloomz"}


def test_arch_label():
    assert ARCH_LABEL["mbert"] == "Encoder-only"
    assert ARCH_LABEL["mt5"] == "Encoder-Decoder"
    assert ARCH_LABEL["bloomz"] == "Decoder-only"


def test_sha256_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h = sha256_file(f)
    assert len(h) == 64
    assert h == sha256_file(f)  # Deterministic


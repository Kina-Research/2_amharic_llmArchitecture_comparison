"""Tests for src.bias – lexicon, swapping, CDA, example weights, probes."""
import numpy as np
import pandas as pd
from src.bias.lexicon import swap_map, bias_terms, GENDER_PAIRS, PROFESSION_PAIRS, CULTURAL_GROUPS
from src.bias.mitigation import swap_terms, contains_bias_term, counterfactual_augment, example_weights


# ──────────────────────────────────── lexicon ───────────────────────────────
def test_bias_lexicon():
    swaps = swap_map()
    assert "እሱ" in swaps
    assert swaps["እሱ"] == "እሷ"
    assert swaps["እሷ"] == "እሱ"
    assert "ዶክተር" in swaps
    assert swaps["ዶክተር"] == "ነርስ"


def test_bias_lexicon_symmetry():
    """Every pair should be bidirectional."""
    swaps = swap_map()
    for a, b in GENDER_PAIRS + PROFESSION_PAIRS:
        assert swaps[a] == b
        assert swaps[b] == a


def test_bias_terms():
    terms = bias_terms()
    assert "እሱ" in terms
    assert "እሷ" in terms
    assert "ዶክተር" in terms
    assert "ነርስ" in terms
    assert len(terms) == 2 * (len(GENDER_PAIRS) + len(PROFESSION_PAIRS))


def test_cultural_groups_non_empty():
    for cat, groups in CULTURAL_GROUPS.items():
        assert len(groups) >= 3, f"Category {cat} has too few groups"


# ──────────────────────────────────── swap_terms ────────────────────────────
def test_swap_terms():
    text = "እሱ ጎበዝ ዶክተር ነው።"
    swapped, count = swap_terms(text)
    assert count == 2
    assert "እሷ" in swapped
    assert "ነርስ" in swapped


def test_swap_terms_no_swappable():
    text = "ዛሬ ፀሐይ ወጥቷል"
    swapped, count = swap_terms(text)
    assert count == 0
    assert swapped == text


def test_swap_terms_with_proclitic():
    text = "የእሱ ቤት ነው"
    swapped, count = swap_terms(text)
    assert count == 1
    assert "የእሷ" in swapped


# ──────────────────────────────────── contains_bias_term ────────────────────
def test_contains_bias_term():
    assert contains_bias_term("የእሱ መጽሐፍ ነው")
    assert not contains_bias_term("ዛሬ ዝናብ እየዘነበ ነው")


# ──────────────────────────────────── counterfactual_augment ────────────────
def test_counterfactual_augment():
    df = pd.DataFrame({
        "id": [f"sent-{i}" for i in range(10)],
        "text": ["እሱ ጥሩ ሰው ነው"] * 5 + ["ዛሬ ፀሐይ ወጥቷል"] * 5,
        "label": ["positive"] * 10
    })
    aug_df, stats = counterfactual_augment(df, target_expansion=0.3, seed=42)
    assert len(aug_df) >= len(df)
    assert stats["realised_expansion"] > 0
    assert "aug" in aug_df.columns
    assert set(aug_df["aug"].unique()) <= {"orig", "swap", "template"}


def test_counterfactual_augment_preserves_originals():
    df = pd.DataFrame({
        "id": ["s0", "s1"],
        "text": ["እሱ ጥሩ ሰው ነው", "ዛሬ ፀሐይ ወጥቷል"],
        "label": ["positive", "negative"]
    })
    aug_df, _ = counterfactual_augment(df, target_expansion=0.5, seed=42)
    orig = aug_df[aug_df["aug"] == "orig"]
    assert len(orig) == len(df)


# ──────────────────────────────────── example_weights ───────────────────────
def test_example_weights():
    texts = ["እሱ ጥሩ ነው", "ዛሬ ፀሐይ ወጥቷል"]
    w = example_weights(texts, factor=2.0)
    assert w[0] == 2.0  # Contains bias term
    assert w[1] == 1.0  # No bias term


def test_example_weights_shape():
    texts = ["a", "b", "c"]
    w = example_weights(texts, factor=3.0)
    assert isinstance(w, np.ndarray)
    assert w.shape == (3,)
    assert w.dtype == np.float32

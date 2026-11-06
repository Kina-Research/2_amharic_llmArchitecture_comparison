"""Tests for src.preprocessing – text cleaning, dedup, filtering, leakage."""
import numpy as np
import pandas as pd
import pytest
from src.preprocessing.text_cleaner import (
    clean_text, dedup_key, ethiopic_ratio, is_malformed,
    clean_frame, verify_no_leakage, CleaningReport,
)


# ──────────────────────────────────── clean_text ────────────────────────────
def test_clean_text():
    sample = "ይህ ምርጥ ዜና ነው! https://t.co/xyz123 @someone &amp; ሠላም  \u200b ዓለም"
    cleaned = clean_text(sample)
    assert "https" not in cleaned
    assert "@someone" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "ይህ" in cleaned
    assert "ዓለም" in cleaned


def test_clean_text_preserves_mentions_when_disabled():
    text = "@user ሰላም"
    assert "@user" in clean_text(text, remove_mentions=False)
    assert "@user" not in clean_text(text, remove_mentions=True)


def test_clean_text_empty_and_non_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_clean_text_nfc_normalization():
    # Composed vs decomposed should produce same result.
    import unicodedata
    composed = "ቤት"
    decomposed = unicodedata.normalize("NFD", composed)
    assert clean_text(composed) == clean_text(decomposed)


def test_clean_text_html_entity():
    assert "&" in clean_text("ሰላም &amp; ዓለም")
    assert "<" in clean_text("ሰላም &lt; ዓለም")


def test_clean_text_multiple_whitespace():
    assert "  " not in clean_text("ሰላም    ዓለም")


# ──────────────────────────────────── dedup_key ─────────────────────────────
def test_dedup_key():
    t1 = "ሰላም፣ ዓለም!"
    t2 = "ሰላም ዓለም"
    assert dedup_key(t1) == dedup_key(t2)


def test_dedup_key_ignores_digits():
    assert dedup_key("ሰላም123") == dedup_key("ሰላም456")


# ──────────────────────────────────── ethiopic_ratio ────────────────────────
def test_ethiopic_ratio():
    am_text = "አዲስ አበባ"
    en_text = "Addis Ababa"
    assert ethiopic_ratio(am_text) == 1.0
    assert ethiopic_ratio(en_text) == 0.0


def test_ethiopic_ratio_empty():
    assert ethiopic_ratio("") == 0.0
    assert ethiopic_ratio("   ") == 0.0


def test_ethiopic_ratio_mixed():
    r = ethiopic_ratio("ሰላም hello")
    assert 0.0 < r < 1.0


# ──────────────────────────────────── is_malformed ──────────────────────────
def test_is_malformed_short():
    assert is_malformed("ab", min_chars=5)
    assert not is_malformed("ሰላም ዓለም", min_chars=5)


def test_is_malformed_low_ethiopic():
    assert is_malformed("hello world 123", min_chars=3)


# ──────────────────────────────────── clean_frame ───────────────────────────
def test_clean_frame_and_leakage():
    df = pd.DataFrame({
        "text": ["ሰላም ጓደኞቼ!", "ሰላም ጓደኞቼ!", "abc", "አጭር", "የኢትዮጵያ ታሪክ አስደናቂ ነው።"],
        "label": ["positive", "positive", "neutral", "neutral", "positive"]
    })
    cleaned = clean_frame(df, ["text"], {"text": 5})
    assert len(cleaned) == 2  # "abc"/"አጭር" filtered, duplicate dropped

    splits = {
        "train": pd.DataFrame({"text": ["ሰላም ጓደኞቼ!"]}),
        "val": pd.DataFrame({"text": ["የኢትዮጵያ ታሪክ አስደናቂ ነው።"]})
    }
    overlaps = verify_no_leakage(splits, "text")
    assert overlaps["train&val"] == 0


def test_clean_frame_drops_conflicting_labels():
    df = pd.DataFrame({
        "text": ["ሰላም ዓለም", "ሰላም ዓለም", "ጥሩ ቀን"],
        "label": ["positive", "negative", "positive"]
    })
    report = CleaningReport()
    cleaned = clean_frame(df, ["text"], {"text": 3}, report,
                          drop_conflicting_labels="label")
    # "ሰላም ዓለም" has conflicting labels -> both copies dropped
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["text"].startswith("ጥሩ")


def test_verify_no_leakage_raises_on_overlap():
    splits = {
        "train": pd.DataFrame({"text": ["ሰላም ዓለም"]}),
        "val": pd.DataFrame({"text": ["ሰላም ዓለም"]})
    }
    with pytest.raises(AssertionError, match="leakage"):
        verify_no_leakage(splits, "text")


# ──────────────────────────────────── CleaningReport ────────────────────────
def test_cleaning_report():
    report = CleaningReport()
    report.log("step1", 100, 95)
    report.log("step2", 95, 90)
    df = report.to_frame()
    assert len(df) == 2
    assert df.iloc[0]["removed"] == 5

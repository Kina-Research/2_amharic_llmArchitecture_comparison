import numpy as np
import pandas as pd
from src.preprocessing.text_cleaner import clean_text, dedup_key, ethiopic_ratio, is_malformed, clean_frame, verify_no_leakage

def test_clean_text():
    sample = "ይህ ምርጥ ዜና ነው! https://t.co/xyz123 @someone &amp; ሠላም  \u200b ዓለም"
    cleaned = clean_text(sample)
    assert "https" not in cleaned
    assert "@someone" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "ይህ" in cleaned
    assert "ዓለም" in cleaned

def test_dedup_key():
    t1 = "ሰላም፣ ዓለም!"
    t2 = "ሰላም ዓለም"
    assert dedup_key(t1) == dedup_key(t2)

def test_ethiopic_ratio():
    am_text = "አዲስ አበባ"
    en_text = "Addis Ababa"
    assert ethiopic_ratio(am_text) == 1.0
    assert ethiopic_ratio(en_text) == 0.0

def test_clean_frame_and_leakage():
    df = pd.DataFrame({
        "text": ["ሰላም ጓደኞቼ!", "ሰላም ጓደኞቼ!", "abc", "አጭር", "የኢትዮጵያ ታሪክ አስደናቂ ነው።"],
        "label": ["positive", "positive", "neutral", "neutral", "positive"]
    })
    cleaned = clean_frame(df, ["text"], {"text": 5})
    assert len(cleaned) == 2 # "abc" and "አጭር" filtered (< 5 chars or low ethiopic), duplicate dropped
    
    splits = {
        "train": pd.DataFrame({"text": ["ሰላም ጓደኞቼ!"]}),
        "val": pd.DataFrame({"text": ["የኢትዮጵያ ታሪክ አስደናቂ ነው።"]})
    }
    overlaps = verify_no_leakage(splits, "text")
    assert overlaps["train&val"] == 0

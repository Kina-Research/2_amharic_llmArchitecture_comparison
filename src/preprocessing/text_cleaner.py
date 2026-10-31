"""General preprocessing shared by both tasks (spec §2).

Steps, applied in this order:
  1. Unicode NFC normalisation
  2. URL removal (+ @mentions / HTML entities, which are social-media noise)
  3. Whitespace standardisation
  4. Deduplication (on a normalised key, so trivially different copies collapse)
  5. Filtering of malformed / extremely short samples
  6. Leakage verification between splits

Every function is pure and deterministic.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")
ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")
# Ethiopic + ASCII punctuation, removed only for the dedup key.
PUNCT_RE = re.compile(r"[\u1360-\u1368\.,!?;:\"'()\[\]{}«»“”‘’\-_/\\|#*~`<>]+")


def clean_text(text: str, remove_mentions: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", html.unescape(text))
    t = URL_RE.sub(" ", t)
    if remove_mentions:
        t = MENTION_RE.sub(" ", t)
    t = t.replace("\u200b", " ").replace("\ufeff", " ")
    return WS_RE.sub(" ", t).strip()


def dedup_key(text: str) -> str:
    """Key used for deduplication and leakage checks: no punctuation/space/digits."""
    t = PUNCT_RE.sub("", text.lower())
    t = re.sub(r"[\d\s]+", "", t)
    return t


def ethiopic_ratio(text: str) -> float:
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return 0.0
    return sum(bool(ETHIOPIC_RE.match(c)) for c in letters) / len(letters)


def is_malformed(text: str, min_chars: int, min_ethiopic_ratio: float = 0.3) -> bool:
    """Too short, or not predominantly Ge'ez script (e.g. emoji-only / Latin-only)."""
    return len(text) < min_chars or ethiopic_ratio(text) < min_ethiopic_ratio


@dataclass
class CleaningReport:
    steps: list[dict] = field(default_factory=list)

    def log(self, step: str, before: int, after: int) -> None:
        self.steps.append({"step": step, "before": before, "after": after, "removed": before - after})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def clean_frame(df: pd.DataFrame, text_cols: list[str], min_chars: dict[str, int],
                report: CleaningReport | None = None, dedup_on: str | None = None,
                drop_conflicting_labels: str | None = None) -> pd.DataFrame:
    """Apply steps 1–5 to the given columns of a DataFrame.

    drop_conflicting_labels: name of a label column; if the same normalised
    text appears with different labels, *all* copies are dropped (they cannot
    be resolved and would otherwise be a source of noise/leakage).
    """
    report = report or CleaningReport()
    n = len(df)
    df = df.copy()
    for c in text_cols:
        df[c] = df[c].map(clean_text)
    report.log("nfc+url+whitespace", n, len(df))

    n = len(df)
    mask = pd.Series(True, index=df.index)
    for c in text_cols:
        mask &= ~df[c].map(lambda t, m=min_chars[c]: is_malformed(t, m))
    df = df[mask]
    report.log("filter_malformed_or_short", n, len(df))

    key_col = dedup_on or text_cols[0]
    df["_key"] = df[key_col].map(dedup_key)
    if drop_conflicting_labels:
        n = len(df)
        n_labels = df.groupby("_key")[drop_conflicting_labels].transform("nunique")
        df = df[n_labels == 1]
        report.log("drop_conflicting_label_duplicates", n, len(df))
    n = len(df)
    df = df.drop_duplicates("_key", keep="first")
    report.log("deduplicate", n, len(df))
    return df.drop(columns="_key").reset_index(drop=True)


def verify_no_leakage(splits: dict[str, pd.DataFrame], col: str) -> dict[str, int]:
    """Return pairwise overlap counts (on the dedup key). All must be 0."""
    keys = {name: set(df[col].map(dedup_key)) for name, df in splits.items()}
    names = list(keys)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}&{b}"] = len(keys[a] & keys[b])
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise AssertionError(f"Cross-split leakage detected: {bad}")
    return overlaps


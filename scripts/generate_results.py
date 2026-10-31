#!/usr/bin/env python
"""Regenerate ALL tables (CSV + Markdown) and figures (PNG + PDF) from results/raw/.

    python scripts/generate_results.py [--results-dir results]

CPU-only; needs no GPU, model weights, or network.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.report import build_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    print(json.dumps(build_all(a.results_dir, a.config), indent=2, default=str))


#!/usr/bin/env python
"""Step 1: download, clean, split and freeze both datasets -> data/processed/."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.data_prep import prepare_all  # noqa: E402
from src.utils import load_config, set_all_seeds  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["experiment"]["seeds"][0]
    set_all_seeds(seed)
    m = prepare_all(cfg, seed)
    print(json.dumps({k: m[k] for k in ("sentiment", "summarization")}, ensure_ascii=False, indent=2,
                     default=str)[:4000])
    print("\nWrote", cfg["experiment"]["processed_data_dir"])


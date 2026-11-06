#!/usr/bin/env python
"""Full pipeline CLI.

Examples
  python scripts/run_experiment.py --stage all                       # everything (spec settings)
  python scripts/run_experiment.py --stage sentiment --models mbert --conditions lora
  python scripts/run_experiment.py --stage summarization --models mt5
  python scripts/run_experiment.py --stage smoke                     # 1-epoch tiny run to test the stack

Each (task, model, condition, seed) run is skipped if its metrics.json already
exists, so the command is resumable after a Colab disconnect (use --force to redo).
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import MODEL_KEYS, load_config, run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["data", "sentiment", "summarization", "results", "all", "smoke"],
                    default="all")
    ap.add_argument("--models", nargs="+", default=list(MODEL_KEYS), choices=MODEL_KEYS)
    ap.add_argument("--conditions", nargs="+", default=None, help="sentiment conditions (default: config)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (debug only)")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg["experiment"]["seeds"]
    conds = args.conditions or cfg["bias"]["conditions"]
    smoke = args.stage == "smoke"
    if smoke:
        # Writes to a separate directory so smoke numbers can never be mistaken for results.
        cfg["experiment"]["results_dir"] = "results_smoke"
        cfg["evaluation"]["latency_samples"], cfg["bias"]["n_positional_samples"] = 10, 100
        args.epochs, conds = 1, ["lora"]

    if args.stage in ("data", "all") or (smoke and not Path(cfg["experiment"]["processed_data_dir"],
                                                           "manifest.json").exists()):
        from src.preprocessing.data_prep import prepare_all
        prepare_all(cfg, seeds[0])

    failures = []
    if args.stage in ("sentiment", "all", "smoke"):
        from src.training.sentiment import run_sentiment
        for seed in seeds:
            for m in args.models:
                for c in conds:
                    if (run_dir(cfg["experiment"]["results_dir"], "sentiment", m, c, seed) / "metrics.json").exists() \
                            and not args.force:
                        print(f"skip sentiment/{m}/{c}/seed{seed} (exists)")
                        continue
                    try:
                        run_sentiment(cfg, m, c, seed, args.save_models, args.epochs, 50 if smoke else None)
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()
                        failures.append(f"sentiment/{m}/{c}/seed{seed}")

    if args.stage in ("summarization", "all", "smoke"):
        from src.training.summarization import run_summarization
        if smoke:
            cfg["datasets"]["summarization"]["eval_size"] = 500
        for seed in seeds:
            for m in args.models:
                if (run_dir(cfg["experiment"]["results_dir"], "summarization", m, "lora", seed) / "metrics.json").exists() \
                        and not args.force:
                    print(f"skip summarization/{m}/seed{seed} (exists)")
                    continue
                try:
                    run_summarization(cfg, m, seed, args.save_models, args.epochs, 200 if smoke else None)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
                    failures.append(f"summarization/{m}/seed{seed}")

    if args.stage in ("results", "all", "smoke"):
        from src.evaluation.report import build_all
        print(build_all(cfg["experiment"]["results_dir"], args.config))

    if failures:
        print("\nFAILED RUNS:", *failures, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()


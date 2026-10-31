"""Download, clean, split and freeze both datasets (spec §1–2).

Outputs (in data/processed/):
  sentiment_train.csv, sentiment_val.csv          columns: id, text, label
  summ_train.csv, summ_eval500.csv                columns: id, source, target
  cleaning_report_*.csv, manifest.json            provenance + sha256 of every file

Run:  python scripts/prepare_data.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.text_cleaner import CleaningReport, clean_frame, verify_no_leakage
from src.utils import capture_environment, save_json, sha256_file


def _load_first_available(candidates: list[dict]) -> tuple[Any, dict]:
    """Try each HF dataset candidate in order; return (DatasetDict, provenance)."""
    from datasets import load_dataset

    errors = []
    for cand in candidates:
        try:
            ds = load_dataset(cand["path"], cand.get("name"), trust_remote_code=True)
            prov = {"hf_path": cand["path"], "hf_config": cand.get("name"),
                    "splits": {k: len(v) for k, v in ds.items()}}
            try:
                from huggingface_hub import HfApi

                prov["hf_revision_sha"] = HfApi().dataset_info(cand["path"]).sha
            except Exception as e:  # noqa: BLE001
                prov["hf_revision_sha"] = f"unavailable ({type(e).__name__})"
            return ds, prov
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cand}: {type(e).__name__}: {e}")
    raise RuntimeError("No dataset candidate could be loaded:\n" + "\n".join(errors))


def _label_to_str(ds_split, label_col: str) -> list[str]:
    feat = ds_split.features[label_col]
    vals = ds_split[label_col]
    if hasattr(feat, "names"):
        return [feat.names[v].lower() for v in vals]
    return [str(v).lower().strip() for v in vals]


def prepare_sentiment(cfg: dict, out_dir: Path, seed: int) -> dict:
    scfg = cfg["datasets"]["sentiment"]
    ds, prov = _load_first_available(scfg["hf_candidates"])
    frames = []
    for split_name, split in ds.items():
        text_col = scfg["text_column"] if scfg["text_column"] in split.column_names else "text"
        frames.append(pd.DataFrame({
            "text": split[text_col],
            "label": _label_to_str(split, scfg["label_column"]),
            "orig_split": split_name,
        }))
    raw = pd.concat(frames, ignore_index=True)
    labels = scfg["labels"]
    unknown = set(raw["label"]) - set(labels)
    if unknown:
        raise ValueError(f"Unexpected sentiment labels {unknown}; expected {labels}")

    report = CleaningReport()
    report.log("raw_pooled_official_splits", len(raw), len(raw))
    df = clean_frame(raw, ["text"], {"text": scfg["min_chars"]}, report,
                     drop_conflicting_labels="label")
    df.insert(0, "id", [f"sent-{i:05d}" for i in range(len(df))])
    train, val = train_test_split(df, test_size=scfg["val_ratio"], stratify=df["label"],
                                  random_state=seed, shuffle=True)
    train, val = train.sort_values("id"), val.sort_values("id")
    overlaps = verify_no_leakage({"train": train, "val": val}, "text")

    cols = ["id", "text", "label", "orig_split"]
    train[cols].to_csv(out_dir / "sentiment_train.csv", index=False, encoding="utf-8")
    val[cols].to_csv(out_dir / "sentiment_val.csv", index=False, encoding="utf-8")
    report.to_frame().to_csv(out_dir / "cleaning_report_sentiment.csv", index=False)
    return {
        "provenance": prov,
        "cleaning": report.steps,
        "n_train": len(train), "n_val": len(val),
        "label_dist_train": train["label"].value_counts().to_dict(),
        "label_dist_val": val["label"].value_counts().to_dict(),
        "leakage_overlaps": overlaps,
    }


def prepare_summarization(cfg: dict, out_dir: Path, seed: int) -> dict:
    scfg = cfg["datasets"]["summarization"]
    ds, prov = _load_first_available(scfg["hf_candidates"])
    src, tgt = scfg["source_column"], scfg["target_column"]
    mins = {"source": scfg["min_source_chars"], "target": scfg["min_target_chars"]}

    def frame(split) -> pd.DataFrame:
        return pd.DataFrame({"source": split[src], "target": split[tgt],
                             "url": split["url"] if "url" in split.column_names else None})

    rep_test, rep_train = CleaningReport(), CleaningReport()
    test = clean_frame(frame(ds["test"]), ["source", "target"], mins, rep_test, dedup_on="source")
    if len(test) < scfg["eval_size"]:
        raise ValueError(f"Only {len(test)} clean test pairs; need {scfg['eval_size']}")
    eval500 = test.sample(n=scfg["eval_size"], random_state=seed).reset_index(drop=True)
    eval500.insert(0, "id", [f"summ-eval-{i:03d}" for i in range(len(eval500))])

    train_parts = [frame(ds[s]) for s in ("train", "validation") if s in ds]
    train = clean_frame(pd.concat(train_parts, ignore_index=True), ["source", "target"], mins,
                        rep_train, dedup_on="source")
    # Remove any train pair whose article OR reference summary appears in eval-500.
    from src.preprocessing.text_cleaner import dedup_key

    ek_src, ek_tgt = set(eval500["source"].map(dedup_key)), set(eval500["target"].map(dedup_key))
    n = len(train)
    train = train[~train["source"].map(dedup_key).isin(ek_src) & ~train["target"].map(dedup_key).isin(ek_tgt)]
    rep_train.log("remove_overlap_with_eval500", n, len(train))
    if scfg.get("train_max"):
        train = train.sample(n=min(scfg["train_max"], len(train)), random_state=seed)
    train = train.reset_index(drop=True)
    train.insert(0, "id", [f"summ-train-{i:05d}" for i in range(len(train))])
    overlaps = {**verify_no_leakage({"train": train, "eval500": eval500}, "source"),
                **{f"{k}(target)": v for k, v in
                   verify_no_leakage({"train": train, "eval500": eval500}, "target").items()}}

    train.to_csv(out_dir / "summ_train.csv", index=False, encoding="utf-8")
    eval500.to_csv(out_dir / "summ_eval500.csv", index=False, encoding="utf-8")
    pd.concat([rep_test.to_frame().assign(split="test"), rep_train.to_frame().assign(split="train")]
              ).to_csv(out_dir / "cleaning_report_summarization.csv", index=False)
    return {"provenance": prov, "cleaning_test": rep_test.steps, "cleaning_train": rep_train.steps,
            "n_train": len(train), "n_eval": len(eval500), "leakage_overlaps": overlaps}


def prepare_all(cfg: dict, seed: int) -> dict:
    out_dir = Path(cfg["experiment"]["processed_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "environment": capture_environment(),
                "sentiment": prepare_sentiment(cfg, out_dir, seed),
                "summarization": prepare_summarization(cfg, out_dir, seed)}
    manifest["sha256"] = {p.name: sha256_file(p) for p in sorted(out_dir.glob("*.csv"))}
    save_json(manifest, out_dir / "manifest.json")
    return manifest


"""Shared utilities: config loading, seeding, environment capture, JSON I/O.

This module deliberately avoids importing torch at module load time so that the
CPU-only parts of the pipeline (preprocessing, statistics, table generation)
work without a deep-learning stack installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_config.yaml"
MODEL_KEYS = ("mbert", "mt5", "bloomz")
ARCH_LABEL = {"mbert": "Encoder-only", "mt5": "Encoder-Decoder", "bloomz": "Decoder-only"}


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU+CUDA) and Hugging Face."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # warn_only: some CUDA kernels have no deterministic variant.
            torch.use_deterministic_algorithms(True, warn_only=True)
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def capture_environment() -> dict[str, Any]:
    """Record software/hardware versions for the run manifest."""
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "datasets", "accelerate", "numpy",
                "sacrebleu", "bert_score", "codecarbon", "sklearn", "scipy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            env[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["gpu"] = p.name
            env["gpu_memory_gb"] = round(p.total_memory / 1024**3, 2)
            env["cuda"] = torch.version.cuda
        else:
            env["gpu"] = None
    except ImportError:
        env["gpu"] = None
    return env


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(results_dir: str | os.PathLike, task: str, model: str, condition: str, seed: int) -> Path:
    """Canonical location of one run's raw outputs."""
    return Path(results_dir) / "raw" / task / model / condition / f"seed{seed}"


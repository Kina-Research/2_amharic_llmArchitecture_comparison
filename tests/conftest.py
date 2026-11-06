# -*- coding: utf-8 -*-
"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable when running pytest from any directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


"""Pytest configuration: force offline/fast mode and make the package importable."""

import os
import sys
from pathlib import Path

# Rule-based lyrics scoring; no torch/transformers, no network.
os.environ.setdefault("MOOD_READER_FAST", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

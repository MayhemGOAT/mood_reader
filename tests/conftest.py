"""Shared pytest configuration for mood_reader tests.

Force lyrics feature extraction into rule-based FAST mode (no torch/transformers
downloads) and make the repo root importable.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MOOD_READER_FAST", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

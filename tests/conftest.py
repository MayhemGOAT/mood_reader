"""Shared pytest configuration for mood_reader tests.

Runs everything in FAST mode (rule-based lyric scoring, no torch/transformers)
and makes the repo root importable regardless of the invocation directory.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MOOD_READER_FAST", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

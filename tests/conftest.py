"""Shared pytest configuration for mood_reader tests."""

import os
import sys
from pathlib import Path

# Force lightweight rule-based lyrics scoring so tests never need torch/network.
os.environ.setdefault("MOOD_READER_FAST", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

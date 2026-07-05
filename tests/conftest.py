"""Shared pytest configuration for mood_reader tests.

Runs in FAST mode (rule-based lyrics scoring, no HuggingFace/torch) so the
suite is fully offline and quick.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOOD_READER_FAST", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

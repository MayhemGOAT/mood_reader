"""Shared pytest fixtures/config for mood_reader tests.

Tests run in FAST mode so lyric emotion scoring uses the rule-based fallback
instead of loading transformer models (no network / heavyweight deps needed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOOD_READER_FAST", "1")

# Ensure the repository root is importable when running `pytest` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

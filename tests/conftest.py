"""Shared pytest configuration for mood_reader tests.

Runs the lyrics pipeline in rule-based FAST mode so tests need no torch /
transformers download and no network access.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOOD_READER_FAST", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

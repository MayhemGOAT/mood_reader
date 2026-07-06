"""Shared pytest configuration for the offline regression suite.

Forces FAST (rule-based) lyrics scoring so tests never need torch/transformers
or network access, and makes the repository root importable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOOD_READER_FAST", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

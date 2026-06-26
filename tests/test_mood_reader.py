"""Regression tests for the mood_reader training + data pipelines.

All tests run offline with MOOD_READER_FAST=1 (set in conftest.py), so no
network access, torch, or transformers download is required.
"""

from __future__ import annotations

import ast
import collections
import json
from pathlib import Path

import pandas as pd
import pytest

from mood_reader.kaggle_merge import _merge_key, _normalize
from mood_reader.train import build_feature_matrix, train

ROOT = Path(__file__).resolve().parents[1]

_VIBE_LYRICS = {
    "happy": "happy bright sunshine joy smile laugh fun dance",
    "melancholic": "sad lonely tears rain cry goodbye missing pain",
    "energetic": "dance party night fire wild jump beat loud",
    "aggressive": "fight rage hate burn war scream break angry",
}
_VIBE_VE = {
    "happy": (0.85, 0.70),
    "melancholic": (0.20, 0.30),
    "energetic": (0.72, 0.90),
    "aggressive": (0.18, 0.88),
}


def _labeled_rows(n: int) -> list[dict]:
    vibes = list(_VIBE_LYRICS)
    rows = []
    for i in range(n):
        vibe = vibes[i % len(vibes)]
        val, eng = _VIBE_VE[vibe]
        rows.append(
            {
                "title": f"S{i}",
                "lyrics": _VIBE_LYRICS[vibe],
                "valence": val,
                "energy": eng,
                "vibe": vibe,
            }
        )
    return rows


def _write_csv(tmp_path: Path, rows: list[dict], name: str = "data.csv") -> Path:
    csv = tmp_path / name
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_train_creates_missing_model_dir_and_valid_json(tmp_path):
    csv = _write_csv(tmp_path, _labeled_rows(20))
    model_dir = tmp_path / "models" / "nested"  # does not exist yet

    report = train(csv, model_dir)

    assert model_dir.is_dir()
    # vibe_classes.json must be real JSON, not a pickle with a .json suffix.
    classes = json.loads((model_dir / "vibe_classes.json").read_text())
    assert isinstance(classes, list)
    assert len(classes) >= 2
    assert report["valence_mae"] is not None


def test_train_handles_partially_labeled_dataset(tmp_path):
    rows = _labeled_rows(16)
    # Lyric-only rows with no valence/energy/vibe (e.g. merge-kaggle --keep-unmatched).
    for i in range(5):
        rows.append({"title": f"U{i}", "lyrics": _VIBE_LYRICS["happy"], "valence": "", "energy": "", "vibe": ""})
    csv = _write_csv(tmp_path, rows)

    report = train(csv, tmp_path / "m")

    # Previously raised "Input y contains NaN"; now trains on labeled subset.
    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None


def test_build_feature_matrix_skips_unusable_lyrics():
    df = pd.DataFrame(
        {
            "lyrics": [
                "happy joy sunshine dance love light",
                "",  # empty
                "[Verse 1]\n[Chorus]",  # cleans down to nothing
                None,  # NaN
            ]
        }
    )

    feats = build_feature_matrix(df)

    assert len(feats) == 1


def test_single_class_vibe_is_skipped(tmp_path):
    rows = [
        {"title": f"S{i}", "lyrics": _VIBE_LYRICS["happy"], "valence": 0.8, "energy": 0.7, "vibe": "happy"}
        for i in range(12)
    ]
    csv = _write_csv(tmp_path, rows)
    model_dir = tmp_path / "m"

    report = train(csv, model_dir)

    # A classifier can't train on a single class; skip it gracefully.
    assert report["vibe_accuracy"] is None
    assert not (model_dir / "vibe_model.joblib").exists()
    # Regressors should still train.
    assert report["valence_mae"] is not None


@pytest.mark.parametrize(
    "accented,plain",
    [
        ("Beyoncé", "Beyonce"),
        ("Rosalía", "Rosalia"),
        ("Måneskin", "Maneskin"),
        ("Céline Dion", "Celine Dion"),
    ],
)
def test_normalize_ascii_folds_accents(accented, plain):
    assert _normalize(accented) == _normalize(plain)
    assert _merge_key(accented, "Song") == _merge_key(plain, "Song")


def test_artist_hits_has_no_duplicate_keys():
    src = (ROOT / "scripts" / "build_diverse_tracks.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    duplicates: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if "Billie Eilish" not in keys:
            continue
        counts = collections.Counter(keys)
        duplicates = sorted(k for k, c in counts.items() if c > 1)
        break

    assert duplicates == []

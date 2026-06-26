"""Regression tests for Mood Reader fixes.

These run on the FAST lyrics path (no transformers/torch) and make no network
calls. Run with:  MOOD_READER_FAST=1 pytest
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("MOOD_READER_FAST", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _lyric(seed: str) -> str:
    """Build a long-enough, unique-ish lyric string for a sample row."""
    return (
        f"{seed} the night is long and the road is bright we keep moving "
        f"through the {seed} light singing every word we know by heart tonight"
    )


def _labeled_df(n: int = 14) -> pd.DataFrame:
    vibes = ["happy", "chill", "dark", "energetic", "romantic", "melancholic", "uplifting"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "title": f"Song {i}",
                "lyrics": _lyric(f"word{i}"),
                "valence": round(0.1 + (i % 9) / 10.0, 3),
                "energy": round(0.2 + (i % 7) / 10.0, 3),
                "vibe": vibes[i % len(vibes)],
            }
        )
    return pd.DataFrame(rows)


def test_train_creates_missing_model_dir(tmp_path):
    """train() must not crash when --model-dir does not yet exist."""
    from mood_reader.train import train

    data_csv = tmp_path / "data.csv"
    _labeled_df().to_csv(data_csv, index=False)

    model_dir = tmp_path / "does" / "not" / "exist"
    report = train(data_csv, model_dir)

    assert model_dir.exists()
    assert (model_dir / "training_features.csv").exists()
    assert report["samples"] >= 10


def test_vibe_classes_written_as_valid_json(tmp_path):
    """vibe_classes.json must be real JSON, not a joblib pickle."""
    from mood_reader.train import train

    data_csv = tmp_path / "data.csv"
    _labeled_df().to_csv(data_csv, index=False)
    model_dir = tmp_path / "models"

    train(data_csv, model_dir)

    vibe_classes_path = model_dir / "vibe_classes.json"
    assert vibe_classes_path.exists()
    with open(vibe_classes_path) as f:
        classes = json.load(f)  # would raise if it were a pickle
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)


def test_partially_labeled_dataset_does_not_crash(tmp_path):
    """Rows with missing valence/energy/vibe must not poison model fitting."""
    from mood_reader.train import train

    df = _labeled_df(14)
    # Blank out labels on a few rows (mixed/partial labeling).
    df.loc[2, ["valence", "energy"]] = pd.NA
    df.loc[5, "vibe"] = pd.NA
    df.loc[9, ["valence", "energy", "vibe"]] = pd.NA

    data_csv = tmp_path / "data.csv"
    df.to_csv(data_csv, index=False)

    report = train(data_csv, tmp_path / "models")
    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None


def test_blank_and_header_only_lyrics_are_skipped(tmp_path):
    """Empty / header-only lyrics rows are skipped instead of crashing."""
    from mood_reader.train import build_feature_matrix

    df = pd.DataFrame(
        {
            "lyrics": [
                _lyric("good"),
                "",
                "   ",
                "[Verse]\n[Chorus]",
                None,
                _lyric("also good"),
            ],
            "valence": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "energy": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )

    feats = build_feature_matrix(df)
    assert len(feats) == 2  # only the two real lyric rows survive


def test_normalize_folds_accents():
    """Accented names must normalize to ASCII so Kaggle joins match."""
    from mood_reader.kaggle_merge import _normalize

    assert _normalize("Beyoncé") == _normalize("Beyonce")
    assert _normalize("Rosalía") == _normalize("Rosalia")
    assert _normalize("Sigur Rós") == _normalize("Sigur Ros")
    assert _normalize(None) == ""
    assert _normalize(float("nan")) == ""


def test_build_diverse_tracks_has_no_duplicate_keys():
    """ARTIST_HITS must not silently drop tracks via duplicate dict keys."""
    src = (REPO_ROOT / "scripts" / "build_diverse_tracks.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and len(node.keys) > 50:
            keys = [k.value for k in node.keys]
            assert len(keys) == len(set(keys)), "duplicate artist keys present"
            return
    pytest.fail("ARTIST_HITS dict not found")

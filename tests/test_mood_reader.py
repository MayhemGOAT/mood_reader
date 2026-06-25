"""Tests for Mood Reader core logic and regression guards for fixed bugs.

The suite runs with the lightweight rule-based feature path
(``MOOD_READER_FAST=1``) so it does not require torch/transformers downloads or
any network access.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MOOD_READER_FAST", "1")

import pandas as pd
import pytest

from mood_reader.fusion import (
    extract_song_features,
    feature_column_names,
    features_to_row,
)
from mood_reader.kaggle_merge import _merge_key, _normalize
from mood_reader.labels import (
    VIBES,
    energy_bucket,
    valence_bucket,
    vibe_from_valence_energy,
)
from mood_reader.lyrics_features import extract_lyrics_features, lyrics_feature_names
from mood_reader.train import build_feature_matrix, train

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO_ROOT / "data" / "sample" / "songs.csv"


def test_buckets_thresholds():
    assert energy_bucket(0.1) == "low"
    assert energy_bucket(0.5) == "medium"
    assert energy_bucket(0.9) == "high"
    assert valence_bucket(0.1) == "negative"
    assert valence_bucket(0.5) == "neutral"
    assert valence_bucket(0.9) == "positive"


def test_vibe_from_valence_energy_returns_known_vibe():
    for valence in (0.0, 0.5, 1.0):
        for energy in (0.0, 0.5, 1.0):
            assert vibe_from_valence_energy(valence, energy) in VIBES


def test_lyrics_features_expected_keys():
    feats = extract_lyrics_features("dance party night, love and joy, fire in the sky")
    for name in lyrics_feature_names():
        assert name in feats, f"missing feature {name}"
    assert feats["lyrics_word_count"] > 0


def test_empty_lyrics_raises():
    with pytest.raises(ValueError):
        extract_lyrics_features("   ")


def test_extract_song_features_fills_audio_with_zero_when_no_audio():
    feats = extract_song_features(lyrics="a calm quiet evening by the lake")
    for name in feature_column_names():
        assert name in feats
    assert feats["tempo_bpm"] == 0.0
    assert feats["mfcc_0_mean"] == 0.0


def test_features_to_row_aligns_with_columns():
    feats = extract_song_features(lyrics="bright sun happy smile laugh")
    row = features_to_row(feats)
    assert list(row.index) == feature_column_names()


def test_build_feature_matrix_skips_missing_lyrics():
    """Regression: empty/NaN lyrics rows must be skipped, not crash or become
    a bogus 'nan' sample."""
    df = pd.DataFrame(
        {
            "lyrics": ["", None, "   ", "dance party night fire love"],
            "vibe": ["happy", "chill", "dark", "energetic"],
        }
    )
    feat_df = build_feature_matrix(df)
    assert len(feat_df) == 1


def test_train_creates_dir_and_writes_valid_json(tmp_path):
    """Regression: train() must create a missing model_dir and write JSON
    (not a pickle) to vibe_classes.json."""
    model_dir = tmp_path / "fresh" / "models"
    assert not model_dir.exists()

    report = train(SAMPLE_CSV, model_dir)

    assert model_dir.exists()
    assert report["samples"] == 30

    classes_path = model_dir / "vibe_classes.json"
    assert classes_path.exists()
    classes = json.loads(classes_path.read_text())  # must be real JSON
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)
    assert set(classes).issubset(set(VIBES))


def test_train_handles_partially_labeled_dataset(tmp_path):
    """Regression: rows missing valence/energy/vibe must be dropped per-target
    instead of raising 'Input y contains NaN'."""
    labeled = pd.read_csv(SAMPLE_CSV)
    unlabeled = pd.DataFrame(
        {
            "track_id": [101, 102, 103],
            "title": ["U1", "U2", "U3"],
            "lyrics": [
                "an unlabeled track with plenty of words but no mood numbers",
                "second unlabeled song just lyrics here and nothing else today",
                "third unlabeled entry only text without any valence or energy",
            ],
            "audio_path": ["", "", ""],
            "vibe": ["", "", ""],
            "valence": [None, None, None],
            "energy": [None, None, None],
        }
    )
    mixed = pd.concat([labeled, unlabeled], ignore_index=True)
    mixed_csv = tmp_path / "mixed.csv"
    mixed.to_csv(mixed_csv, index=False)

    report = train(mixed_csv, tmp_path / "models")
    # Regressors should still train from the labeled rows.
    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None


def test_kaggle_normalize_folds_accents():
    """Regression: accented international names must match unaccented spellings."""
    assert _normalize("Beyoncé") == _normalize("Beyonce")
    assert _normalize("Rosalía") == _normalize("Rosalia")
    assert _normalize("niño") == "nino"
    assert _merge_key("Beyoncé", "Déjà Vu") == _merge_key("Beyonce", "Deja Vu")

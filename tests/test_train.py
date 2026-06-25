"""Regression tests for the training pipeline.

Runs fully offline with the rule-based lyrics scorer (no torch/transformers,
no network) by forcing MOOD_READER_FAST before importing the package.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MOOD_READER_FAST", "1")

from pathlib import Path

import pytest

from mood_reader.train import train

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample" / "songs.csv"


def test_train_creates_nested_model_dir(tmp_path):
    """train() must create a non-existent (nested) --model-dir, not crash."""
    model_dir = tmp_path / "nested" / "models"
    assert not model_dir.exists()

    report = train(SAMPLE_CSV, model_dir)

    assert model_dir.is_dir()
    assert report["samples"] == 30
    assert (model_dir / "metadata.json").exists()
    assert (model_dir / "valence_model.joblib").exists()
    assert (model_dir / "energy_model.joblib").exists()


def test_vibe_classes_artifact_is_valid_json(tmp_path):
    """vibe_classes.json must be real JSON (not a pickle blob)."""
    model_dir = tmp_path / "models"
    train(SAMPLE_CSV, model_dir)

    classes_path = model_dir / "vibe_classes.json"
    assert classes_path.exists()

    with open(classes_path, encoding="utf-8") as f:
        classes = json.load(f)

    assert isinstance(classes, list)
    assert classes == sorted(classes)
    assert all(isinstance(c, str) for c in classes)


def test_committed_vibe_classes_is_valid_json():
    """The artifact checked into models/ must also be valid JSON."""
    committed = Path(__file__).resolve().parents[1] / "models" / "vibe_classes.json"
    with open(committed, encoding="utf-8") as f:
        classes = json.load(f)
    assert isinstance(classes, list)
    assert classes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

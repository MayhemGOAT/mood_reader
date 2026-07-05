"""Regression tests for known mood_reader bugs.

All tests run in MOOD_READER_FAST mode (set in conftest) and require no network.
Each test documents the specific defect it guards against.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mood_reader.train import build_feature_matrix, train, train_from_dataframe

REPO_ROOT = Path(__file__).resolve().parents[1]

# A handful of distinct lyric snippets so VADER/keyword features actually vary
# between rows (constant lyrics would collapse feature variance).
_LYRIC_POOL = [
    "We dance all night, the fire burns bright, pure joy and light",
    "Lonely tears fall, I miss you, the pain of goodbye lingers",
    "Rage and fury, we fight and burn, war in the streets tonight",
    "Rise and shine, chase the dream, we are free to fly so high",
    "Dark shadows creep, blood and hell, the devil calls my name",
    "Baby I love you, my heart is yours, a tender kiss goodnight",
    "Party in the club, wild and loud, dance until the dawn breaks",
    "Quiet morning coffee, soft rain, calm and slow and gentle",
    "Thunder drums and roaring crowds, adrenaline never rests",
    "Reflective and blue, the memory fades, sadness in the rain",
    "Hopeful sunrise, brand new day, everything will be alright",
    "Smooth groove and mellow vibes, laid back easy summer nights",
]

_VIBES = ["happy", "melancholic", "energetic", "chill"]


def _make_dataset(n: int = 24, *, with_vibe: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(n):
        valence = round(0.15 + 0.7 * ((i * 37) % 100) / 100.0, 3)
        energy = round(0.15 + 0.7 * ((i * 53) % 100) / 100.0, 3)
        row = {
            "track_id": f"t{i}",
            "artist": f"Artist {i % 5}",
            "title": f"Song {i}",
            "lyrics": _LYRIC_POOL[i % len(_LYRIC_POOL)] + f" (verse {i})",
            "valence": valence,
            "energy": energy,
            "tempo": 90.0 + (i % 7) * 12.0,
            "danceability": round(((i * 17) % 100) / 100.0, 3),
        }
        if with_vibe:
            row["vibe"] = _VIBES[i % len(_VIBES)]
        rows.append(row)
    return pd.DataFrame(rows)


# --- Bug 1: train() must create a missing model dir before writing CSV -------


def test_train_creates_missing_model_dir(tmp_path):
    data_csv = tmp_path / "songs.csv"
    _make_dataset(24).to_csv(data_csv, index=False)

    fresh_model_dir = tmp_path / "does" / "not" / "exist" / "models"
    assert not fresh_model_dir.exists()

    report = train(data_csv, fresh_model_dir)

    assert fresh_model_dir.exists()
    assert (fresh_model_dir / "training_features.csv").exists()
    assert report["samples"] == 24


# --- Bug 2: NaN valence/energy targets must not crash the fit ----------------


def test_train_from_dataframe_tolerates_nan_targets(tmp_path):
    df = _make_dataset(24)
    # Blank out some valence/energy values (still >= 10 valid rows remain).
    df.loc[[1, 4, 7], "valence"] = np.nan
    df.loc[[2, 5, 8], "energy"] = np.nan

    feat_df = build_feature_matrix(df)
    report = train_from_dataframe(feat_df, tmp_path / "models")

    # Should train without "Input y contains NaN" and report finite MAEs.
    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None
    assert np.isfinite(report["valence_mae"])
    assert np.isfinite(report["energy_mae"])


def test_train_from_dataframe_single_vibe_class_does_not_crash(tmp_path):
    df = _make_dataset(24, with_vibe=True)
    df["vibe"] = "chill"  # only one class

    feat_df = build_feature_matrix(df)
    report = train_from_dataframe(feat_df, tmp_path / "models")

    # GradientBoostingClassifier needs >=2 classes; single-class must be skipped
    # gracefully rather than raising.
    assert report["vibe_accuracy"] is None


# --- Bug 3: empty / header-only lyrics must be skipped, not fatal ------------


def test_build_feature_matrix_skips_empty_lyrics():
    df = _make_dataset(12)
    df.loc[0, "lyrics"] = ""          # empty string
    df.loc[3, "lyrics"] = "   "       # whitespace
    df.loc[6, "lyrics"] = "[Verse]"   # collapses to empty after header strip
    df.loc[9, "lyrics"] = np.nan      # missing

    feat_df = build_feature_matrix(df)

    assert len(feat_df) == 8
    # Source index is preserved so downstream code can realign labels.
    assert list(feat_df.index) == [1, 2, 4, 5, 7, 8, 10, 11]


def test_build_feature_matrix_all_empty_returns_empty():
    df = _make_dataset(4)
    df["lyrics"] = ""
    feat_df = build_feature_matrix(df)
    assert len(feat_df) == 0


# --- Bug 4: vibe_classes.json must be JSON text, not a pickle ----------------


def test_vibe_classes_written_as_json(tmp_path):
    df = _make_dataset(24, with_vibe=True)
    feat_df = build_feature_matrix(df)
    model_dir = tmp_path / "models"
    train_from_dataframe(feat_df, model_dir)

    classes_path = model_dir / "vibe_classes.json"
    assert classes_path.exists()
    with open(classes_path) as f:
        classes = json.load(f)  # would raise if joblib/pickle bytes were written
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)


def test_committed_vibe_classes_is_valid_json():
    path = REPO_ROOT / "models" / "vibe_classes.json"
    if not path.exists():
        pytest.skip("no committed models/vibe_classes.json")
    with open(path) as f:
        classes = json.load(f)
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)


# --- Bug 5: normalization must fold accents so "Beyoncé" == "Beyonce" --------


def test_normalize_folds_accents():
    from mood_reader.kaggle_merge import _normalize

    assert _normalize("Beyoncé") == _normalize("Beyonce")
    assert _normalize("Beyoncé") == "beyonce"
    assert _normalize("Rosalía") == "rosalia"
    assert _normalize("Céline Dion") == _normalize("Celine Dion")
    # sanity: still strips punctuation/spacing and lowercases
    assert _normalize("J Balvin!") == "jbalvin"
    assert _normalize(None) == ""
    assert _normalize(float("nan")) == ""


# --- Bug 6: the curated artist->hits dict must have no duplicate keys --------


def test_build_diverse_tracks_has_no_duplicate_keys():
    src = (REPO_ROOT / "scripts" / "build_diverse_tracks.py").read_text()
    tree = ast.parse(src)

    dup_report = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and len(node.keys) > 20:
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            seen: set[str] = set()
            dups = {k for k in keys if k in seen or seen.add(k)}
            dup_report = {"total": len(keys), "unique": len(set(keys)), "dups": sorted(dups)}

    assert dup_report, "ARTIST_HITS dict not found"
    assert dup_report["dups"] == [], f"duplicate artist keys drop tracks: {dup_report}"
    assert dup_report["total"] == dup_report["unique"]


# --- Bug 3 downstream: evaluate must realign rows after skipping --------------


def test_evaluate_realigns_rows_after_skipping(tmp_path):
    pytest.importorskip("matplotlib")
    from mood_reader.evaluate import train_evaluate_labeled_split

    df = _make_dataset(30, with_vibe=True)
    # Blank several lyrics so build_feature_matrix skips them; the positional
    # slicing bug would raise a length-mismatch when assigning predictions.
    for i in (2, 5, 11, 17, 23, 28):
        df.loc[i, "lyrics"] = ""
    data_csv = tmp_path / "labeled.csv"
    df.to_csv(data_csv, index=False)

    model_dir = tmp_path / "models"
    report = train_evaluate_labeled_split(
        data_csv,
        train_size=20,
        test_size=10,
        model_dir=model_dir,
        charts_dir=model_dir / "charts",
        random_state=7,
        include_spotify=True,
    )

    assert "vibe_accuracy_test" in report

    # Predictions must cover exactly the test rows that had usable lyrics.
    test_split = pd.read_csv(model_dir / "test_split.csv")
    usable = test_split["lyrics"].apply(lambda x: bool(str(x).strip()) and str(x) != "nan").sum()
    preds = pd.read_csv(model_dir / "test_predictions.csv")
    assert len(preds) == usable

"""Regression tests for mood_reader training/merge bug fixes.

Runs in FAST rule-based mode (see conftest.py) so no torch/network is needed.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from mood_reader import kaggle_merge, train

_REPO_ROOT = Path(__file__).resolve().parents[1]

_LYRIC_POOL = [
    "I feel so happy and alive tonight dancing in the bright light",
    "tears keep falling down my lonely broken heart in the dark",
    "we run wild and free burning bright with fire and raw energy",
    "calm quiet morning soft and gentle peace over the still sea",
    "love you forever my darling hold me close and never let go",
    "anger and rage screaming loud against the whole world tonight",
]

_VIBE_CYCLE = ["happy", "melancholic", "energetic"]


def _make_dataset(
    n: int,
    *,
    blanks: int = 0,
    single_vibe: bool = False,
    nan_targets: int = 0,
    with_targets: bool = True,
    with_vibe: bool = True,
) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "artist": f"Artist {i}",
            "title": f"Title {i}",
            "lyrics": f"{_LYRIC_POOL[i % len(_LYRIC_POOL)]} verse {i}",
        }
        if with_targets:
            row["valence"] = round(0.1 + (i % 9) / 10.0, 3)
            row["energy"] = round(0.15 + (i % 7) / 10.0, 3)
            row["tempo"] = float(80 + (i % 60))
            row["danceability"] = round(0.2 + (i % 8) / 10.0, 3)
        if with_vibe:
            row["vibe"] = "happy" if single_vibe else _VIBE_CYCLE[i % len(_VIBE_CYCLE)]
        rows.append(row)

    df = pd.DataFrame(rows)

    for j in range(blanks):
        df.loc[len(df) - 1 - j, "lyrics"] = "" if j % 2 == 0 else "   "

    if nan_targets and with_targets:
        for col in ("valence", "energy"):
            df.loc[: nan_targets - 1, col] = float("nan")

    return df


def _load_build_diverse_tracks():
    path = _REPO_ROOT / "scripts" / "build_diverse_tracks.py"
    spec = importlib.util.spec_from_file_location("build_diverse_tracks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- BUG 1: train() must create a fresh, nested model dir before writing ---
def test_train_creates_missing_model_dir(tmp_path):
    csv_path = tmp_path / "songs.csv"
    _make_dataset(12).to_csv(csv_path, index=False)
    model_dir = tmp_path / "fresh" / "nested" / "models"

    report = train.train(csv_path, model_dir=model_dir)

    assert (model_dir / "training_features.csv").exists()
    assert (model_dir / "metadata.json").exists()
    assert report["samples"] == 12


# --- BUG 2: NaN targets must be dropped, not fed into the regressor ---
def test_train_from_dataframe_handles_nan_targets(tmp_path):
    df = _make_dataset(15, nan_targets=3, with_vibe=False)
    feat_df = train.build_feature_matrix(df)

    report = train.train_from_dataframe(feat_df, tmp_path)

    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None


# --- BUG 2b: a single vibe class must not crash the classifier ---
def test_train_single_vibe_class(tmp_path):
    df = _make_dataset(14, with_targets=False, with_vibe=True, single_vibe=True)
    feat_df = train.build_feature_matrix(df)

    report = train.train_from_dataframe(feat_df, tmp_path)

    assert report["vibe_accuracy"] is None
    assert not (tmp_path / "vibe_model.joblib").exists()


# --- BUG 3: empty / header-only lyrics rows are skipped (index preserved) ---
def test_build_feature_matrix_skips_empty_lyrics():
    df = pd.DataFrame(
        {
            "artist": ["A", "B", "C", "D", "E"],
            "title": ["a", "b", "c", "d", "e"],
            "lyrics": ["real words here", "more real words", "", "   ", "[Verse]"],
        }
    )

    feat_df = train.build_feature_matrix(df)

    assert len(feat_df) == 2
    assert list(feat_df.index) == [0, 1]


# --- Happy path: full multimodal training produces all models ---
def test_train_happy_path(tmp_path):
    df = _make_dataset(30)
    feat_df = train.build_feature_matrix(df)

    report = train.train_from_dataframe(feat_df, tmp_path)

    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None
    assert report["vibe_accuracy"] is not None
    assert (tmp_path / "valence_model.joblib").exists()
    assert (tmp_path / "vibe_model.joblib").exists()


# --- BUG 4: vibe_classes.json must be JSON text, not a pickle ---
def test_vibe_classes_json_is_valid_json(tmp_path):
    feat_df = train.build_feature_matrix(_make_dataset(30))
    train.train_from_dataframe(feat_df, tmp_path)

    classes = json.loads((tmp_path / "vibe_classes.json").read_text(encoding="utf-8"))
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)


def test_committed_vibe_classes_json_valid():
    classes = json.loads((_REPO_ROOT / "models" / "vibe_classes.json").read_text(encoding="utf-8"))
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)


# --- BUG 5: normalization must fold accents so accented names match ---
def test_normalize_folds_accents():
    assert kaggle_merge._normalize("Beyoncé") == "beyonce"
    assert kaggle_merge._normalize("Beyoncé") == kaggle_merge._normalize("Beyonce")
    assert kaggle_merge._normalize("Café Tacvba") == kaggle_merge._normalize("Cafe Tacvba")


# --- BUG 6: the artist->tracks dict must not contain duplicate keys ---
def test_diverse_tracks_script_has_no_duplicate_keys():
    path = _REPO_ROOT / "scripts" / "build_diverse_tracks.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: list = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
            keys.extend(k.value for k in node.value.keys if isinstance(k, ast.Constant))

    assert keys, "expected an artist dict literal in the script"
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert dups == [], f"duplicate artist keys: {dups}"


def test_diverse_tracks_file_has_no_duplicate_lines():
    mod = _load_build_diverse_tracks()  # importable without side effects at import time
    assert mod.ARTIST_HITS

    lines = [
        line
        for line in (_REPO_ROOT / "data" / "diverse_tracks.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lowered = [line.lower() for line in lines]
    assert len(lowered) == len(set(lowered))


# --- Regression: build_feature_matrix skips must not misalign eval predictions ---
def test_evaluate_row_alignment_after_skip(tmp_path):
    pytest.importorskip("matplotlib")
    from mood_reader import evaluate

    csv_path = tmp_path / "labeled.csv"
    _make_dataset(60, blanks=8).to_csv(csv_path, index=False)

    report = evaluate.train_evaluate_labeled_split(
        csv_path,
        train_size=45,
        test_size=15,
        model_dir=tmp_path / "out",
        random_state=1,
        include_spotify=True,
    )

    preds = pd.read_csv(tmp_path / "out" / "test_predictions.csv")
    test_split = pd.read_csv(tmp_path / "out" / "test_split.csv")

    assert len(preds) == report["test_songs"]
    assert len(preds) < len(test_split)

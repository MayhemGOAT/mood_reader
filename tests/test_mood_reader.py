"""Regression tests for mood_reader crash bugs and data integrity.

All tests run in FAST (rule-based) mode with no network access; see conftest.py.
"""

import ast
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_LYRIC_POOL = [
    "sunshine and laughter fill the bright and golden summer air",
    "tears fall slowly through the lonely quiet of the night",
    "we run and jump and dance until the early morning light",
    "hold me close beneath the soft and gentle silver moon",
    "the storm is raging louder than my restless angry heart",
    "walking down an empty road with nothing left to lose",
    "bright city lights are shining as the whole crowd sings along",
    "memories of you keep fading like an old and worn photograph",
]


def _make_dataset(
    n: int,
    *,
    blanks: int = 0,
    single_vibe: bool = False,
    nan_targets: int = 0,
    with_targets: bool = True,
    with_vibe: bool = True,
) -> pd.DataFrame:
    """Build a synthetic labeled song dataframe for training/eval tests."""
    vibe_cycle = ["happy", "melancholic", "energetic"]
    rows = []
    for i in range(n):
        row = {
            "artist": f"Artist {i}",
            "title": f"Title {i}",
            "lyrics": f"{_LYRIC_POOL[i % len(_LYRIC_POOL)]} verse {i}",
            "tempo": 90.0 + (i % 40),
            "danceability": round(0.3 + (i % 5) * 0.1, 3),
        }
        if with_targets:
            row["valence"] = round(0.1 + (i % 9) * 0.1, 3)
            row["energy"] = round(0.9 - (i % 9) * 0.08, 3)
        if with_vibe:
            row["vibe"] = "happy" if single_vibe else vibe_cycle[i % 3]
        rows.append(row)
    df = pd.DataFrame(rows)

    for j in range(blanks):
        df.loc[len(df) - 1 - j, "lyrics"] = "" if j % 2 == 0 else "   "
    for j in range(nan_targets):
        df.loc[len(df) - 1 - j, "valence"] = float("nan")
        df.loc[len(df) - 1 - j, "energy"] = float("nan")

    return df


def _load_build_diverse_tracks():
    path = _REPO_ROOT / "scripts" / "build_diverse_tracks.py"
    spec = importlib.util.spec_from_file_location("build_diverse_tracks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Bug 1: train() must create a missing model dir before writing to it ---
def test_train_creates_missing_model_dir(tmp_path):
    from mood_reader.train import train

    csv = tmp_path / "songs.csv"
    _make_dataset(15).to_csv(csv, index=False)
    model_dir = tmp_path / "does" / "not" / "exist"

    report = train(csv, model_dir)  # must not raise OSError

    assert (model_dir / "training_features.csv").exists()
    assert (model_dir / "metadata.json").exists()
    assert report["samples"] == 15


# --- Bug 2: regressors must ignore rows whose target is NaN ---
def test_train_handles_nan_targets(tmp_path):
    from mood_reader.train import build_feature_matrix, train_from_dataframe

    df = _make_dataset(15, nan_targets=3, with_vibe=False)
    feat = build_feature_matrix(df)

    report = train_from_dataframe(feat, tmp_path)  # must not raise "Input y contains NaN"

    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None


# --- Bug 2b: a single vibe class must not crash the classifier ---
def test_train_single_vibe_class(tmp_path):
    from mood_reader.train import build_feature_matrix, train_from_dataframe

    df = _make_dataset(15, single_vibe=True, with_targets=False, with_vibe=True)
    feat = build_feature_matrix(df)

    report = train_from_dataframe(feat, tmp_path)  # must not raise "1 class" ValueError

    assert report["vibe_accuracy"] is None
    assert not (tmp_path / "vibe_model.joblib").exists()


# --- Bug 3: empty / header-only lyrics are skipped, not fatal, and index is kept ---
def test_build_feature_matrix_skips_empty_lyrics():
    from mood_reader.train import build_feature_matrix

    df = pd.DataFrame(
        {"lyrics": ["real song words here", "another real lyric line", "", "   ", "[Verse]"]}
    )
    feat = build_feature_matrix(df)

    assert len(feat) == 2
    assert list(feat.index) == [0, 1]


# --- Bug 4: vibe_classes.json must be JSON, not a pickle ---
def test_vibe_classes_json_written_as_json(tmp_path):
    from mood_reader.train import build_feature_matrix, train_from_dataframe

    df = _make_dataset(15)
    feat = build_feature_matrix(df)
    train_from_dataframe(feat, tmp_path)

    path = tmp_path / "vibe_classes.json"
    assert path.exists()
    with open(path) as f:
        classes = json.load(f)
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)


# --- Bug 5: normalization must fold accents so joins match ---
def test_normalize_folds_accents():
    from mood_reader.kaggle_merge import _normalize

    assert _normalize("Beyoncé") == "beyonce"
    assert _normalize("Beyoncé") == _normalize("Beyonce")
    assert _normalize("Rosalía") == "rosalia"


# --- Bug 6: curated artist map must not contain duplicate keys ---
def test_diverse_tracks_no_duplicate_keys():
    path = _REPO_ROOT / "scripts" / "build_diverse_tracks.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "ARTIST_HITS":
            keys = [k.value for k in node.value.keys]
    assert keys, "ARTIST_HITS annotated assignment not found"
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert dups == []


# --- The committed model artifact must be valid JSON (regression for bug 4) ---
def test_committed_vibe_classes_json_is_valid_json():
    path = _REPO_ROOT / "models" / "vibe_classes.json"
    with open(path) as f:
        classes = json.load(f)
    assert isinstance(classes, list)
    assert "happy" in classes


# --- The regenerated track list must have no duplicate lines ---
def test_diverse_tracks_file_has_no_duplicate_lines():
    path = _REPO_ROOT / "data" / "diverse_tracks.txt"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(set(lines))


# --- Bug 3 downstream: evaluate must realign rows after skipping blanks ---
def test_evaluate_row_alignment_after_skipping_blanks(tmp_path):
    pytest.importorskip("matplotlib")
    from mood_reader.evaluate import train_evaluate_labeled_split

    csv = tmp_path / "data.csv"
    _make_dataset(60, blanks=8).to_csv(csv, index=False)

    report = train_evaluate_labeled_split(
        csv,
        train_size=45,
        test_size=15,
        model_dir=tmp_path / "m",
        random_state=1,
    )

    test_split = pd.read_csv(tmp_path / "m" / "test_split.csv")
    test_preds = pd.read_csv(tmp_path / "m" / "test_predictions.csv")

    assert len(test_preds) == report["test_songs"]
    assert len(test_preds) < len(test_split)


# --- Happy path: a full labeled dataset trains all three models ---
def test_train_happy_path_full_pipeline(tmp_path):
    from mood_reader.train import train

    csv = tmp_path / "songs.csv"
    _make_dataset(30).to_csv(csv, index=False)

    report = train(csv, tmp_path / "m")

    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None
    assert report["vibe_accuracy"] is not None
    assert (tmp_path / "m" / "valence_model.joblib").exists()
    assert (tmp_path / "m" / "energy_model.joblib").exists()
    assert (tmp_path / "m" / "vibe_model.joblib").exists()

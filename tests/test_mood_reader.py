"""Regression tests for mood_reader training + data pipeline bug fixes.

All tests run in MOOD_READER_FAST mode (rule-based lyrics scoring, no network).
Each test targets a specific bug that previously crashed or corrupted output.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Distinct snippets so lyric feature scores vary across synthetic rows.
_LYRIC_POOL = [
    "sunshine dancing in the golden morning light we feel so alive",
    "tears fall down slowly in the lonely quiet empty room tonight",
    "we run so fast the fire burning loud and wild and free",
    "hold me close and whisper softly love will never fade away",
    "the darkness creeps around the corner cold and sharp and grey",
    "jumping high the crowd is roaring beat drops hard again",
    "gentle rain against the window calm and slow and blue",
    "shout it out we own the night the neon streets alight",
    "broken hearts and faded photographs of what we used to be",
    "smiling faces summer picnic laughter carried on the breeze",
    "thunder rolling engines revving adrenaline takes hold",
    "quiet candle flickers dimly memories of yesterday",
]

_VIBE_CYCLE = ["happy", "melancholic", "energetic"]


def _make_dataset(n: int, *, blanks: int = 0, single_vibe: bool = False,
                  nan_targets: int = 0, with_targets: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        lyrics = _LYRIC_POOL[i % len(_LYRIC_POOL)] + f" verse {i}"
        vibe = "happy" if single_vibe else _VIBE_CYCLE[i % len(_VIBE_CYCLE)]
        row = {
            "artist": f"Artist {i}",
            "title": f"Title {i}",
            "lyrics": lyrics,
            "vibe": vibe,
        }
        if with_targets:
            row["valence"] = float(rng.uniform(0.1, 0.9))
            row["energy"] = float(rng.uniform(0.1, 0.9))
            row["tempo"] = float(rng.uniform(70, 180))
            row["danceability"] = float(rng.uniform(0.1, 0.9))
        rows.append(row)
    df = pd.DataFrame(rows)
    for i in range(blanks):
        df.loc[i, "lyrics"] = "" if i % 2 == 0 else "   "
    for i in range(nan_targets):
        df.loc[n - 1 - i, "valence"] = np.nan
        df.loc[n - 1 - i, "energy"] = np.nan
    return df


def _load_build_diverse_tracks():
    path = REPO_ROOT / "scripts" / "build_diverse_tracks.py"
    spec = importlib.util.spec_from_file_location("build_diverse_tracks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Bug 3: empty/whitespace/header-only lyrics must be skipped, not crash ---

def test_build_feature_matrix_skips_empty_lyrics():
    from mood_reader.train import build_feature_matrix

    df = pd.DataFrame({
        "lyrics": [
            "real lyrics with plenty of happy words here",
            "another line full of energy and joy tonight",
            "",
            "   ",
            "[Verse]",
        ],
    })
    feat = build_feature_matrix(df)
    # Empty and whitespace rows are skipped up front; the "[Verse]" header
    # strips to empty and is caught via the ValueError guard. Only real rows kept.
    assert len(feat) == 2
    # Source positional index is preserved for downstream alignment.
    assert list(feat.index) == [0, 1]


# --- Bug 1: train() must create a fresh (nested) model dir before writing ---

def test_train_creates_fresh_model_dir(tmp_path):
    from mood_reader.train import train

    df = _make_dataset(14)
    csv = tmp_path / "songs.csv"
    df.to_csv(csv, index=False)

    model_dir = tmp_path / "deeply" / "nested" / "models"
    report = train(str(csv), str(model_dir))

    assert model_dir.exists()
    assert (model_dir / "training_features.csv").exists()
    assert (model_dir / "metadata.json").exists()
    assert report["samples"] == 14


# --- Bug 2: NaN valence/energy rows must not blow up the regressors ---

def test_train_handles_nan_targets(tmp_path):
    from mood_reader.train import build_feature_matrix, train_from_dataframe

    df = _make_dataset(20, nan_targets=6)
    feat = build_feature_matrix(df)
    report = train_from_dataframe(feat, tmp_path / "m")

    # 14 valid target rows remain (>= 10), so models still train.
    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None
    assert (tmp_path / "m" / "valence_model.joblib").exists()


# --- Bug 2b: a single vibe class must not crash the classifier ---

def test_train_single_vibe_class(tmp_path):
    from mood_reader.train import build_feature_matrix, train_from_dataframe

    df = _make_dataset(15, single_vibe=True, with_targets=False)
    feat = build_feature_matrix(df)
    report = train_from_dataframe(feat, tmp_path / "m")

    assert report["vibe_accuracy"] is None
    assert not (tmp_path / "m" / "vibe_model.joblib").exists()


# --- Bug 4: vibe_classes.json must be JSON text, not a pickle ---

def test_vibe_classes_json_is_valid_json(tmp_path):
    from mood_reader.train import build_feature_matrix, train_from_dataframe

    df = _make_dataset(20, with_targets=False)  # cycles >= 2 vibes
    feat = build_feature_matrix(df)
    report = train_from_dataframe(feat, tmp_path / "m")

    assert report["vibe_accuracy"] is not None
    classes_path = tmp_path / "m" / "vibe_classes.json"
    assert classes_path.exists()
    with open(classes_path) as f:
        classes = json.load(f)
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)


def test_committed_vibe_classes_json_valid():
    with open(REPO_ROOT / "models" / "vibe_classes.json") as f:
        classes = json.load(f)
    assert isinstance(classes, list) and classes


# --- Bug 5: artist/title normalization must fold accents for merge keys ---

def test_normalize_folds_accents():
    from mood_reader.kaggle_merge import _merge_key, _normalize

    assert _normalize("Beyoncé") == "beyonce"
    assert _normalize("Rosalía") == "rosalia"
    assert _merge_key("Beyoncé", "Déjà Vu") == _merge_key("Beyonce", "Deja Vu")


# --- Bug 6: no duplicate artist keys in ARTIST_HITS + track list integrity ---

def test_no_duplicate_artist_keys():
    src = (REPO_ROOT / "scripts" / "build_diverse_tracks.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "ARTIST_HITS":
            keys = [k.value for k in node.value.keys]
    assert keys, "ARTIST_HITS not found"
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert dups == [], f"duplicate artist keys: {dups}"


def test_diverse_tracks_merged_and_unique():
    module = _load_build_diverse_tracks()
    hits = module.ARTIST_HITS
    # Previously-dropped first-occurrence tracks are now retained.
    assert "Hello" in hits["Adele"] and "Skyfall" in hits["Adele"]
    assert "Fantastic Baby" in hits["BIGBANG"] and "FXXK IT" in hits["BIGBANG"]

    committed = (REPO_ROOT / "data" / "diverse_tracks.txt").read_text(encoding="utf-8")
    lines = [ln for ln in committed.splitlines() if ln.strip()]
    assert len(lines) == len(set(lines)), "duplicate track lines in diverse_tracks.txt"
    assert "Adele - Hello" in lines and "Adele - Skyfall" in lines


# --- Bug 3 downstream: evaluate must stay row-aligned after skipping blanks ---

def test_evaluate_row_alignment_after_skip(tmp_path):
    pytest.importorskip("matplotlib")
    from mood_reader.evaluate import train_evaluate_labeled_split

    df = _make_dataset(60, blanks=8)
    csv = tmp_path / "labeled.csv"
    df.to_csv(csv, index=False)

    train_evaluate_labeled_split(
        str(csv),
        train_size=45,
        test_size=15,
        model_dir=str(tmp_path / "m"),
        random_state=1,
    )

    test_split = pd.read_csv(tmp_path / "m" / "test_split.csv")
    preds = pd.read_csv(tmp_path / "m" / "test_predictions.csv")

    usable = test_split["lyrics"].apply(lambda x: pd.notna(x) and str(x).strip() != "").sum()
    # Predictions align exactly with the usable-lyric rows of the test split,
    # and at least one blank row was dropped (regression guard for the old
    # positional slice that produced a length mismatch).
    assert len(preds) == int(usable)
    assert len(preds) < len(test_split)

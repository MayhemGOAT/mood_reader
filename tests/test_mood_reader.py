"""Offline regression tests for the mood_reader training/data pipeline.

These run in FAST mode (rule-based lyric scoring, see conftest) so they need
neither torch/transformers nor network access. Each test pins a concrete bug
that the pipeline previously crashed on or silently mishandled.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mood_reader import train as train_mod
from mood_reader.kaggle_merge import _normalize

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Distinct lyric snippets so extracted features (and thus targets) actually vary.
_LYRIC_POOL = [
    "I feel the love in my heart tonight dancing under bright lights",
    "Tears fall as I cry alone lonely and lost and blue without you",
    "We fight and burn with rage and fire the war inside my soul",
    "Rise and shine we dream and fly hope lights up the whole sky",
    "Dark shadows creep the devil calls blood on the cold stone walls",
    "Party all night the club is wild feel the beat and lose your mind",
    "Baby hold me close and kiss me slow romance in the candle glow",
    "Smile and laugh the joy is real sunshine on a happy field",
]

_VIBE_CYCLE = ["happy", "melancholic", "energetic"]


def _make_dataset(
    n,
    *,
    blanks=0,
    single_vibe=False,
    nan_targets=0,
    with_targets=True,
    with_vibe=False,
):
    """Build a synthetic labeled dataset with the columns the pipeline expects."""
    rows = []
    for i in range(n):
        row = {
            "artist": f"Artist {i}",
            "title": f"Song {i}",
            "lyrics": _LYRIC_POOL[i % len(_LYRIC_POOL)] + f" verse {i}",
        }
        if with_targets:
            row["valence"] = round(0.1 + (i % 9) / 10.0, 3)
            row["energy"] = round(0.2 + (i % 7) / 10.0, 3)
            row["tempo"] = float(90 + (i % 60))
            row["danceability"] = round(0.3 + (i % 5) / 10.0, 3)
        if with_vibe:
            row["vibe"] = "happy" if single_vibe else _VIBE_CYCLE[i % len(_VIBE_CYCLE)]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Blank/garbage-out some lyrics (last rows) to exercise the skip path.
    for j in range(min(blanks, n)):
        idx = n - 1 - j
        df.at[idx, "lyrics"] = "" if j % 2 == 0 else "[Verse]"

    # NaN out some numeric targets (first rows) to exercise the NaN guard.
    if nan_targets and with_targets:
        for j in range(min(nan_targets, n)):
            df.at[j, "valence"] = np.nan
            df.at[j, "energy"] = np.nan

    return df


def _load_build_diverse_tracks():
    path = _REPO_ROOT / "scripts" / "build_diverse_tracks.py"
    spec = importlib.util.spec_from_file_location("build_diverse_tracks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Bug 1: train() must create a missing model dir before writing --------------

def test_train_creates_missing_model_dir(tmp_path):
    df = _make_dataset(14, with_targets=True, with_vibe=True)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    model_dir = tmp_path / "models" / "does_not_exist_yet"
    report = train_mod.train(csv, model_dir)

    assert model_dir.exists()
    assert (model_dir / "training_features.csv").exists()
    assert (model_dir / "metadata.json").exists()
    assert isinstance(report, dict)


# --- Bug 2: NaN targets must be filtered, not fed to the regressor --------------

def test_train_handles_nan_targets(tmp_path):
    df = _make_dataset(15, nan_targets=3, with_targets=True, with_vibe=True)
    feat = train_mod.build_feature_matrix(df)
    report = train_mod.train_from_dataframe(feat, tmp_path)

    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None
    assert (tmp_path / "valence_model.joblib").exists()


# --- Bug 2b: a single vibe class must not crash the classifier ------------------

def test_train_single_vibe_class_no_crash(tmp_path):
    df = _make_dataset(15, with_targets=False, with_vibe=True, single_vibe=True)
    feat = train_mod.build_feature_matrix(df)
    report = train_mod.train_from_dataframe(feat, tmp_path)

    assert report["vibe_accuracy"] is None
    assert not (tmp_path / "vibe_model.joblib").exists()


# --- Bug 3: empty / header-only lyrics must be skipped, not abort the run -------

def test_build_feature_matrix_skips_empty_lyrics():
    df = pd.DataFrame(
        {
            "lyrics": [
                "I feel the love and joy tonight",
                "Tears and pain lonely and blue",
                "",
                "   ",
                "[Verse]",
            ],
            "valence": [0.8, 0.2, 0.5, 0.5, 0.5],
            "energy": [0.7, 0.3, 0.5, 0.5, 0.5],
        }
    )
    feat = train_mod.build_feature_matrix(df)

    # "", "   " are dropped up front; "[Verse]" reduces to empty after header
    # stripping and is dropped via the ValueError guard.
    assert len(feat) == 2
    assert list(feat.index) == [0, 1]


# --- Bug 4: vibe_classes.json must be real JSON, not a pickle -------------------

def test_vibe_classes_written_as_json(tmp_path):
    df = _make_dataset(15, with_targets=True, with_vibe=True)
    feat = train_mod.build_feature_matrix(df)
    train_mod.train_from_dataframe(feat, tmp_path)

    path = tmp_path / "vibe_classes.json"
    assert path.exists()
    with open(path) as f:
        classes = json.load(f)
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)
    assert sorted(classes) == classes


def test_committed_vibe_classes_json_is_valid_json():
    path = _REPO_ROOT / "models" / "vibe_classes.json"
    with open(path) as f:
        classes = json.load(f)
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)


# --- Bug 5: normalization must fold accents so join keys align ------------------

def test_normalize_folds_accents():
    assert _normalize("Beyoncé") == "beyonce"
    assert _normalize("Beyoncé") == _normalize("Beyonce")
    assert _normalize("Rosalía") == "rosalia"
    assert _normalize(None) == ""


# --- Bug 6: curated artist table must not silently drop duplicate keys ----------

def test_diverse_tracks_source_has_no_duplicate_keys():
    path = _REPO_ROOT / "scripts" / "build_diverse_tracks.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    keys: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "ARTIST_HITS":
            for key_node in node.value.keys:
                keys.append(key_node.value)

    assert keys, "ARTIST_HITS dict literal not found"
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert dups == [], f"duplicate artist keys silently drop tracks: {dups}"


def test_diverse_tracks_output_has_no_duplicate_lines():
    mod = _load_build_diverse_tracks()

    lines: list[str] = []
    seen: set[str] = set()
    for artist, titles in mod.ARTIST_HITS.items():
        for title in titles:
            key = f"{artist.lower()}|{title.lower()}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{artist} - {title}")

    assert len(lines) == len(set(lines))
    # Every artist contributes at least one output line (none fully dropped).
    artists_in_output = {line.split(" - ", 1)[0] for line in lines}
    assert artists_in_output == set(mod.ARTIST_HITS.keys())


# --- Bug 3 downstream: evaluate must keep predictions aligned after skips -------

def test_evaluate_realigns_rows_after_skipping_lyrics(tmp_path):
    pytest.importorskip("matplotlib")
    from mood_reader import evaluate as eval_mod

    df = _make_dataset(60, blanks=8, with_targets=True, with_vibe=True)
    csv = tmp_path / "labeled.csv"
    df.to_csv(csv, index=False)

    model_dir = tmp_path / "models"
    report = eval_mod.train_evaluate_labeled_split(
        csv,
        train_size=45,
        test_size=15,
        model_dir=model_dir,
        random_state=1,
    )

    test_split = pd.read_csv(model_dir / "test_split.csv")
    test_preds = pd.read_csv(model_dir / "test_predictions.csv")

    assert len(test_split) == 15
    # Blank/header-only lyrics are dropped, so fewer predictions than raw rows.
    assert len(test_preds) < len(test_split)
    assert report["test_songs"] == len(test_preds)
    assert "pred_valence" in test_preds.columns
    assert test_preds["pred_valence"].notna().all()


# --- Happy path: a well-formed dataset trains all three heads -------------------

def test_train_full_pipeline_happy_path(tmp_path):
    df = _make_dataset(30, with_targets=True, with_vibe=True)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    model_dir = tmp_path / "models" / "out"
    report = train_mod.train(csv, model_dir)

    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None
    assert report["vibe_accuracy"] is not None
    assert (model_dir / "valence_model.joblib").exists()
    assert (model_dir / "energy_model.joblib").exists()
    assert (model_dir / "vibe_model.joblib").exists()

    with open(model_dir / "vibe_classes.json") as f:
        json.load(f)
    with open(model_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["report"]["vibe_accuracy"] is not None

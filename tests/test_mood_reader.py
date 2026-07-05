"""Regression tests for known mood_reader bugs.

Each test maps to a concrete defect in the training / data pipeline. They run in
FAST mode (see conftest) and require no network access.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mood_reader.kaggle_merge import _merge_key, _normalize
from mood_reader.train import build_feature_matrix, train, train_from_dataframe

REPO_ROOT = Path(__file__).resolve().parents[1]

_LYRIC_SNIPPETS = [
    "I love you baby, my heart is on fire tonight",
    "We dance all night, the party never stops, we are wild and free",
    "Lonely tears fall down, I miss you and the pain won't go",
    "Rise up and shine, chase the dream, we can fly so high",
    "Dark shadows fall, the devil calls from down in hell",
    "Sunshine morning, coffee and a smile, everything is good",
    "Rage and fire, we fight until the war is over now",
    "Quiet river flowing, calm and slow, the mellow evening glow",
]


def _make_dataset(n: int, *, vibes: list[str] | None = None) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        valence = round(float(rng.uniform(0.1, 0.9)), 3)
        energy = round(float(rng.uniform(0.1, 0.9)), 3)
        row = {
            "track_id": f"t{i}",
            "title": f"Song {i}",
            "artist": f"Artist {i % 5}",
            "lyrics": _LYRIC_SNIPPETS[i % len(_LYRIC_SNIPPETS)] + f" number {i}",
            "audio_path": "",
            "valence": valence,
            "energy": energy,
            "tempo": round(float(rng.uniform(60, 180)), 1),
        }
        if vibes is not None:
            row["vibe"] = vibes[i % len(vibes)]
        rows.append(row)
    return pd.DataFrame(rows)


# --- Bug 1: train() must create a missing model_dir before writing to it ---
def test_train_creates_missing_model_dir(tmp_path):
    csv = tmp_path / "songs.csv"
    _make_dataset(24, vibes=["happy", "chill", "melancholic"]).to_csv(csv, index=False)

    model_dir = tmp_path / "nested" / "does_not_exist" / "models"
    assert not model_dir.exists()

    report = train(str(csv), str(model_dir))

    assert model_dir.exists()
    assert (model_dir / "training_features.csv").exists()
    assert report["samples"] == 24


# --- Bug 2: partially-labeled valence/energy must not raise "Input y contains NaN" ---
def test_train_handles_partial_numeric_labels(tmp_path):
    df = _make_dataset(24, vibes=["happy", "chill", "melancholic"])
    # Blank out some (but < the >=10 usable threshold) valence/energy values.
    df.loc[df.index[:6], "valence"] = np.nan
    df.loc[df.index[:6], "energy"] = np.nan

    feat_df = build_feature_matrix(df)
    report = train_from_dataframe(feat_df, tmp_path / "models")

    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None


# --- Bug 2b: a single vibe class must be skipped, not crash the classifier ---
def test_train_single_vibe_class_is_skipped(tmp_path):
    df = _make_dataset(24, vibes=["happy"])  # only one class
    feat_df = build_feature_matrix(df)
    report = train_from_dataframe(feat_df, tmp_path / "models")

    assert report["vibe_accuracy"] is None
    assert not (tmp_path / "models" / "vibe_model.joblib").exists()


# --- Bug 3: empty / header-only / NaN lyrics must be skipped, not abort the run ---
def test_build_feature_matrix_skips_unusable_lyrics():
    df = pd.DataFrame(
        {
            "lyrics": [
                "I love you baby tonight",  # valid
                "",  # empty
                "   ",  # whitespace
                "[Verse]\n[Chorus]",  # header-only -> cleans to empty
                np.nan,  # missing
                "We dance all night long",  # valid
            ]
        }
    )
    feat_df = build_feature_matrix(df)
    assert len(feat_df) == 2


# --- Bug 4: vibe_classes.json must be valid JSON (not a joblib pickle) ---
def test_vibe_classes_written_as_json(tmp_path):
    df = _make_dataset(30, vibes=["happy", "chill", "melancholic"])
    feat_df = build_feature_matrix(df)
    model_dir = tmp_path / "models"
    report = train_from_dataframe(feat_df, model_dir)

    assert report["vibe_accuracy"] is not None
    classes_path = model_dir / "vibe_classes.json"
    assert classes_path.exists()

    loaded = json.loads(classes_path.read_text())
    assert isinstance(loaded, list)
    assert set(loaded) == {"happy", "chill", "melancholic"}


def test_committed_vibe_classes_is_valid_json():
    path = REPO_ROOT / "models" / "vibe_classes.json"
    if path.exists():
        loaded = json.loads(path.read_text())
        assert isinstance(loaded, list)


# --- Bug 5: merge key must ASCII-fold accents so "Beyonce" == "Beyonce" ---
@pytest.mark.parametrize(
    "accented,plain",
    [
        ("Beyoncé", "Beyonce"),
        ("Rosalía", "Rosalia"),
        ("Céline Dion", "Celine Dion"),
        ("Sinéad O'Connor", "Sinead O'Connor"),
    ],
)
def test_normalize_folds_accents(accented, plain):
    assert _normalize(accented) == _normalize(plain)
    assert _normalize(accented) != ""


def test_merge_key_matches_accent_variants():
    assert _merge_key("Beyoncé", "Déjà Vu") == _merge_key("Beyonce", "Deja Vu")


# --- Regression: evaluate must stay aligned when build_feature_matrix skips rows ---
def test_evaluate_aligns_after_skipped_lyrics(tmp_path):
    pytest.importorskip("matplotlib")
    from mood_reader.evaluate import train_evaluate_labeled_split

    df = _make_dataset(40, vibes=["happy", "chill", "melancholic", "energetic"])
    # Inject a couple of rows that build_feature_matrix will skip (unusable lyrics).
    df.loc[5, "lyrics"] = ""
    df.loc[25, "lyrics"] = "[Verse]\n[Chorus]"

    report = train_evaluate_labeled_split(
        str(_write_csv(df, tmp_path / "labeled.csv")),
        train_size=28,
        test_size=12,
        model_dir=tmp_path / "models",
        charts_dir=tmp_path / "charts",
    )
    # Should complete without a length-mismatch and produce aligned predictions.
    assert 0.0 <= report["vibe_accuracy_test"] <= 1.0
    preds = pd.read_csv(tmp_path / "models" / "test_predictions.csv")
    assert "pred_valence" in preds.columns
    assert len(preds) > 0


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


# --- Bug 6: the curated artist dict must have no duplicate keys ---
def test_build_diverse_tracks_has_no_duplicate_keys():
    src = (REPO_ROOT / "scripts" / "build_diverse_tracks.py").read_text()
    tree = ast.parse(src)

    artist_dicts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict) and len(node.keys) > 20
    ]
    assert artist_dicts, "Could not locate the ARTIST_HITS dict"

    keys = [k.value for k in artist_dicts[0].keys if isinstance(k, ast.Constant)]
    assert len(keys) == len(set(keys)), "Duplicate artist keys present"

"""Regression tests for the mood_reader training + data pipeline.

Each test targets a concrete bug that previously crashed or silently corrupted
the pipeline. Everything runs in FAST mode (rule-based lyrics scoring) so the
suite is offline and quick.
"""

from __future__ import annotations

import ast
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mood_reader.kaggle_merge import _merge_key, _normalize
from mood_reader.train import build_feature_matrix, train, train_from_dataframe

ROOT = Path(__file__).resolve().parents[1]

# Distinct snippets so VADER / rule-based emotion scores actually vary per row.
_LYRIC_POOL = [
    "I love you baby my heart will shine so bright for you tonight",
    "We dance all night at the party the fire burns wild and free",
    "Tears fall down my lonely face the pain of goodbye hurts so bad",
    "Rage and fury burn the world we fight and hate until the bitter end",
    "Rise up and shine we dream of hope and freedom flying ever high",
    "Dark shadows creep the devil calls from hell in this endless night",
    "Sunshine warms my happy soul as we smile and laugh and play again",
    "Blue and sad I cry alone here missing you every single lonely day",
    "Hold me close under the moon your gentle kiss is all I need now",
    "Thunder roars the crowd goes wild we jump and scream into the storm",
]

_DEFAULT_VIBES = ["happy", "melancholic", "energetic"]


def _make_dataset(n: int, *, vibes: list[str] | None = None) -> pd.DataFrame:
    """Build a synthetic labeled song dataset with varied lyrics + Spotify cols."""
    rng = np.random.default_rng(0)
    vibes = vibes or _DEFAULT_VIBES
    rows = []
    for i in range(n):
        rows.append(
            {
                "artist": f"Artist {i}",
                "title": f"Title {i}",
                "lyrics": f"{_LYRIC_POOL[i % len(_LYRIC_POOL)]} (verse {i})",
                "valence": round(float(rng.uniform(0.1, 0.9)), 3),
                "energy": round(float(rng.uniform(0.1, 0.9)), 3),
                "tempo": round(float(rng.uniform(70.0, 180.0)), 1),
                "danceability": round(float(rng.uniform(0.2, 0.9)), 3),
                "vibe": vibes[i % len(vibes)],
            }
        )
    return pd.DataFrame(rows)


# --- Bug 3: empty / whitespace lyrics must be skipped, not crash -------------
def test_build_feature_matrix_skips_empty_lyrics():
    df = _make_dataset(10)
    df.loc[0, "lyrics"] = ""
    df.loc[1, "lyrics"] = "   "
    df.loc[2, "lyrics"] = np.nan

    feat = build_feature_matrix(df)

    assert len(feat) == 7
    # Source-row index is preserved so callers can realign labels.
    assert list(feat.index) == [3, 4, 5, 6, 7, 8, 9]


# --- Bug 1: train() must create a missing model dir before writing ----------
def test_train_creates_missing_model_dir(tmp_path):
    df = _make_dataset(20)
    csv = tmp_path / "songs.csv"
    df.to_csv(csv, index=False)

    model_dir = tmp_path / "does" / "not" / "exist"
    report = train(str(csv), str(model_dir))

    assert (model_dir / "training_features.csv").exists()
    assert report["samples"] == 20


# --- Bug 2a: NaN valence/energy must be masked, not fed to the regressor ----
def test_train_from_dataframe_handles_nan_targets(tmp_path):
    df = _make_dataset(30)
    df.loc[:4, "valence"] = np.nan
    df.loc[:4, "energy"] = np.nan

    feat_df = build_feature_matrix(df)
    report = train_from_dataframe(feat_df, tmp_path)

    # 25 usable rows >= 10 -> a model is trained and a numeric MAE reported.
    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None


# --- Bug 2b: a single vibe class must not crash the classifier --------------
def test_train_from_dataframe_single_vibe_class(tmp_path):
    df = _make_dataset(20, vibes=["happy"])
    feat_df = build_feature_matrix(df)

    report = train_from_dataframe(feat_df, tmp_path)

    assert report["vibe_accuracy"] is None


# --- Bug 4: vibe_classes.json must be JSON text, not a pickle ---------------
def test_vibe_classes_json_is_valid_json(tmp_path):
    df = _make_dataset(40)
    feat_df = build_feature_matrix(df)
    train_from_dataframe(feat_df, tmp_path)

    classes_path = tmp_path / "vibe_classes.json"
    assert classes_path.exists()
    data = json.loads(classes_path.read_text())  # would raise if pickled
    assert isinstance(data, list) and data


def test_committed_vibe_classes_json_valid():
    data = json.loads((ROOT / "models" / "vibe_classes.json").read_text())
    assert isinstance(data, list) and data


# --- Bug 5: normalization must fold accents so names match ------------------
def test_normalize_folds_accents():
    assert _normalize("Beyoncé") == "beyonce"
    assert _normalize("Beyoncé") == _normalize("Beyonce")
    assert _normalize("Sí Señor") == _normalize("Si Senor")


def test_merge_key_matches_accented_names():
    assert _merge_key("Beyoncé", "Déjà Vu") == _merge_key("Beyonce", "Deja Vu")


# --- Bug 6: curated artist list must have no duplicate dict keys ------------
def test_no_duplicate_artist_keys():
    src = (ROOT / "scripts" / "build_diverse_tracks.py").read_text()
    tree = ast.parse(src)
    keys: list = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "ARTIST_HITS"
            and isinstance(node.value, ast.Dict)
        ):
            keys = [k.value for k in node.value.keys]

    assert keys, "ARTIST_HITS dict not found"
    dups = {k: v for k, v in collections.Counter(keys).items() if v > 1}
    assert not dups, f"duplicate artist keys drop tracks: {dups}"


# --- Bug 3 downstream: evaluate must stay row-aligned after skipping --------
def test_evaluate_row_alignment_after_skip(tmp_path):
    pytest.importorskip("matplotlib")
    from mood_reader.evaluate import train_evaluate_labeled_split

    df = _make_dataset(60)
    # Blank out lyrics in a spread of rows; these must be dropped from features
    # but still appear in the raw test_split.csv.
    for i in (3, 11, 18, 27, 34, 41, 52, 58):
        df.loc[i, "lyrics"] = "" if i % 2 else "   "
    data_csv = tmp_path / "labeled.csv"
    df.to_csv(data_csv, index=False)

    train_evaluate_labeled_split(
        str(data_csv),
        train_size=45,
        test_size=15,
        model_dir=str(tmp_path / "out"),
        random_state=42,
    )

    test_split = pd.read_csv(tmp_path / "out" / "test_split.csv")
    test_pred = pd.read_csv(tmp_path / "out" / "test_predictions.csv")

    usable = int(
        test_split["lyrics"].apply(lambda x: not (pd.isna(x) or not str(x).strip())).sum()
    )
    assert len(test_pred) == usable
    assert len(test_pred) < len(test_split)  # some test rows were genuinely skipped

"""Regression tests for the mood_reader training / data pipeline bugs.

All tests run in FAST mode (see conftest.py) so they need no network access,
no Spotify/Genius keys, and no torch/transformers.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from mood_reader import kaggle_merge
from mood_reader.train import build_feature_matrix, train, train_from_dataframe

REPO_ROOT = Path(__file__).resolve().parents[1]

# Distinct snippets so lyric features vary row-to-row (avoids degenerate models).
_LYRIC_POOL = [
    "i love you baby hold me close tonight",
    "dance all night party fire in the club",
    "tears fall down alone crying in the rain",
    "rise up shine bright chase the golden dream",
    "dark shadows creep the devil calls my name",
    "sunshine smiles laughing hearts feeling so alive",
    "broken heart goodbye i miss you every single day",
    "fight the rage burn it down never back down",
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
    """Build a small synthetic labeled dataset.

    - ``blanks``: number of leading rows whose lyrics are blank/whitespace.
    - ``single_vibe``: give every row the same vibe (tests the single-class guard).
    - ``nan_targets``: number of trailing rows with NaN valence/energy.
    - ``with_targets``: include numeric valence/energy/tempo/danceability columns.
    - ``with_vibe``: include a ``vibe`` label column.
    """
    vibes = ["happy", "melancholic", "energetic"]
    rows = []
    for i in range(n):
        row = {
            "artist": f"Artist {i}",
            "title": f"Title {i}",
            "lyrics": f"{_LYRIC_POOL[i % len(_LYRIC_POOL)]} verse {i}",
        }
        if with_targets:
            row["valence"] = round(0.15 + (i % 5) * 0.17, 3)
            row["energy"] = round(0.20 + (i % 4) * 0.19, 3)
            row["tempo"] = float(80 + (i % 6) * 12)
            row["danceability"] = round(0.35 + (i % 3) * 0.21, 3)
        if with_vibe:
            row["vibe"] = "happy" if single_vibe else vibes[i % len(vibes)]
        rows.append(row)

    df = pd.DataFrame(rows)

    for j in range(min(blanks, n)):
        df.loc[j, "lyrics"] = "" if j % 2 == 0 else "   "

    if nan_targets and with_targets:
        for j in range(n - nan_targets, n):
            df.loc[j, "valence"] = float("nan")
            df.loc[j, "energy"] = float("nan")

    return df


def test_train_creates_fresh_model_dir(tmp_path):
    """BUG 1: train() must create the model dir before writing training_features.csv."""
    csv_path = tmp_path / "songs.csv"
    _make_dataset(16).to_csv(csv_path, index=False)

    fresh_dir = tmp_path / "does" / "not" / "exist"
    report = train(csv_path, fresh_dir)

    assert (fresh_dir / "training_features.csv").exists()
    assert report["samples"] == 16


def test_train_masks_nan_targets(tmp_path):
    """BUG 2: NaN valence/energy rows must be dropped before fitting regressors."""
    df = _make_dataset(15, nan_targets=3)
    feat_df = build_feature_matrix(df)

    report = train_from_dataframe(feat_df, tmp_path)

    # 12 usable target rows (>= 10) so the regressors still train.
    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None


def test_train_single_vibe_class_no_crash(tmp_path):
    """BUG 2b: a single vibe class must not crash the classifier fit."""
    df = _make_dataset(14, single_vibe=True, with_targets=False)
    feat_df = build_feature_matrix(df)

    report = train_from_dataframe(feat_df, tmp_path)

    # Only one class present -> classifier is skipped, not fit.
    assert report["vibe_accuracy"] is None
    assert not (tmp_path / "vibe_model.joblib").exists()


def test_build_feature_matrix_skips_empty_lyrics():
    """BUG 3: empty / header-only lyrics are skipped, not fatal."""
    df = pd.DataFrame(
        {"lyrics": ["real love lyrics here", "more real dancing lyrics", "", "   ", "[Verse]"]}
    )

    feat_df = build_feature_matrix(df)

    # Rows 0 and 1 survive; the '[Verse]' row cleans to empty and is skipped too.
    assert len(feat_df) == 2
    assert list(feat_df.index) == [0, 1]


def test_train_writes_valid_json_vibe_classes(tmp_path):
    """BUG 4: vibe_classes.json must be valid JSON, not a joblib pickle."""
    csv_path = tmp_path / "songs.csv"
    _make_dataset(18).to_csv(csv_path, index=False)

    train(csv_path, tmp_path)

    classes_path = tmp_path / "vibe_classes.json"
    assert classes_path.exists()
    with open(classes_path) as f:
        classes = json.load(f)
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)
    assert len(classes) >= 2


def test_committed_vibe_classes_is_valid_json():
    """The committed models/vibe_classes.json artifact must be loadable JSON."""
    path = REPO_ROOT / "models" / "vibe_classes.json"
    with open(path) as f:
        classes = json.load(f)
    assert isinstance(classes, list)
    assert all(isinstance(c, str) for c in classes)


def test_train_full_pipeline_reports_all_metrics(tmp_path):
    """Happy path: valence/energy/vibe all train and produce metrics + models."""
    csv_path = tmp_path / "songs.csv"
    _make_dataset(30).to_csv(csv_path, index=False)

    report = train(csv_path, tmp_path)

    assert report["valence_mae"] is not None
    assert report["energy_mae"] is not None
    assert report["vibe_accuracy"] is not None
    for name in ("valence_model.joblib", "energy_model.joblib", "vibe_model.joblib"):
        assert (tmp_path / name).exists()


def test_normalize_folds_accents():
    """BUG 5: _normalize must fold accents so accented spellings match."""
    assert kaggle_merge._normalize("Beyoncé") == "beyonce"
    assert kaggle_merge._normalize("José") == "jose"
    assert kaggle_merge._normalize("Sinéad O'Connor") == "sineadoconnor"
    assert kaggle_merge._merge_key("Beyoncé", "Déjà Vu") == kaggle_merge._merge_key(
        "Beyonce", "Deja Vu"
    )


def _load_build_diverse_tracks():
    path = REPO_ROOT / "scripts" / "build_diverse_tracks.py"
    spec = importlib.util.spec_from_file_location("build_diverse_tracks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def test_no_duplicate_artist_keys():
    """BUG 6: ARTIST_HITS must not contain duplicate dict keys (silently dropped)."""
    _, path = _load_build_diverse_tracks()
    tree = ast.parse(path.read_text(encoding="utf-8"))

    dups: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
            if len(keys) < 50:
                continue
            seen: set[str] = set()
            for k in keys:
                if k in seen:
                    dups.append(k)
                seen.add(k)

    assert dups == [], f"duplicate artist keys still present: {sorted(set(dups))}"


def test_diverse_tracks_file_has_no_duplicate_lines():
    """The regenerated data file should have no duplicate track lines."""
    lines = (REPO_ROOT / "data" / "diverse_tracks.txt").read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    assert len(lines) == len(set(lines))


def test_evaluate_row_alignment_after_skip(tmp_path):
    """Regression: skipping empty-lyric rows must not misalign test predictions.

    Before the fix, evaluate.py used positional slicing plus ``test_df.copy()`` and
    would raise a length-mismatch when any row was skipped during feature extraction.
    """
    pytest.importorskip("matplotlib")
    from mood_reader.evaluate import train_evaluate_labeled_split

    csv_path = tmp_path / "labeled.csv"
    _make_dataset(60, blanks=8).to_csv(csv_path, index=False)

    train_evaluate_labeled_split(
        csv_path,
        train_size=45,
        test_size=15,
        model_dir=tmp_path / "model",
        random_state=1,
        include_spotify=True,
    )

    test_split = pd.read_csv(tmp_path / "model" / "test_split.csv")
    test_preds = pd.read_csv(tmp_path / "model" / "test_predictions.csv")

    usable = test_split["lyrics"].apply(lambda x: bool(str(x).strip()) and str(x) != "nan").sum()
    assert len(test_preds) == usable
    # At least one blank-lyric row landed in the test split, proving the alignment fix.
    assert len(test_preds) < len(test_split)

"""Train mood/energy models from a labeled song dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mood_reader.config import load_config
from mood_reader.fusion import extract_song_features, feature_column_names
from mood_reader.labels import VIBES, vibe_from_valence_energy


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Expected CSV columns:
      - track_id (optional)
      - title (optional)
      - lyrics (required)
      - audio_path (optional; empty uses lyrics-only with zero audio features)
      - vibe (optional discrete label)
      - valence (optional 0-1)
      - energy (optional 0-1)
    """
    df = pd.read_csv(csv_path)
    if "lyrics" not in df.columns:
        raise ValueError("Dataset CSV must include a 'lyrics' column")
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        lyrics = row.get("lyrics")
        if pd.isna(lyrics) or not str(lyrics).strip():
            skipped += 1
            continue

        audio_path = row.get("audio_path")
        if pd.isna(audio_path) or not str(audio_path).strip():
            audio_path = None
        elif not Path(str(audio_path)).exists():
            audio_path = None

        features = extract_song_features(
            lyrics=str(lyrics),
            audio_path=audio_path,
        )
        features["track_id"] = row.get("track_id", "")
        features["title"] = row.get("title", "")
        if "vibe" in row and pd.notna(row["vibe"]):
            features["vibe"] = row["vibe"]
        if "valence" in row and pd.notna(row["valence"]):
            features["valence"] = float(row["valence"])
        if "energy" in row and pd.notna(row["energy"]):
            features["energy"] = float(row["energy"])
        rows.append(features)

    if skipped:
        print(f"Skipped {skipped} row(s) with missing lyrics", file=sys.stderr)

    return pd.DataFrame(rows)


def _make_regressor(cfg: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=cfg["random_state"],
        )),
    ])


def _make_classifier(cfg: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=cfg["random_state"],
        )),
    ])


def train_from_dataframe(feat_df: pd.DataFrame, model_dir: str | Path) -> dict:
    cfg = load_config()
    model_cfg = cfg["model"]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = feature_column_names()
    X = feat_df[feature_cols].astype(float)

    report: dict = {"samples": len(feat_df), "feature_count": len(feature_cols)}

    # --- valence regressor ---
    if "valence" in feat_df.columns and feat_df["valence"].notna().sum() >= 10:
        y_val = feat_df["valence"].astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_val, test_size=0.2, random_state=42)
        val_model = _make_regressor(model_cfg)
        val_model.fit(X_tr, y_tr)
        preds = val_model.predict(X_te)
        report["valence_mae"] = float(mean_absolute_error(y_te, preds))
        joblib.dump(val_model, model_dir / "valence_model.joblib")
    else:
        val_model = None
        report["valence_mae"] = None

    # --- energy regressor ---
    if "energy" in feat_df.columns and feat_df["energy"].notna().sum() >= 10:
        y_eng = feat_df["energy"].astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_eng, test_size=0.2, random_state=42)
        eng_model = _make_regressor(model_cfg)
        eng_model.fit(X_tr, y_tr)
        preds = eng_model.predict(X_te)
        report["energy_mae"] = float(mean_absolute_error(y_te, preds))
        joblib.dump(eng_model, model_dir / "energy_model.joblib")
    else:
        eng_model = None
        report["energy_mae"] = None

    # --- vibe classifier ---
    vibe_col = feat_df["vibe"] if "vibe" in feat_df.columns else None
    if vibe_col is None or vibe_col.notna().sum() < 10:
        if val_model and eng_model:
            inferred = [
                vibe_from_valence_energy(v, e)
                for v, e in zip(feat_df["valence"], feat_df["energy"])
            ]
            feat_df = feat_df.copy()
            feat_df["vibe"] = inferred
            vibe_col = feat_df["vibe"]

    if vibe_col is not None and vibe_col.notna().sum() >= 10:
        y_vibe = vibe_col.astype(str)
        split_kwargs = {"test_size": 0.2, "random_state": 42}
        class_counts = y_vibe.value_counts()
        min_class_count = int(class_counts.min())
        test_count = max(int(round(len(y_vibe) * split_kwargs["test_size"])), 1)
        if min_class_count >= 2 and test_count >= len(class_counts):
            split_kwargs["stratify"] = y_vibe
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_vibe, **split_kwargs)
        vibe_model = _make_classifier(model_cfg)
        vibe_model.fit(X_tr, y_tr)
        preds = vibe_model.predict(X_te)
        report["vibe_accuracy"] = float(accuracy_score(y_te, preds))
        report["vibe_report"] = classification_report(y_te, preds, zero_division=0)
        joblib.dump(vibe_model, model_dir / "vibe_model.joblib")
        with open(model_dir / "vibe_classes.json", "w") as f:
            json.dump(sorted(str(v) for v in y_vibe.unique()), f, indent=2)
    else:
        report["vibe_accuracy"] = None

    meta = {
        "feature_columns": feature_cols,
        "vibes": VIBES,
        "report": report,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return report


def train(csv_path: str | Path, model_dir: str | Path | None = None) -> dict:
    cfg = load_config()
    model_dir = Path(model_dir or cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    feat_df = build_feature_matrix(df)
    feat_df.to_csv(model_dir / "training_features.csv", index=False)
    return train_from_dataframe(feat_df, model_dir)

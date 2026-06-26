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

        try:
            features = extract_song_features(
                lyrics=str(lyrics),
                audio_path=audio_path,
            )
        except ValueError:
            # Lyrics that clean down to nothing (e.g. only section headers).
            skipped += 1
            continue

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
        print(f"Skipped {skipped} row(s) with empty or unusable lyrics", file=sys.stderr)

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

    # Coerce labels to numeric up front so partially-labeled datasets (e.g.
    # merge-kaggle --keep-unmatched) don't push NaN/blank values into model.fit.
    valence_num = (
        pd.to_numeric(feat_df["valence"], errors="coerce")
        if "valence" in feat_df.columns
        else None
    )
    energy_num = (
        pd.to_numeric(feat_df["energy"], errors="coerce")
        if "energy" in feat_df.columns
        else None
    )

    def _fit_regressor(target: str, y_num: pd.Series | None) -> Pipeline | None:
        if y_num is None or y_num.notna().sum() < 10:
            report[f"{target}_mae"] = None
            return None
        mask = y_num.notna()
        X_lab = X.loc[mask]
        y_lab = y_num.loc[mask].astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(X_lab, y_lab, test_size=0.2, random_state=42)
        model = _make_regressor(model_cfg)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        report[f"{target}_mae"] = float(mean_absolute_error(y_te, preds))
        joblib.dump(model, model_dir / f"{target}_model.joblib")
        return model

    # --- valence regressor ---
    val_model = _fit_regressor("valence", valence_num)

    # --- energy regressor ---
    eng_model = _fit_regressor("energy", energy_num)

    # --- vibe classifier ---
    def _valid_vibe_mask(series: pd.Series) -> pd.Series:
        stripped = series.astype(str).str.strip()
        return series.notna() & (stripped != "") & (stripped.str.lower() != "nan")

    vibe_col = feat_df["vibe"] if "vibe" in feat_df.columns else None
    vibe_mask = _valid_vibe_mask(vibe_col) if vibe_col is not None else None

    if vibe_mask is None or vibe_mask.sum() < 10:
        if val_model is not None and eng_model is not None:
            inferred = [
                vibe_from_valence_energy(v, e) if pd.notna(v) and pd.notna(e) else float("nan")
                for v, e in zip(valence_num, energy_num)
            ]
            vibe_col = pd.Series(inferred, index=feat_df.index)
            vibe_mask = _valid_vibe_mask(vibe_col)

    report["vibe_accuracy"] = None
    y_vibe = vibe_col.loc[vibe_mask].astype(str) if vibe_mask is not None else None
    # A classifier needs at least two distinct labels to train.
    if y_vibe is not None and len(y_vibe) >= 10 and y_vibe.nunique() >= 2:
        X_vibe = X.loc[vibe_mask]
        split_kwargs = {"test_size": 0.2, "random_state": 42}
        class_counts = y_vibe.value_counts()
        min_class_count = int(class_counts.min())
        test_count = max(int(round(len(y_vibe) * split_kwargs["test_size"])), 1)
        if min_class_count >= 2 and test_count >= len(class_counts):
            split_kwargs["stratify"] = y_vibe
        X_tr, X_te, y_tr, y_te = train_test_split(X_vibe, y_vibe, **split_kwargs)
        vibe_model = _make_classifier(model_cfg)
        vibe_model.fit(X_tr, y_tr)
        preds = vibe_model.predict(X_te)
        report["vibe_accuracy"] = float(accuracy_score(y_te, preds))
        report["vibe_report"] = classification_report(y_te, preds, zero_division=0)
        joblib.dump(vibe_model, model_dir / "vibe_model.joblib")
        vibe_classes = [str(c) for c in sorted(y_vibe.unique())]
        with open(model_dir / "vibe_classes.json", "w") as f:
            json.dump(vibe_classes, f, indent=2)

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

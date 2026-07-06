"""Train/evaluate on a labeled train/test split with charts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mood_reader.config import load_config
from mood_reader.fusion import feature_column_names, lyrics_feature_column_names, spotify_feature_names
from mood_reader.labels import VIBES
from mood_reader.train import build_feature_matrix


def _feature_cols_for_target(target: str, *, include_spotify: bool) -> list[str]:
    """Lyrics + Spotify inputs; exclude the target Spotify column to avoid leakage."""
    cols = lyrics_feature_column_names()
    if not include_spotify:
        return cols
    exclude = {
        "valence": "sp_valence",
        "energy": "sp_energy",
        "tempo": "sp_tempo",
    }.get(target)
    for name in spotify_feature_names():
        if name != exclude:
            cols.append(name)
    return cols


def _zero_spotify_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in spotify_feature_names():
        if name in out.columns:
            out[name] = 0.0
    return out


def _model_cfg_for_size(base_cfg: dict, n_train: int) -> dict:
    """Use simpler models on small datasets to reduce overfitting."""
    cfg = dict(base_cfg)
    if n_train <= 150:
        cfg["n_estimators"] = min(cfg.get("n_estimators", 300), 100)
        cfg["max_depth"] = min(cfg.get("max_depth", 8), 4)
        cfg["learning_rate"] = max(cfg.get("learning_rate", 0.05), 0.08)
    return cfg


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


def _scatter_actual_pred(ax, y_true, y_pred, *, title: str) -> None:
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="none")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Actual (music)")
    ax.set_ylabel("Predicted (from lyrics)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def split_labeled_dataset(
    df: pd.DataFrame,
    *,
    train_size: int,
    test_size: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size is None:
        test_size = len(df) - train_size
    if train_size + test_size != len(df):
        raise ValueError(
            f"train_size ({train_size}) + test_size ({test_size}) must equal dataset size ({len(df)})"
        )

    stratify = df["vibe"] if "vibe" in df.columns else None
    if stratify is not None:
        counts = stratify.value_counts()
        if int(counts.min()) < 2 or test_size < len(counts):
            stratify = None

    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_evaluate_labeled_split(
    data_csv: str | Path,
    *,
    train_size: int = 100,
    test_size: int | None = None,
    model_dir: str | Path | None = None,
    charts_dir: str | Path | None = None,
    random_state: int = 42,
    include_spotify: bool = True,
) -> dict:
    """
    Train on N labeled songs, test on held-out labeled songs.
    With include_spotify=True, uses lyrics + Spotify audio features jointly.
    """
    cfg = load_config()
    base_model_cfg = cfg["model"]
    model_dir = Path(model_dir or cfg["paths"]["model_dir"])
    charts_dir = Path(charts_dir or model_dir / "charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    if "valence" not in df.columns and "audio_valence" in df.columns:
        df["valence"] = df["audio_valence"]
    if "mode" not in df.columns and "audio_mode" in df.columns:
        df["mode"] = df["audio_mode"]

    required = {"lyrics", "valence", "energy", "vibe", "artist", "title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df = df.dropna(subset=["valence", "energy", "vibe"]).copy()
    if include_spotify and "tempo" not in df.columns:
        raise ValueError("Dataset needs Spotify columns (tempo, danceability, etc.) for multimodal training")

    train_df, test_df = split_labeled_dataset(
        df, train_size=train_size, test_size=test_size, random_state=random_state
    )

    train_df.to_csv(model_dir / "train_split.csv", index=False)
    test_df.to_csv(model_dir / "test_split.csv", index=False)

    n_train = len(train_df)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    feat_df = build_feature_matrix(combined)
    # build_feature_matrix preserves source row indices and may drop rows with
    # empty/unusable lyrics, so split by original index rather than by position.
    train_feat = feat_df[feat_df.index < n_train].copy()
    test_feat = feat_df[feat_df.index >= n_train].copy()
    # Realign the raw test rows to exactly the test rows that survived feature
    # extraction, so predictions line up with test_df in test_predictions.csv.
    test_df = test_df.iloc[[i - n_train for i in test_feat.index]].reset_index(drop=True)

    val_cols = _feature_cols_for_target("valence", include_spotify=include_spotify)
    eng_cols = _feature_cols_for_target("energy", include_spotify=include_spotify)
    tempo_cols = _feature_cols_for_target("tempo", include_spotify=include_spotify)
    vibe_cols = _feature_cols_for_target("vibe", include_spotify=include_spotify)

    X_val_tr = train_feat[val_cols].astype(float)
    X_eng_tr = train_feat[eng_cols].astype(float)
    X_tempo_tr = train_feat[tempo_cols].astype(float)
    X_vibe_tr = train_feat[vibe_cols].astype(float)

    X_val_te = test_feat[val_cols].astype(float)
    X_eng_te = test_feat[eng_cols].astype(float)
    X_tempo_te = test_feat[tempo_cols].astype(float)
    X_vibe_te = test_feat[vibe_cols].astype(float)

    if include_spotify:
        X_val_te_lyrics = _zero_spotify_columns(test_feat)[val_cols].astype(float)
        X_eng_te_lyrics = _zero_spotify_columns(test_feat)[eng_cols].astype(float)
        X_vibe_te_lyrics = _zero_spotify_columns(test_feat)[vibe_cols].astype(float)
    else:
        X_val_te_lyrics = X_val_te
        X_eng_te_lyrics = X_eng_te
        X_vibe_te_lyrics = X_vibe_te

    y_val_tr = train_feat["valence"].astype(float)
    y_eng_tr = train_feat["energy"].astype(float)
    y_vibe_tr = train_feat["vibe"].astype(str)
    y_tempo_tr = train_feat["tempo"].astype(float) if "tempo" in train_feat.columns else None

    y_val_te = test_feat["valence"].astype(float)
    y_eng_te = test_feat["energy"].astype(float)
    y_vibe_te = test_feat["vibe"].astype(str)
    y_tempo_te = test_feat["tempo"].astype(float) if "tempo" in test_feat.columns else None

    model_cfg = _model_cfg_for_size(base_model_cfg, len(train_feat))

    val_model = _make_regressor(model_cfg)
    eng_model = _make_regressor(model_cfg)
    vibe_model = _make_classifier(model_cfg)
    tempo_model = _make_regressor(model_cfg) if y_tempo_tr is not None else None

    val_model.fit(X_val_tr, y_val_tr)
    eng_model.fit(X_eng_tr, y_eng_tr)
    vibe_model.fit(X_vibe_tr, y_vibe_tr)
    if tempo_model is not None and y_tempo_tr is not None:
        tempo_model.fit(X_tempo_tr, y_tempo_tr)

    joblib.dump(val_model, model_dir / "valence_model.joblib")
    joblib.dump(eng_model, model_dir / "energy_model.joblib")
    joblib.dump(vibe_model, model_dir / "vibe_model.joblib")
    if tempo_model is not None:
        joblib.dump(tempo_model, model_dir / "tempo_model.joblib")
    with open(model_dir / "vibe_classes.json", "w") as f:
        json.dump(sorted(y_vibe_tr.unique()), f)

    val_pred = np.clip(val_model.predict(X_val_te), 0, 1)
    eng_pred = np.clip(eng_model.predict(X_eng_te), 0, 1)
    vibe_pred = vibe_model.predict(X_vibe_te)
    tempo_pred = tempo_model.predict(X_tempo_te) if tempo_model is not None and y_tempo_te is not None else None

    val_pred_lyrics = np.clip(val_model.predict(X_val_te_lyrics), 0, 1)
    eng_pred_lyrics = np.clip(eng_model.predict(X_eng_te_lyrics), 0, 1)
    vibe_pred_lyrics = vibe_model.predict(X_vibe_te_lyrics)

    vibe_report = classification_report(y_vibe_te, vibe_pred, zero_division=0)
    vibe_report_lyrics = classification_report(y_vibe_te, vibe_pred_lyrics, zero_division=0)

    report = {
        "train_songs": len(train_feat),
        "test_songs": len(test_feat),
        "include_spotify_features": include_spotify,
        "valence_mae_test": float(mean_absolute_error(y_val_te, val_pred)),
        "valence_r2_test": float(r2_score(y_val_te, val_pred)),
        "energy_mae_test": float(mean_absolute_error(y_eng_te, eng_pred)),
        "energy_r2_test": float(r2_score(y_eng_te, eng_pred)),
        "vibe_accuracy_test": float(accuracy_score(y_vibe_te, vibe_pred)),
        "valence_mae_test_lyrics_only": float(mean_absolute_error(y_val_te, val_pred_lyrics)),
        "energy_mae_test_lyrics_only": float(mean_absolute_error(y_eng_te, eng_pred_lyrics)),
        "vibe_accuracy_test_lyrics_only": float(accuracy_score(y_vibe_te, vibe_pred_lyrics)),
        "vibe_report_test": vibe_report,
        "vibe_report_test_lyrics_only": vibe_report_lyrics,
        "model_config_used": model_cfg,
        "feature_columns": {
            "valence": val_cols,
            "energy": eng_cols,
            "tempo": tempo_cols,
            "vibe": vibe_cols,
        },
    }
    if tempo_pred is not None and y_tempo_te is not None:
        report["tempo_mae_test"] = float(mean_absolute_error(y_tempo_te, tempo_pred))
        report["tempo_r2_test"] = float(r2_score(y_tempo_te, tempo_pred))

    test_out = test_df.copy()
    test_out["pred_valence"] = val_pred
    test_out["pred_energy"] = eng_pred
    test_out["pred_vibe"] = vibe_pred
    test_out["pred_valence_lyrics_only"] = val_pred_lyrics
    test_out["pred_energy_lyrics_only"] = eng_pred_lyrics
    test_out["pred_vibe_lyrics_only"] = vibe_pred_lyrics
    if tempo_pred is not None:
        test_out["pred_tempo"] = tempo_pred
    test_out.to_csv(model_dir / "test_predictions.csv", index=False)
    train_feat.to_csv(model_dir / "training_features.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _scatter_actual_pred(
        axes[0], y_val_te.values, val_pred,
        title=f"Valence (test n={len(y_val_te)}, MAE={report['valence_mae_test']:.3f})",
    )
    _scatter_actual_pred(
        axes[1], y_eng_te.values, eng_pred,
        title=f"Energy (test n={len(y_eng_te)}, MAE={report['energy_mae_test']:.3f})",
    )
    fig.suptitle(
        "Lyrics + Spotify → music mood on held-out test set"
        if include_spotify
        else "Lyrics → music mood on held-out test set",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(charts_dir / "test_valence_energy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_val_te, y_eng_te, c="tab:blue", alpha=0.6, label="Actual music mood", s=40)
    ax.scatter(val_pred, eng_pred, c="tab:orange", alpha=0.6, label="Predicted (lyrics+Spotify)", s=40)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Energy")
    ax.set_title("Russell circumplex: actual vs predicted (test set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "test_circumplex.png", dpi=150)
    plt.close(fig)

    labels = sorted(set(y_vibe_te) | set(vibe_pred), key=lambda x: VIBES.index(x) if x in VIBES else 99)
    cm = confusion_matrix(y_vibe_te, vibe_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted vibe")
    ax.set_ylabel("Actual vibe")
    ax.set_title(f"Vibe confusion matrix (accuracy={report['vibe_accuracy_test']:.1%})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(charts_dir / "test_vibe_confusion.png", dpi=150)
    plt.close(fig)

    metrics = ["Valence MAE", "Energy MAE", "Vibe accuracy"]
    values = [
        report["valence_mae_test"],
        report["energy_mae_test"],
        report["vibe_accuracy_test"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(metrics, values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylim(0, max(max(values), 0.5) * 1.15)
    ax.set_title("Test set metrics (lower MAE better; higher accuracy better)")
    for bar, val in zip(bars, values):
        label = f"{val:.3f}" if val < 1 else f"{val:.1%}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(charts_dir / "test_metrics_summary.png", dpi=150)
    plt.close(fig)

    meta = {
        "feature_columns": report["feature_columns"],
        "vibes": VIBES,
        "include_spotify_features": include_spotify,
        "report": {k: v for k, v in report.items() if not k.startswith("vibe_report")},
        "charts_dir": str(charts_dir),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    with open(charts_dir / "report.json", "w") as f:
        json.dump({k: v for k, v in report.items() if k != "vibe_report_test"}, f, indent=2, default=str)
    with open(charts_dir / "vibe_classification_report.txt", "w") as f:
        f.write(vibe_report)

    return report


def train_evaluate_and_plot(
    train_csv: str | Path,
    full_csv: str | Path,
    **kwargs,
) -> dict:
    """Legacy entry: use labeled split on train_csv if it has enough labeled rows."""
    del full_csv, kwargs
    df = pd.read_csv(train_csv)
    n = len(df)
    train_size = min(100, n - 10)
    return train_evaluate_labeled_split(train_csv, train_size=train_size, test_size=n - train_size)

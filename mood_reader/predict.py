"""Run mood/energy inference on a song."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mood_reader.config import load_config
from mood_reader.fusion import extract_song_features, feature_column_names
from mood_reader.labels import energy_bucket, valence_bucket, vibe_from_valence_energy


class MoodReader:
    def __init__(self, model_dir: str | Path | None = None):
        cfg = load_config()
        self.model_dir = Path(model_dir or cfg["paths"]["model_dir"])
        self.feature_columns = feature_column_names()
        self.feature_columns_by_target: dict[str, list[str]] = {}

        meta_path = self.model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            fc = meta.get("feature_columns")
            if isinstance(fc, dict):
                self.feature_columns_by_target = fc
            elif isinstance(fc, list):
                self.feature_columns = fc

        self.valence_model = self._load("valence_model.joblib")
        self.energy_model = self._load("energy_model.joblib")
        self.vibe_model = self._load("vibe_model.joblib")

    def _load(self, name: str):
        path = self.model_dir / name
        if path.exists():
            return joblib.load(path)
        return None

    def _vectorize(self, features: dict[str, float], columns: list[str] | None = None) -> pd.DataFrame:
        cols = columns or self.feature_columns
        row = {c: features.get(c, 0.0) for c in cols}
        return pd.DataFrame([row], columns=cols)

    def predict(
        self,
        *,
        lyrics: str,
        audio_path: str | Path | None = None,
        spotify: dict | None = None,
    ) -> dict:
        features = extract_song_features(
            lyrics=lyrics,
            audio_path=audio_path,
            spotify_row=spotify,
        )
        val_cols = self.feature_columns_by_target.get("valence", self.feature_columns)
        eng_cols = self.feature_columns_by_target.get("energy", self.feature_columns)
        vibe_cols = self.feature_columns_by_target.get("vibe", self.feature_columns)

        X_val = self._vectorize(features, val_cols)
        X_eng = self._vectorize(features, eng_cols)
        X_vibe = self._vectorize(features, vibe_cols)

        result: dict = {
            "features_used": len(set(val_cols) | set(eng_cols) | set(vibe_cols)),
            "spotify_features_used": spotify is not None,
        }

        if self.valence_model is not None:
            valence = float(np.clip(self.valence_model.predict(X_val)[0], 0, 1))
        else:
            valence = float(np.clip(features.get("vader_compound", 0) * 0.5 + 0.5, 0, 1))

        if self.energy_model is not None:
            energy = float(np.clip(self.energy_model.predict(X_eng)[0], 0, 1))
        else:
            tempo = features.get("tempo_bpm", 0)
            rms = features.get("rms_mean", 0)
            energy = float(np.clip(0.4 * (tempo / 180.0) + 0.6 * min(rms * 10, 1.0), 0, 1))

        if self.vibe_model is not None:
            vibe = str(self.vibe_model.predict(X_vibe)[0])
            if hasattr(self.vibe_model, "predict_proba"):
                probs = self.vibe_model.predict_proba(X_vibe)[0]
                classes = list(self.vibe_model.named_steps["model"].classes_)
                vibe_scores = {c: float(p) for c, p in zip(classes, probs)}
            else:
                vibe_scores = {vibe: 1.0}
        else:
            vibe = vibe_from_valence_energy(valence, energy)
            vibe_scores = {vibe: 1.0}

        result.update({
            "valence": round(valence, 3),
            "energy": round(energy, 3),
            "valence_label": valence_bucket(valence),
            "energy_label": energy_bucket(energy),
            "vibe": vibe,
            "vibe_scores": {k: round(v, 3) for k, v in sorted(vibe_scores.items(), key=lambda x: -x[1])},
            "audio_analyzed": audio_path is not None,
        })
        return result


def predict_song(
    lyrics: str,
    audio_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    spotify: dict | None = None,
) -> dict:
    return MoodReader(model_dir).predict(lyrics=lyrics, audio_path=audio_path, spotify=spotify)

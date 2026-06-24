"""Fuse lyrics + audio features into a single feature vector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mood_reader.audio_features import audio_feature_names, extract_audio_features
from mood_reader.lyrics_features import extract_lyrics_features, lyrics_feature_names


def extract_song_features(
    *,
    lyrics: str,
    audio_path: str | Path | None = None,
) -> dict[str, float]:
    features = extract_lyrics_features(lyrics)
    if audio_path is not None:
        features.update(extract_audio_features(audio_path))
    else:
        for name in audio_feature_names():
            features[name] = 0.0
    return features


def feature_column_names() -> list[str]:
    return lyrics_feature_names() + audio_feature_names()


def features_to_row(features: dict[str, float]) -> pd.Series:
    cols = feature_column_names()
    return pd.Series({c: features.get(c, 0.0) for c in cols})

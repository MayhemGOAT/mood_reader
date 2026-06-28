"""Fuse lyrics + audio + Spotify metadata into a single feature vector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mood_reader.audio_features import audio_feature_names, extract_audio_features
from mood_reader.lyrics_features import extract_lyrics_features, lyrics_feature_names

# Spotify / Kaggle audio-analysis columns (prefixed sp_ in feature vectors)
SPOTIFY_FEATURE_COLUMNS = [
    "sp_valence",
    "sp_energy",
    "sp_tempo",
    "sp_danceability",
    "sp_acousticness",
    "sp_instrumentalness",
    "sp_speechiness",
    "sp_liveness",
    "sp_loudness",
    "sp_key",
    "sp_mode",
    "sp_time_signature",
    "sp_duration_ms",
]

_ROW_TO_SPOTIFY = {
    "valence": "sp_valence",
    "audio_valence": "sp_valence",
    "energy": "sp_energy",
    "tempo": "sp_tempo",
    "danceability": "sp_danceability",
    "acousticness": "sp_acousticness",
    "instrumentalness": "sp_instrumentalness",
    "speechiness": "sp_speechiness",
    "liveness": "sp_liveness",
    "loudness": "sp_loudness",
    "key": "sp_key",
    "mode": "sp_mode",
    "audio_mode": "sp_mode",
    "time_signature": "sp_time_signature",
    "song_duration_ms": "sp_duration_ms",
}


def spotify_feature_names() -> list[str]:
    return list(SPOTIFY_FEATURE_COLUMNS)


def extract_spotify_features(row: pd.Series | dict) -> dict[str, float]:
    """Read Spotify-style numeric columns from a dataset row."""
    if isinstance(row, pd.Series):
        row = row.to_dict()

    out = {name: 0.0 for name in SPOTIFY_FEATURE_COLUMNS}
    for src, dst in _ROW_TO_SPOTIFY.items():
        val = row.get(src)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            out[dst] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def extract_song_features(
    *,
    lyrics: str,
    audio_path: str | Path | None = None,
    spotify_row: pd.Series | dict | None = None,
) -> dict[str, float]:
    features = extract_lyrics_features(lyrics)
    if audio_path is not None:
        features.update(extract_audio_features(audio_path))
    else:
        for name in audio_feature_names():
            features[name] = 0.0
    if spotify_row is not None:
        features.update(extract_spotify_features(spotify_row))
    return features


def lyrics_feature_column_names() -> list[str]:
    return lyrics_feature_names() + audio_feature_names()


def feature_column_names(*, include_spotify: bool = True) -> list[str]:
    cols = lyrics_feature_column_names()
    if include_spotify:
        cols = cols + spotify_feature_names()
    return cols


def features_to_row(features: dict[str, float], *, include_spotify: bool = True) -> pd.Series:
    cols = feature_column_names(include_spotify=include_spotify)
    return pd.Series({c: features.get(c, 0.0) for c in cols})

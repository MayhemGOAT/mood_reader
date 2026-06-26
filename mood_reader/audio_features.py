"""Extract audio features for mood/energy classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mood_reader.config import load_config


def extract_audio_features(
    audio_path: str | Path,
    *,
    sample_rate: int | None = None,
    duration_seconds: float | None = None,
    n_mfcc: int | None = None,
) -> dict[str, float]:
    # Imported lazily so the lyrics-only pipeline (train/predict without audio)
    # does not require librosa and its heavy native dependencies.
    import librosa

    cfg = load_config()["audio"]
    sr = sample_rate or cfg["sample_rate"]
    duration = duration_seconds or cfg["duration_seconds"]
    n_mfcc = n_mfcc or cfg["n_mfcc"]

    y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    if len(y) == 0:
        raise ValueError(f"Could not load audio from {audio_path}")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    features: dict[str, float] = {
        "tempo_bpm": tempo,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_contrast_mean": float(np.mean(spectral_contrast)),
        "chroma_mean": float(np.mean(chroma)),
        "chroma_std": float(np.std(chroma)),
    }

    for i in range(n_mfcc):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

    return features


def audio_feature_names(n_mfcc: int | None = None) -> list[str]:
    cfg = load_config()["audio"]
    n_mfcc = n_mfcc or cfg["n_mfcc"]
    base = [
        "tempo_bpm",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean",
        "spectral_contrast_mean",
        "chroma_mean",
        "chroma_std",
    ]
    mfcc_names = [f"mfcc_{i}_{stat}" for i in range(n_mfcc) for stat in ("mean", "std")]
    return base + mfcc_names

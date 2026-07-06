"""Download Kaggle Spotify features and merge with lyrical CSV datasets."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from mood_reader.apis.credentials import configure_kaggle_env, get_credentials, require_kaggle
from mood_reader.clean import has_usable_lyrics
from mood_reader.labels import vibe_from_valence_energy

DEFAULT_DATASET = "joebeachcapital/30000-spotify-songs"

KAGGLE_FEATURE_COLUMNS = [
    "valence",
    "energy",
    "tempo",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "key",
    "mode",
]

KAGGLE_TO_OUR = {
    "track_id": "spotify_id",
    "track_name": "kaggle_track_name",
    "track_artist": "kaggle_artist",
    "track_album_name": "album",
}


def _normalize(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    # Fold accents (é -> e) so "Beyoncé" and "Beyonce" match during the join.
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", folded.lower())


def _merge_key(artist: object, title: object) -> str:
    return f"{_normalize(artist)}|{_normalize(title)}"


def find_kaggle_csv(dataset_dir: str | Path) -> Path:
    dataset_dir = Path(dataset_dir)
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found under {dataset_dir}")
    if len(csv_files) == 1:
        return csv_files[0]
    for path in csv_files:
        if "spotify" in path.name.lower():
            return path
    return csv_files[0]


def download_kaggle_dataset(
    dataset: str = DEFAULT_DATASET,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download dataset via kagglehub; returns path to directory containing CSV."""
    import os

    configure_kaggle_env()
    require_kaggle(get_credentials())

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = str(Path(cache_dir).resolve())

    import kagglehub

    path = kagglehub.dataset_download(dataset)
    return Path(path)


def load_kaggle_spotify_features(csv_path: str | Path) -> pd.DataFrame:
    """Load and dedupe Kaggle Spotify song features."""
    df = pd.read_csv(csv_path)
    required = {"track_name", "track_artist", "track_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kaggle CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["_merge_key"] = df.apply(
        lambda row: _merge_key(row["track_artist"], row["track_name"]),
        axis=1,
    )
    df = df.drop_duplicates(subset=["_merge_key"], keep="first")
    return df


def merge_lyrics_with_kaggle(
    lyrics_csv: str | Path,
    kaggle_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join lyrical rows with Kaggle audio-feature columns on artist + title."""
    lyrics_df = pd.read_csv(lyrics_csv).copy()
    lyrics_df["_merge_key"] = lyrics_df.apply(
        lambda row: _merge_key(row.get("artist"), row.get("title")),
        axis=1,
    )

    drop_from_lyrics = [
        c for c in ["spotify_id", "album", *KAGGLE_FEATURE_COLUMNS, "label_source"]
        if c in lyrics_df.columns
    ]
    lyrics_df = lyrics_df.drop(columns=drop_from_lyrics)

    feature_cols = [c for c in KAGGLE_FEATURE_COLUMNS if c in kaggle_df.columns]
    rename_cols = {src: dst for src, dst in KAGGLE_TO_OUR.items() if src in kaggle_df.columns}
    kaggle_subset = kaggle_df[list(rename_cols.keys()) + feature_cols + ["_merge_key"]].rename(
        columns=rename_cols
    )

    merged = lyrics_df.merge(kaggle_subset, on="_merge_key", how="left")
    merged = merged.drop(columns=["_merge_key"])

    merged["label_source"] = merged["spotify_id"].apply(
        lambda x: "kaggle" if str(x).strip() and str(x).lower() != "nan" else ""
    )

    def _fill_vibe(row: pd.Series) -> str:
        existing = str(row.get("vibe") or "").strip()
        if existing and existing.lower() != "nan":
            return existing
        if pd.notna(row.get("valence")) and pd.notna(row.get("energy")):
            return vibe_from_valence_energy(float(row["valence"]), float(row["energy"]))
        return ""

    if "vibe" not in merged.columns:
        merged["vibe"] = ""
    merged["vibe"] = merged.apply(_fill_vibe, axis=1)

    return merged


def filter_lyrics_and_music(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with usable lyrics and Kaggle/Spotify feature labels."""
    mask = df.apply(
        lambda row: has_usable_lyrics(row.get("lyrics"))
        and str(row.get("spotify_id") or "").strip() not in ("", "nan")
        and pd.notna(row.get("valence"))
        and pd.notna(row.get("energy")),
        axis=1,
    )
    return df.loc[mask].copy()


def merge_kaggle_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    dataset: str = DEFAULT_DATASET,
    kaggle_csv: str | Path | None = None,
    cache_dir: str | Path | None = None,
    clean_only: bool = True,
) -> dict:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if kaggle_csv:
        kaggle_path = Path(kaggle_csv)
        dataset_dir = kaggle_path.parent
    else:
        dataset_dir = download_kaggle_dataset(dataset, cache_dir=cache_dir)
        kaggle_path = find_kaggle_csv(dataset_dir)

    kaggle_df = load_kaggle_spotify_features(kaggle_path)
    merged = merge_lyrics_with_kaggle(input_csv, kaggle_df)

    if clean_only:
        out_df = filter_lyrics_and_music(merged)
    else:
        out_df = merged

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    summary = {
        "input": str(input_csv),
        "output": str(output_csv),
        "kaggle_csv": str(kaggle_path),
        "kaggle_unique_tracks": len(kaggle_df),
        "input_rows": len(merged),
        "output_rows": len(out_df),
        "matched_with_features": int((merged["label_source"] == "kaggle").sum()),
        "clean_only": clean_only,
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

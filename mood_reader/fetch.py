"""Fetch songs from Genius + Spotify + Last.fm and build training datasets."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mood_reader.apis.song_lookup import download_preview, lookup_song
from mood_reader.config import load_config


def parse_track_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if " - " in line:
        artist, title = line.split(" - ", 1)
    elif "|" in line:
        artist, title = line.split("|", 1)
    elif "," in line:
        artist, title = line.split(",", 1)
    else:
        return None

    artist, title = artist.strip(), title.strip()
    if artist and title:
        return artist, title
    return None


def load_track_list(path: str | Path) -> list[tuple[str, str]]:
    tracks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parsed = parse_track_line(line)
        if parsed:
            tracks.append(parsed)
    return tracks


def fetch_songs(
    tracks: list[tuple[str, str]],
    *,
    download_previews: bool = False,
    preview_dir: str | Path | None = None,
) -> list[dict]:
    cfg = load_config()
    preview_dir = Path(preview_dir or cfg["paths"]["data_dir"]) / "previews"
    rows = []

    for artist, title in tqdm(tracks, desc="Fetching songs"):
        info = lookup_song(title, artist)
        row = info.to_dataset_row()

        if download_previews and info.preview_url:
            preview = download_preview(info, preview_dir)
            if preview:
                row["audio_path"] = str(preview)

        row["fetch_errors"] = "; ".join(info.errors)
        row["sources"] = "|".join(info.sources)
        rows.append(row)

    return rows


def save_dataset(rows: list[dict], output_path: str | Path, *, append: bool = False) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame(rows)
    if append and output_path.exists():
        df_old = pd.read_csv(output_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["artist", "title"], keep="last")
    else:
        df = df_new

    df.to_csv(output_path, index=False)
    return output_path


def fetch_tracks_to_csv(
    tracks: list[tuple[str, str]],
    output_path: str | Path,
    *,
    append: bool = False,
    download_previews: bool = False,
) -> Path:
    rows = fetch_songs(tracks, download_previews=download_previews)
    return save_dataset(rows, output_path, append=append)

"""Export song lists and fetch Spotify metadata for existing lyrical rows."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mood_reader.apis.base import SongInfo
from mood_reader.apis.song_lookup import download_preview
from mood_reader.apis.spotify import enrich_with_spotify
from mood_reader.clean import _merge_spotify_into_row, has_music_data, has_usable_lyrics
from mood_reader.config import load_config
from mood_reader.fetch import load_track_list, save_dataset


def _track_key(artist: str, title: str) -> tuple[str, str]:
    return artist.strip().lower(), title.strip().lower()


def export_songs_with_lyrics(
    input_csv: str | Path,
    output_txt: str | Path,
    *,
    require_lyrics: bool = True,
) -> dict:
    """Write 'Artist - Title' lines for rows that have usable lyrics."""
    df = pd.read_csv(input_csv)
    lines: list[str] = []

    for _, row in df.iterrows():
        if require_lyrics and not has_usable_lyrics(row.get("lyrics")):
            continue
        artist = str(row.get("artist") or "").strip()
        title = str(row.get("title") or "").strip()
        if not artist or not title or artist.lower() == "nan" or title.lower() == "nan":
            continue
        lines.append(f"{artist} - {title}")

    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "input_csv": str(input_csv),
        "output_txt": str(output_txt),
        "songs_listed": len(lines),
    }


def _row_index_by_key(df: pd.DataFrame) -> dict[tuple[str, str], int]:
    index: dict[tuple[str, str], int] = {}
    for i, row in df.iterrows():
        artist = str(row.get("artist") or "").strip()
        title = str(row.get("title") or "").strip()
        if artist and title:
            index[_track_key(artist, title)] = int(i)
    return index


def fetch_spotify_for_csv(
    data_csv: str | Path,
    *,
    tracks_file: str | Path | None = None,
    download_previews: bool = False,
    delay_seconds: float = 2.0,
    save_every: int = 25,
    skip_existing: bool = True,
) -> dict:
    """
    Look up Spotify music data for songs in the CSV (lyrics preserved).
    Optionally restrict to a track list file ('Artist - Title' per line).
    """
    data_csv = Path(data_csv)
    df = pd.read_csv(data_csv)
    rows = df.to_dict(orient="records")
    row_index = _row_index_by_key(df)

    if tracks_file:
        tracks = load_track_list(tracks_file)
    else:
        tracks = []
        for _, row in df.iterrows():
            if not has_usable_lyrics(row.get("lyrics")):
                continue
            artist = str(row.get("artist") or "").strip()
            title = str(row.get("title") or "").strip()
            if artist and title:
                tracks.append((artist, title))

    cfg = load_config()
    preview_dir = Path(cfg["paths"]["data_dir"]) / "previews"

    spotify_before = sum(
        1 for row in rows if has_music_data(pd.Series(row))
    )
    processed = 0
    enriched = 0
    skipped = 0
    missing = 0

    for artist, title in tqdm(tracks, desc="Fetching Spotify"):
        key = _track_key(artist, title)
        idx = row_index.get(key)
        if idx is None:
            missing += 1
            continue

        row = rows[idx]
        series = pd.Series(row)

        if skip_existing and has_music_data(series) and has_usable_lyrics(series.get("lyrics")):
            skipped += 1
            continue

        if not has_usable_lyrics(series.get("lyrics")):
            skipped += 1
            continue

        info = SongInfo(
            title=str(row.get("title") or title),
            artist=str(row.get("artist") or artist),
            lyrics=row.get("lyrics"),
        )
        info = enrich_with_spotify(info)

        audio_path = None
        if download_previews and info.preview_url:
            audio_path = download_preview(info, preview_dir)

        rows[idx] = _merge_spotify_into_row(series, info, audio_path).to_dict()
        processed += 1
        if info.spotify_id:
            enriched += 1

        if processed % save_every == 0:
            save_dataset(rows, data_csv, append=False)

        time.sleep(delay_seconds)

    save_dataset(rows, data_csv, append=False)

    spotify_after = sum(
        1 for row in rows if has_music_data(pd.Series(row))
    )

    summary = {
        "data_csv": str(data_csv),
        "tracks_file": str(tracks_file) if tracks_file else None,
        "tracks_requested": len(tracks),
        "processed": processed,
        "spotify_matches": enriched,
        "skipped_existing": skipped,
        "not_in_csv": missing,
        "spotify_before": spotify_before,
        "spotify_after": spotify_after,
    }
    summary_path = data_csv.with_suffix(".spotify_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

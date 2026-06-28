"""Fetch songs from Genius + Spotify + Last.fm and build training datasets."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mood_reader.apis.genius import fetch_genius_lyrics
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


def _track_key(artist: str, title: str) -> tuple[str, str]:
    return artist.strip().lower(), title.strip().lower()


def load_existing_keys(output_path: str | Path) -> set[tuple[str, str]]:
    output_path = Path(output_path)
    if not output_path.exists():
        return set()

    df = pd.read_csv(output_path)
    if "artist" not in df.columns or "title" not in df.columns:
        return set()

    keys: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        artist = row.get("artist")
        title = row.get("title")
        if pd.isna(artist) or pd.isna(title):
            continue
        keys.add(_track_key(str(artist), str(title)))
    return keys


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


def fetch_songs(
    tracks: list[tuple[str, str]],
    *,
    download_previews: bool = False,
    preview_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    resume: bool = True,
    save_every: int = 10,
) -> list[dict]:
    cfg = load_config()
    preview_dir = Path(preview_dir or cfg["paths"]["data_dir"]) / "previews"
    output_path = Path(output_path) if output_path else None

    existing_keys: set[tuple[str, str]] = set()
    if resume and output_path is not None:
        existing_keys = load_existing_keys(output_path)

    pending = [
        (artist, title)
        for artist, title in tracks
        if _track_key(artist, title) not in existing_keys
    ]

    rows: list[dict] = []
    fetched = 0

    for artist, title in tqdm(pending, desc="Fetching songs"):
        info = lookup_song(title, artist)
        row = info.to_dataset_row()

        if download_previews and info.preview_url:
            preview = download_preview(info, preview_dir)
            if preview:
                row["audio_path"] = str(preview)

        row["fetch_errors"] = "; ".join(info.errors)
        row["sources"] = "|".join(info.sources)
        rows.append(row)
        fetched += 1

        if output_path is not None and fetched % save_every == 0:
            save_dataset(rows, output_path, append=True)
            rows = []

    if output_path is not None and rows:
        save_dataset(rows, output_path, append=True)

    return rows


def fetch_tracks_to_csv(
    tracks: list[tuple[str, str]],
    output_path: str | Path,
    *,
    append: bool = False,
    download_previews: bool = False,
    resume: bool = True,
    save_every: int = 10,
) -> dict:
    output_path = Path(output_path)
    resume = resume or append

    if not resume and output_path.exists():
        output_path.unlink()

    existing_keys = load_existing_keys(output_path) if resume and output_path.exists() else set()
    pending = [t for t in tracks if _track_key(t[0], t[1]) not in existing_keys]
    skipped = len(tracks) - len(pending)

    if pending:
        fetch_songs(
            tracks,
            download_previews=download_previews,
            output_path=output_path,
            resume=resume,
            save_every=save_every,
        )

    total_saved = len(load_existing_keys(output_path)) if output_path.exists() else 0

    summary = {
        "saved": str(output_path),
        "tracks_in_list": len(tracks),
        "fetched_this_run": len(pending),
        "skipped_existing": skipped,
        "total_in_csv": total_saved,
    }

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fetch_lyrics_for_csv(
    csv_path: str | Path,
    *,
    delay_seconds: float = 1.0,
    save_every: int = 5,
) -> dict:
    """Backfill Genius lyrics for rows in CSV that are missing usable lyrics."""
    from mood_reader.clean import has_usable_lyrics

    csv_path = Path(csv_path)
    rows = pd.read_csv(csv_path).to_dict(orient="records")
    fetched = 0
    skipped = 0

    for i, row in enumerate(tqdm(rows, desc="Fetching Genius lyrics")):
        if has_usable_lyrics(row.get("lyrics")):
            skipped += 1
            continue

        artist = str(row.get("artist") or "").strip()
        title = str(row.get("title") or "").strip()
        if not artist or not title or artist.lower() == "nan" or title.lower() == "nan":
            skipped += 1
            continue

        info = fetch_genius_lyrics(title, artist)
        if info.lyrics:
            row["lyrics"] = info.lyrics
            row["genius_url"] = info.genius_url or row.get("genius_url", "")
            sources = str(row.get("sources") or "")
            if "genius" not in sources:
                row["sources"] = "genius" if not sources else f"{sources}|genius"
            fetched += 1

        rows[i] = row
        if fetched and fetched % save_every == 0:
            save_dataset(rows, csv_path, append=False)
        time.sleep(delay_seconds)

    save_dataset(rows, csv_path, append=False)
    return {
        "csv": str(csv_path),
        "fetched": fetched,
        "skipped": skipped,
        "total_rows": len(rows),
    }

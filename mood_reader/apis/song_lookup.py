"""Combine Genius, Spotify, and Last.fm into one song lookup."""

from __future__ import annotations

import tempfile
from pathlib import Path

import requests

from mood_reader.apis.base import SongInfo
from mood_reader.apis.genius import fetch_genius_lyrics
from mood_reader.apis.lastfm import enrich_with_lastfm
from mood_reader.apis.spotify import enrich_with_spotify


def lookup_song(
    title: str,
    artist: str,
    *,
    include_lyrics: bool = True,
    include_spotify: bool = True,
    include_lastfm: bool = True,
) -> SongInfo:
    if include_lyrics:
        info = fetch_genius_lyrics(title, artist)
    else:
        info = SongInfo(title=title, artist=artist)

    if include_spotify:
        info = enrich_with_spotify(info)
    if include_lastfm:
        info = enrich_with_lastfm(info)

    return info


def download_preview(info: SongInfo, dest_dir: str | Path) -> Path | None:
    """Download Spotify 30s preview MP3 for librosa analysis."""
    if not info.preview_url:
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{info.artist}_{info.title}".replace("/", "-").replace(" ", "_")
    dest = dest_dir / f"{safe_name}.mp3"

    if dest.exists():
        return dest

    resp = requests.get(info.preview_url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def lookup_with_preview(
    title: str,
    artist: str,
    *,
    preview_dir: str | Path | None = None,
) -> tuple[SongInfo, Path | None]:
    info = lookup_song(title, artist)
    preview_path = None
    if info.preview_url:
        preview_dir = preview_dir or Path(tempfile.gettempdir()) / "mood_reader_previews"
        preview_path = download_preview(info, preview_dir)
    return info, preview_path

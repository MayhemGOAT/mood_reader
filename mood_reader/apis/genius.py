"""Genius API — search songs and scrape lyrics."""

from __future__ import annotations

import re

import lyricsgenius

from mood_reader.apis.base import SongInfo
from mood_reader.apis.credentials import get_credentials, require_genius


def _clean_lyrics(raw: str, title: str) -> str:
    text = raw.strip()
    # lyricsgenius prefixes e.g. "Song Title Lyrics"
    header_pattern = rf"^{re.escape(title)}\s*Lyrics\s*\n"
    text = re.sub(header_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+\s*Contributors?.*?\n", "", text, flags=re.IGNORECASE)
    return text.strip()


def fetch_genius_lyrics(title: str, artist: str) -> SongInfo:
    token = require_genius(get_credentials())
    genius = lyricsgenius.Genius(
        access_token=token,
        remove_section_headers=True,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)", "(Acoustic)", "(Demo)"],
        timeout=15,
    )

    try:
        song = genius.search_song(title, artist)
    except AssertionError as exc:
        if "401" in str(exc) or "invalid_token" in str(exc):
            raise ValueError(
                "Genius API rejected the token. Use the **Client Access Token** from "
                "https://genius.com/api-clients (not Client ID or Client Secret). "
                "Set it as GENIUS_ACCESS_TOKEN in .env"
            ) from exc
        raise
    info = SongInfo(title=title, artist=artist)

    if song is None:
        info.errors.append(f"Genius: no match for {artist} - {title}")
        return info

    info.title = song.title or title
    info.artist = song.artist or artist
    info.genius_id = song._body.get("id")
    info.genius_url = song.url
    info.lyrics = _clean_lyrics(song.lyrics or "", info.title)
    info.sources.append("genius")
    return info

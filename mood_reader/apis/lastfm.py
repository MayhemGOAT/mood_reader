"""Last.fm API — mood/genre tags for songs."""

from __future__ import annotations

import requests

from mood_reader.apis.base import SongInfo
from mood_reader.apis.credentials import get_credentials


def enrich_with_lastfm(info: SongInfo) -> SongInfo:
    api_key = get_credentials().lastfm_api_key
    if not api_key:
        return info

    try:
        resp = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "track.getInfo",
                "api_key": api_key,
                "artist": info.artist,
                "track": info.title,
                "format": "json",
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        track = payload.get("track")
        if not track:
            info.errors.append(f"Last.fm: no match for {info.artist} - {info.title}")
            return info

        info.lastfm_url = track.get("url")
        tags = track.get("toptags", {}).get("tag", [])
        if isinstance(tags, dict):
            tags = [tags]
        info.lastfm_tags = [t.get("name", "") for t in tags if t.get("name")][:10]

        album = track.get("album")
        if album and isinstance(album, dict) and not info.album:
            info.album = album.get("title")

        if "lastfm" not in info.sources:
            info.sources.append("lastfm")
    except requests.RequestException as exc:
        info.errors.append(f"Last.fm: {exc}")

    return info

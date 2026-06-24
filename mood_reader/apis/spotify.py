"""Spotify Web API — track search and audio features."""

from __future__ import annotations

import time
from typing import Any

import requests

from mood_reader.apis.base import SongInfo
from mood_reader.apis.credentials import get_credentials, require_spotify

_TOKEN: str | None = None
_TOKEN_EXPIRES_AT: float = 0.0


def _spotify_token() -> str:
    global _TOKEN, _TOKEN_EXPIRES_AT
    if _TOKEN and time.time() < _TOKEN_EXPIRES_AT - 30:
        return _TOKEN

    client_id, client_secret = require_spotify(get_credentials())
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    _TOKEN = payload["access_token"]
    _TOKEN_EXPIRES_AT = time.time() + payload.get("expires_in", 3600)
    return _TOKEN


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_spotify_token()}"}


def _best_track(title: str, artist: str) -> dict[str, Any] | None:
    query = f"track:{title} artist:{artist}"
    resp = requests.get(
        "https://api.spotify.com/v1/search",
        params={"q": query, "type": "track", "limit": 5},
        headers=_headers(),
        timeout=20,
    )
    resp.raise_for_status()
    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        query = f"{title} {artist}"
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            params={"q": query, "type": "track", "limit": 5},
            headers=_headers(),
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json().get("tracks", {}).get("items", [])
    return items[0] if items else None


def _audio_features(track_id: str) -> dict[str, Any] | None:
    resp = requests.get(
        f"https://api.spotify.com/v1/audio-features/{track_id}",
        headers=_headers(),
        timeout=20,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def enrich_with_spotify(info: SongInfo) -> SongInfo:
    creds = get_credentials()
    if not creds.spotify_client_id or not creds.spotify_client_secret:
        info.errors.append("Spotify: credentials not configured (skipped)")
        return info

    try:
        track = _best_track(info.title, info.artist)
        if track is None:
            info.errors.append(f"Spotify: no match for {info.artist} - {info.title}")
            return info

        info.spotify_id = track["id"]
        info.title = track.get("name") or info.title
        info.album = (track.get("album") or {}).get("name")
        info.preview_url = track.get("preview_url")
        artists = track.get("artists") or []
        if artists:
            info.artist = artists[0].get("name") or info.artist

        features = _audio_features(track["id"])
        if features:
            info.valence = features.get("valence")
            info.energy = features.get("energy")
            info.tempo = features.get("tempo")
            info.danceability = features.get("danceability")
            info.acousticness = features.get("acousticness")
            info.instrumentalness = features.get("instrumentalness")
            info.speechiness = features.get("speechiness")
            info.liveness = features.get("liveness")
            info.key = features.get("key")
            info.mode = features.get("mode")

        if "spotify" not in info.sources:
            info.sources.append("spotify")
    except requests.RequestException as exc:
        info.errors.append(f"Spotify: {exc}")

    return info

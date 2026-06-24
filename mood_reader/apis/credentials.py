"""Load API credentials from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ApiCredentials:
    genius_token: str | None
    spotify_client_id: str | None
    spotify_client_secret: str | None
    lastfm_api_key: str | None


def get_credentials() -> ApiCredentials:
    return ApiCredentials(
        genius_token=os.getenv("GENIUS_ACCESS_TOKEN"),
        spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        lastfm_api_key=os.getenv("LASTFM_API_KEY"),
    )


def require_genius(creds: ApiCredentials) -> str:
    if not creds.genius_token:
        raise ValueError(
            "GENIUS_ACCESS_TOKEN is required. "
            "Create a client at https://genius.com/api-clients and add it to .env"
        )
    return creds.genius_token


def require_spotify(creds: ApiCredentials) -> tuple[str, str]:
    if not creds.spotify_client_id or not creds.spotify_client_secret:
        raise ValueError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are required. "
            "Create an app at https://developer.spotify.com/dashboard and add them to .env"
        )
    return creds.spotify_client_id, creds.spotify_client_secret

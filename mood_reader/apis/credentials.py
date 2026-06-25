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
    kaggle_api_token: str | None
    kaggle_username: str | None
    kaggle_key: str | None


def get_credentials() -> ApiCredentials:
    return ApiCredentials(
        genius_token=os.getenv("GENIUS_ACCESS_TOKEN"),
        spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        lastfm_api_key=os.getenv("LASTFM_API_KEY"),
        kaggle_api_token=os.getenv("KAGGLE_API_TOKEN"),
        kaggle_username=os.getenv("KAGGLE_USERNAME"),
        kaggle_key=os.getenv("KAGGLE_KEY"),
    )


def configure_kaggle_env() -> None:
    """Expose Kaggle credentials to kagglehub / Kaggle API clients."""
    creds = get_credentials()
    if creds.kaggle_api_token:
        os.environ["KAGGLE_API_TOKEN"] = creds.kaggle_api_token
    if creds.kaggle_username:
        os.environ["KAGGLE_USERNAME"] = creds.kaggle_username
    if creds.kaggle_key:
        os.environ["KAGGLE_KEY"] = creds.kaggle_key


def require_kaggle(creds: ApiCredentials) -> None:
    if creds.kaggle_api_token:
        return
    if creds.kaggle_username and creds.kaggle_key:
        return
    raise ValueError(
        "KAGGLE_API_TOKEN is required in .env. "
        "Generate one at https://www.kaggle.com/settings → API → Generate New Token. "
        "(Legacy option: KAGGLE_USERNAME + KAGGLE_KEY from kaggle.json)"
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

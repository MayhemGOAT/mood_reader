"""Discover diverse tracks on Spotify and enrich with Genius lyrics."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from mood_reader.apis.genius import fetch_genius_lyrics
from mood_reader.apis.lastfm import enrich_with_lastfm
from mood_reader.apis.song_lookup import download_preview
from mood_reader.apis.spotify import (
    enrich_from_track,
    get_track,
    search_tracks,
    track_artist_name,
)
from mood_reader.config import load_config
from mood_reader.fetch import save_dataset

# Markets spanning multiple languages / regions
DISCOVERY_MARKETS = [
    "US", "GB", "CA", "AU", "MX", "BR", "AR", "ES", "FR", "DE", "IT", "PT",
    "NL", "SE", "PL", "TR", "JP", "KR", "TW", "IN", "ID", "PH", "TH", "ZA",
]

GENRE_QUERIES = [
    "genre:pop", "genre:rock", "genre:hip-hop", "genre:r-n-b", "genre:soul",
    "genre:latin", "genre:reggaeton", "genre:k-pop", "genre:indie", "genre:folk",
    "genre:country", "genre:metal", "genre:punk", "genre:blues", "genre:dance",
    "genre:electronic", "genre:afrobeat", "genre:reggae", "genre:bollywood",
    "genre:spanish", "genre:french", "genre:german", "genre:jazz", "genre:classical",
]

SEARCH_QUERIES = [
    "a", "e", "i", "o", "u", "love", "life", "heart", "night", "day",
    "amor", "corazon", "vie", "herz", "vida", "noite", "amour", "luna",
    "star", "dream", "world", "time", "home", "light", "fire", "rain",
]


@dataclass(frozen=True)
class TrackCandidate:
    title: str
    artist: str
    spotify_id: str
    market: str
    genre_hint: str = ""


def _to_candidate(track: dict, market: str, genre_hint: str = "") -> TrackCandidate | None:
    track_id = track.get("id")
    title = track.get("name")
    if not track_id or not title:
        return None
    return TrackCandidate(
        title=title,
        artist=track_artist_name(track),
        spotify_id=track_id,
        market=market,
        genre_hint=genre_hint,
    )


def discover_candidates(
    *,
    pool_size: int = 3000,
    seed: int | None = None,
) -> list[TrackCandidate]:
    """Build a shuffled pool via international artist search on Spotify."""
    from mood_reader.discovery_artists import INTERNATIONAL_ARTISTS

    rng = random.Random(seed)
    seen_ids: set[str] = set()
    candidates: list[TrackCandidate] = []

    def add_tracks(tracks: list[dict], market: str, genre_hint: str = "") -> int:
        added = 0
        for track in tracks:
            track_id = track.get("id")
            if not track_id or track_id in seen_ids:
                continue
            candidate = _to_candidate(track, market, genre_hint)
            if candidate is None:
                continue
            seen_ids.add(track_id)
            candidates.append(candidate)
            added += 1
        return added

    artists = INTERNATIONAL_ARTISTS.copy()
    rng.shuffle(artists)

    with tqdm(total=pool_size, desc="Discovering on Spotify") as bar:
        for artist, market in artists:
            if len(candidates) >= pool_size:
                break
            try:
                tracks = search_tracks(
                    f"artist:{artist}",
                    market=market,
                    limit=10,
                )
                added = add_tracks(tracks, market)
                bar.update(added)
            except requests.RequestException:
                time.sleep(2)
            time.sleep(0.6)

        # Top up with genre search if the artist list is too small
        attempts = 0
        while len(candidates) < pool_size and attempts < 100:
            attempts += 1
            market = rng.choice(DISCOVERY_MARKETS)
            year = rng.randint(2005, 2025)
            base = rng.choice(GENRE_QUERIES)
            query = f"{base} year:{year}-{year}"
            genre_hint = base.replace("genre:", "")
            offset = rng.randrange(0, 200, 10)
            try:
                tracks = search_tracks(query, market=market, limit=10, offset=offset)
                added = add_tracks(tracks, market, genre_hint)
                bar.update(added)
            except requests.RequestException:
                time.sleep(2)
            time.sleep(0.6)

    rng.shuffle(candidates)
    return candidates[:pool_size]


def _has_usable_lyrics(lyrics: str | None) -> bool:
    if not lyrics:
        return False
    text = lyrics.strip().lower()
    if len(text) < 40:
        return False
    if text in {"instrumental", "[instrumental]", "(instrumental)"}:
        return False
    return True


def lookup_candidate(
    candidate: TrackCandidate,
    track: dict,
    *,
    include_lastfm: bool = True,
) -> dict:
    """Fetch Genius lyrics and merge with Spotify metadata."""
    from mood_reader.labels import (
        valence_energy_from_genre,
        valence_energy_from_market,
        valence_energy_from_tags,
        vibe_from_valence_energy,
    )

    info = fetch_genius_lyrics(candidate.title, candidate.artist)
    enrich_from_track(info, track)

    if include_lastfm:
        info = enrich_with_lastfm(info)

    row = info.to_dataset_row()
    row["market"] = candidate.market
    row["spotify_id"] = candidate.spotify_id
    row["has_lyrics"] = _has_usable_lyrics(info.lyrics)
    row["fetch_errors"] = "; ".join(info.errors)
    row["sources"] = "|".join(info.sources)

    label_source = ""
    if row.get("valence") is not None and row.get("energy") is not None:
        label_source = "spotify"
    elif info.lastfm_tags:
        inferred = valence_energy_from_tags(info.lastfm_tags)
        if inferred:
            row["valence"], row["energy"] = inferred
            row["vibe"] = vibe_from_valence_energy(row["valence"], row["energy"])
            label_source = "lastfm"
    if label_source == "" and candidate.genre_hint:
        inferred = valence_energy_from_genre(candidate.genre_hint)
        if inferred:
            row["valence"], row["energy"] = inferred
            row["vibe"] = vibe_from_valence_energy(row["valence"], row["energy"])
            label_source = "genre_estimate"
    if label_source == "":
        inferred = valence_energy_from_market(candidate.market)
        if inferred:
            row["valence"], row["energy"] = inferred
            row["vibe"] = vibe_from_valence_energy(row["valence"], row["energy"])
            label_source = "market_estimate"

    row["label_source"] = label_source
    return row


def discover_and_fetch(
    count: int = 500,
    *,
    output_path: str | Path = "data/discovered_songs.csv",
    pool_size: int | None = None,
    max_per_artist: int = 2,
    require_lyrics: bool = True,
    download_previews: bool = False,
    seed: int | None = None,
    save_every: int = 25,
) -> Path:
    cfg = load_config()
    output_path = Path(output_path)
    preview_dir = Path(cfg["paths"]["data_dir"]) / "previews"

    if pool_size is None:
        pool_size = max(count * 5, 2500)

    candidates = discover_candidates(pool_size=pool_size, seed=seed)

    rows: list[dict] = []
    artist_counts: dict[str, int] = {}
    skipped = 0
    scanned = 0

    with tqdm(total=count, desc="Fetching Genius + Spotify") as bar:
        for candidate in candidates:
            if len(rows) >= count:
                break

            scanned += 1
            artist_key = candidate.artist.lower()
            if artist_counts.get(artist_key, 0) >= max_per_artist:
                skipped += 1
                continue

            try:
                track = get_track(candidate.spotify_id, market=candidate.market)
            except requests.RequestException:
                skipped += 1
                time.sleep(0.5)
                continue
            if track is None:
                skipped += 1
                continue

            row = lookup_candidate(candidate, track)

            if require_lyrics and not row["has_lyrics"]:
                skipped += 1
                continue

            if row.get("valence") is None or row.get("energy") is None:
                skipped += 1
                continue

            if download_previews and row.get("preview_url"):
                from mood_reader.apis.base import SongInfo

                preview_info = SongInfo(
                    title=row["title"],
                    artist=row["artist"],
                    preview_url=row["preview_url"],
                )
                preview = download_preview(preview_info, preview_dir)
                if preview:
                    row["audio_path"] = str(preview)

            rows.append(row)
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
            bar.update(1)

            if len(rows) % save_every == 0:
                save_dataset(rows, output_path, append=False)

            time.sleep(0.15)

    save_dataset(rows, output_path, append=False)

    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "saved": str(output_path),
        "tracks_saved": len(rows),
        "target": count,
        "candidates_scanned": scanned,
        "pool_size": len(candidates),
        "skipped": skipped,
        "unique_artists": len(artist_counts),
        "require_lyrics": require_lyrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return output_path

"""Filter and re-enrich song datasets."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mood_reader.apis.base import SongInfo
from mood_reader.apis.song_lookup import download_preview
from mood_reader.apis.spotify import enrich_with_spotify
from mood_reader.config import load_config
from mood_reader.fetch import save_dataset
from mood_reader.labels import vibe_from_valence_energy

MIN_LYRICS_LEN = 40
INSTRUMENTAL_MARKERS = {"instrumental", "[instrumental]", "(instrumental)"}


def has_usable_lyrics(lyrics: object) -> bool:
    if lyrics is None or (isinstance(lyrics, float) and pd.isna(lyrics)):
        return False
    text = str(lyrics).strip()
    if len(text) < MIN_LYRICS_LEN:
        return False
    if text.lower() in INSTRUMENTAL_MARKERS:
        return False
    return True


def has_music_data(row: pd.Series, *, require_audio_file: bool = False) -> bool:
    spotify_id = str(row.get("spotify_id") or "").strip()
    if not spotify_id or spotify_id.lower() == "nan":
        return False

    valence = row.get("valence")
    energy = row.get("energy")
    has_spotify_labels = pd.notna(valence) and pd.notna(energy)

    preview_url = str(row.get("preview_url") or "").strip()
    has_preview = bool(preview_url) and preview_url.lower() != "nan"

    audio_path = str(row.get("audio_path") or "").strip()
    has_audio = bool(audio_path) and audio_path.lower() != "nan"
    if has_audio:
        has_audio = Path(audio_path).exists()

    if require_audio_file:
        return has_audio

    return has_spotify_labels or has_preview or has_audio


def filter_complete_rows(
    df: pd.DataFrame,
    *,
    require_audio_file: bool = False,
) -> pd.DataFrame:
    if "artist" not in df.columns or "title" not in df.columns:
        raise ValueError("Dataset must include 'artist' and 'title' columns")

    mask = df.apply(
        lambda row: has_usable_lyrics(row.get("lyrics")) and has_music_data(
            row, require_audio_file=require_audio_file
        ),
        axis=1,
    )
    return df.loc[mask].copy()


def _merge_spotify_into_row(row: pd.Series, info: SongInfo, audio_path: str | None) -> pd.Series:
    updated = row.copy()
    updated["title"] = info.title or row.get("title")
    updated["artist"] = info.artist or row.get("artist")
    updated["spotify_id"] = info.spotify_id or ""
    updated["preview_url"] = info.preview_url or ""
    updated["album"] = info.album or row.get("album", "")

    for col in (
        "valence", "energy", "tempo", "danceability", "acousticness",
        "instrumentalness", "speechiness", "liveness",
    ):
        value = getattr(info, col, None)
        updated[col] = value

    if info.valence is not None and info.energy is not None:
        updated["vibe"] = vibe_from_valence_energy(float(info.valence), float(info.energy))
    elif not str(updated.get("vibe") or "").strip():
        updated["vibe"] = ""

    if audio_path:
        updated["audio_path"] = str(audio_path)

    sources = str(row.get("sources") or "")
    source_parts = [s for s in sources.split("|") if s]
    for src in info.sources:
        if src not in source_parts:
            source_parts.append(src)
    updated["sources"] = "|".join(source_parts)

    errors = str(row.get("fetch_errors") or "")
    if info.errors:
        updated["fetch_errors"] = "; ".join(info.errors)
    elif info.spotify_id:
        updated["fetch_errors"] = errors.replace("Spotify: 429 Client Error", "").strip(" ;")
    else:
        updated["fetch_errors"] = errors

    return updated


def enrich_spotify_rows(
    df: pd.DataFrame,
    *,
    download_previews: bool = False,
    delay_seconds: float = 2.0,
    save_every: int = 25,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Backfill Spotify metadata for rows that already have lyrics."""
    cfg = load_config()
    preview_dir = Path(cfg["paths"]["data_dir"]) / "previews"
    rows = df.to_dict(orient="records")
    enriched = 0
    processed = 0

    for i, row in enumerate(tqdm(rows, desc="Re-enriching Spotify")):
        series = pd.Series(row)
        if has_music_data(series) and has_usable_lyrics(series.get("lyrics")):
            continue
        if not has_usable_lyrics(series.get("lyrics")):
            continue

        title = str(row.get("title") or "").strip()
        artist = str(row.get("artist") or "").strip()
        if not title or not artist or artist.lower() == "nan" or title.lower() == "nan":
            continue

        info = SongInfo(title=title, artist=artist, lyrics=row.get("lyrics"))
        info = enrich_with_spotify(info)

        audio_path = None
        if download_previews and info.preview_url:
            audio_path = download_preview(info, preview_dir)

        rows[i] = _merge_spotify_into_row(series, info, audio_path).to_dict()
        if info.spotify_id:
            enriched += 1

        processed += 1
        if output_path is not None and processed % save_every == 0:
            save_dataset(rows, output_path, append=False)

        time.sleep(delay_seconds)

    result = pd.DataFrame(rows)
    if output_path is not None:
        save_dataset(rows, output_path, append=False)

    return result


def clean_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    do_re_enrich: bool = False,
    download_previews: bool = False,
    require_audio_file: bool = False,
    delay_seconds: float = 2.0,
    save_every: int = 25,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    lyrics_ok = df.apply(lambda r: has_usable_lyrics(r.get("lyrics")), axis=1).sum()
    music_ok = df.apply(lambda r: has_music_data(r, require_audio_file=require_audio_file), axis=1).sum()

    working = df
    if do_re_enrich:
        working_path = output_path.with_suffix(".working.csv")
        working = enrich_spotify_rows(
            df,
            download_previews=download_previews,
            delay_seconds=delay_seconds,
            save_every=save_every,
            output_path=working_path,
        )

    cleaned = filter_complete_rows(working, require_audio_file=require_audio_file)
    cleaned.to_csv(output_path, index=False)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": len(df),
        "rows_with_lyrics": int(lyrics_ok),
        "rows_with_music_data_before_clean": int(music_ok),
        "output_rows": len(cleaned),
        "re_enriched_spotify": do_re_enrich,
        "require_audio_file": require_audio_file,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def enrich_lyrics_with_spotify(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    download_previews: bool = False,
    delay_seconds: float = 2.0,
    save_every: int = 25,
) -> dict:
    """For every row with lyrics, look up Spotify track metadata (+ optional preview)."""
    input_path = Path(input_path)
    output_path = Path(output_path or input_path)

    df = pd.read_csv(input_path)
    lyrics_count = int(df.apply(lambda r: has_usable_lyrics(r.get("lyrics")), axis=1).sum())
    already = int(df.apply(lambda r: has_music_data(r), axis=1).sum())

    enriched_df = enrich_spotify_rows(
        df,
        download_previews=download_previews,
        delay_seconds=delay_seconds,
        save_every=save_every,
        output_path=output_path,
    )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "rows_total": len(df),
        "rows_with_lyrics": lyrics_count,
        "spotify_before": already,
        "spotify_after": int(enriched_df.apply(lambda r: has_music_data(r), axis=1).sum()),
    }
    summary_path = output_path.with_suffix(".enrich_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

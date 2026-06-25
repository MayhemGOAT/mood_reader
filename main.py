#!/usr/bin/env python3
"""CLI entry point for Mood Reader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify song mood from lyrics + audio")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train models from a labeled CSV dataset")
    train_p.add_argument("--data", required=True, help="Path to songs.csv")
    train_p.add_argument("--model-dir", default=None, help="Output directory for models")

    fetch_p = sub.add_parser("fetch", help="Fetch lyrics + song info from APIs into a CSV")
    fetch_p.add_argument("--artist", action="append", default=[], help="Artist name (repeatable)")
    fetch_p.add_argument("--title", action="append", default=[], help="Title (repeatable, pairs with --artist)")
    fetch_p.add_argument("--tracks", default=None, help="Text file: 'Artist - Title' per line")
    fetch_p.add_argument("--output", default="data/fetched_songs.csv", help="Output CSV path")
    fetch_p.add_argument("--append", action="store_true", help="Append to existing CSV (same as --resume)")
    fetch_p.add_argument("--no-resume", action="store_true", help="Re-fetch all tracks; overwrite existing CSV entries")
    fetch_p.add_argument("--save-every", type=int, default=10, help="Checkpoint CSV every N songs (default: 10)")
    fetch_p.add_argument("--download-previews", action="store_true", help="Save Spotify 30s previews for audio features")

    disc_p = sub.add_parser("discover", help="Discover diverse tracks on Spotify and fetch Genius lyrics")
    disc_p.add_argument("--count", type=int, default=500, help="Number of songs to save")
    disc_p.add_argument("--output", default="data/discovered_songs.csv", help="Output CSV path")
    disc_p.add_argument("--pool-size", type=int, default=None, help="Spotify candidate pool size")
    disc_p.add_argument("--max-per-artist", type=int, default=2, help="Max tracks per artist")
    disc_p.add_argument("--include-instrumental", action="store_true", help="Keep tracks without Genius lyrics")
    disc_p.add_argument("--download-previews", action="store_true", help="Save Spotify 30s previews")
    disc_p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    clean_p = sub.add_parser("clean", help="Filter dataset to rows with lyrics + Spotify music data")
    clean_p.add_argument("--input", default="data/discovered_songs.csv", help="Input CSV path")
    clean_p.add_argument("--output", default="data/discovered_songs_clean.csv", help="Output CSV path")
    clean_p.add_argument("--re-enrich-spotify", action="store_true", help="Backfill missing Spotify metadata before filtering")
    clean_p.add_argument("--download-previews", action="store_true", help="Download Spotify preview MP3s during re-enrich")
    clean_p.add_argument("--require-audio-file", action="store_true", help="Require a local preview/audio file on disk")
    clean_p.add_argument("--delay", type=float, default=2.0, help="Seconds between Spotify API calls during re-enrich")
    clean_p.add_argument("--save-every", type=int, default=25, help="Checkpoint working CSV every N enriched rows")

    enrich_p = sub.add_parser("enrich", help="Add Spotify music data to rows that already have lyrics")
    enrich_p.add_argument("--input", default="data/discovered_songs.csv", help="Input CSV path")
    enrich_p.add_argument("--output", default=None, help="Output CSV (default: overwrite --input)")
    enrich_p.add_argument("--download-previews", action="store_true", help="Download Spotify 30s preview MP3s")
    enrich_p.add_argument("--delay", type=float, default=2.0, help="Seconds between Spotify API calls")
    enrich_p.add_argument("--save-every", type=int, default=25, help="Checkpoint every N lyric rows processed")

    list_p = sub.add_parser("list-songs", help="Export songs with lyrics to a track list file")
    list_p.add_argument("--input", default="data/discovered_songs.csv", help="Input CSV path")
    list_p.add_argument("--output", default="data/songs_with_lyrics.txt", help="Output track list path")

    spotify_p = sub.add_parser("fetch-spotify", help="Fetch Spotify music data for songs in CSV (lyrics unchanged)")
    spotify_p.add_argument("--data", default="data/discovered_songs.csv", help="CSV with lyrics to update")
    spotify_p.add_argument("--tracks", default=None, help="Track list file (default: all lyrical rows in --data)")
    spotify_p.add_argument("--download-previews", action="store_true", help="Download Spotify 30s preview MP3s")
    spotify_p.add_argument("--delay", type=float, default=2.0, help="Seconds between Spotify API calls")
    spotify_p.add_argument("--save-every", type=int, default=25, help="Checkpoint CSV every N songs processed")
    spotify_p.add_argument("--no-skip-existing", action="store_true", help="Re-fetch Spotify even if already present")

    merge_p = sub.add_parser("merge-kaggle", help="Merge lyrics CSV with Kaggle Spotify audio features")
    merge_p.add_argument("--input", default="data/discovered_songs.csv", help="CSV with Genius lyrics")
    merge_p.add_argument("--output", default="data/discovered_songs_clean.csv", help="Merged output CSV")
    merge_p.add_argument(
        "--dataset",
        default="joebeachcapital/30000-spotify-songs",
        help="Kaggle dataset slug (owner/name)",
    )
    merge_p.add_argument("--kaggle-csv", default=None, help="Local Kaggle CSV path (skip download)")
    merge_p.add_argument("--cache-dir", default="data/kaggle", help="Download/cache directory")
    merge_p.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Keep rows without Kaggle match (default: lyrics + features only)",
    )

    pred_p = sub.add_parser("predict", help="Predict mood for one song")
    pred_p.add_argument("--lyrics", default=None, help="Lyrics text")
    pred_p.add_argument("--lyrics-file", default=None, help="Path to lyrics .txt file")
    pred_p.add_argument("--artist", default=None, help="Artist (fetches lyrics + metadata from APIs)")
    pred_p.add_argument("--title", default=None, help="Title (use with --artist)")
    pred_p.add_argument("--audio", default=None, help="Path to audio file (.mp3, .wav, etc.)")
    pred_p.add_argument("--use-preview", action="store_true", help="Use Spotify preview for audio analysis")
    pred_p.add_argument("--model-dir", default=None, help="Directory with trained models")

    args = parser.parse_args()

    if args.command == "train":
        from mood_reader.train import train

        report = train(args.data, args.model_dir)
        print(json.dumps(report, indent=2))
        return

    if args.command == "fetch":
        from mood_reader.fetch import fetch_tracks_to_csv, load_track_list

        tracks: list[tuple[str, str]] = []
        if args.tracks:
            tracks.extend(load_track_list(args.tracks))

        if args.artist:
            if len(args.artist) != len(args.title):
                print("--artist and --title must be provided the same number of times", file=sys.stderr)
                sys.exit(1)
            tracks.extend(zip(args.artist, args.title))

        if not tracks:
            print("Provide --tracks file and/or --artist + --title pairs", file=sys.stderr)
            sys.exit(1)

        summary = fetch_tracks_to_csv(
            tracks,
            args.output,
            append=args.append,
            download_previews=args.download_previews,
            resume=not args.no_resume,
            save_every=args.save_every,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "discover":
        from mood_reader.discover import discover_and_fetch

        out = discover_and_fetch(
            args.count,
            output_path=args.output,
            pool_size=args.pool_size,
            max_per_artist=args.max_per_artist,
            require_lyrics=not args.include_instrumental,
            download_previews=args.download_previews,
            seed=args.seed,
        )
        summary_path = out.with_suffix(".summary.json")
        print(json.dumps({"saved": str(out), "summary": json.loads(summary_path.read_text())}, indent=2))
        return

    if args.command == "list-songs":
        from mood_reader.spotify_fetch import export_songs_with_lyrics

        summary = export_songs_with_lyrics(args.input, args.output)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "fetch-spotify":
        from mood_reader.spotify_fetch import fetch_spotify_for_csv

        summary = fetch_spotify_for_csv(
            args.data,
            tracks_file=args.tracks,
            download_previews=args.download_previews,
            delay_seconds=args.delay,
            save_every=args.save_every,
            skip_existing=not args.no_skip_existing,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "merge-kaggle":
        from mood_reader.kaggle_merge import merge_kaggle_dataset

        summary = merge_kaggle_dataset(
            args.input,
            args.output,
            dataset=args.dataset,
            kaggle_csv=args.kaggle_csv,
            cache_dir=args.cache_dir,
            clean_only=not args.keep_unmatched,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "enrich":
        from mood_reader.clean import enrich_lyrics_with_spotify

        summary = enrich_lyrics_with_spotify(
            args.input,
            args.output,
            download_previews=args.download_previews,
            delay_seconds=args.delay,
            save_every=args.save_every,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "clean":
        from mood_reader.clean import clean_dataset

        summary = clean_dataset(
            args.input,
            args.output,
            do_re_enrich=args.re_enrich_spotify,
            download_previews=args.download_previews,
            require_audio_file=args.require_audio_file,
            delay_seconds=args.delay,
            save_every=args.save_every,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "predict":
        from mood_reader.predict import predict_song

        lyrics = None
        audio_path = args.audio

        if args.artist and args.title:
            from mood_reader.apis.song_lookup import lookup_with_preview

            info, preview_path = lookup_with_preview(args.title, args.artist)
            if not info.lyrics:
                print(json.dumps({"error": "Could not fetch lyrics", "details": info.to_dict()}, indent=2))
                sys.exit(1)

            lyrics = info.lyrics
            if args.use_preview and preview_path:
                audio_path = str(preview_path)

            result = predict_song(lyrics=lyrics, audio_path=audio_path, model_dir=args.model_dir)
            result["song"] = {
                "title": info.title,
                "artist": info.artist,
                "sources": info.sources,
                "spotify_valence": info.valence,
                "spotify_energy": info.energy,
                "lastfm_tags": info.lastfm_tags,
            }
            print(json.dumps(result, indent=2))
            return

        if args.lyrics_file:
            lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")
        elif args.lyrics:
            lyrics = args.lyrics
        else:
            print("Provide --lyrics, --lyrics-file, or --artist + --title", file=sys.stderr)
            sys.exit(1)

        result = predict_song(lyrics=lyrics, audio_path=audio_path, model_dir=args.model_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

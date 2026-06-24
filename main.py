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
    fetch_p.add_argument("--append", action="store_true", help="Append to existing CSV")
    fetch_p.add_argument("--download-previews", action="store_true", help="Save Spotify 30s previews for audio features")

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

        out = fetch_tracks_to_csv(
            tracks,
            args.output,
            append=args.append,
            download_previews=args.download_previews,
        )
        print(json.dumps({"saved": str(out), "tracks": len(tracks)}, indent=2))
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

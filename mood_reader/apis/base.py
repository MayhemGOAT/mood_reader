"""Shared types for external song metadata APIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SongInfo:
    title: str
    artist: str
    lyrics: str | None = None
    genius_url: str | None = None
    genius_id: int | None = None
    spotify_id: str | None = None
    album: str | None = None
    preview_url: str | None = None
    valence: float | None = None
    energy: float | None = None
    tempo: float | None = None
    danceability: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    speechiness: float | None = None
    liveness: float | None = None
    key: int | None = None
    mode: int | None = None
    lastfm_tags: list[str] = field(default_factory=list)
    lastfm_url: str | None = None
    sources: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dataset_row(self) -> dict:
        from mood_reader.labels import vibe_from_valence_energy

        row = {
            "title": self.title,
            "artist": self.artist,
            "lyrics": self.lyrics or "",
            "audio_path": "",
            "spotify_id": self.spotify_id or "",
            "genius_url": self.genius_url or "",
            "preview_url": self.preview_url or "",
            "album": self.album or "",
            "valence": self.valence,
            "energy": self.energy,
            "tempo": self.tempo,
            "danceability": self.danceability,
            "acousticness": self.acousticness,
            "instrumentalness": self.instrumentalness,
            "speechiness": self.speechiness,
            "liveness": self.liveness,
            "lastfm_tags": "|".join(self.lastfm_tags),
        }
        if self.valence is not None and self.energy is not None:
            row["vibe"] = vibe_from_valence_energy(self.valence, self.energy)
        else:
            row["vibe"] = ""
        return row

    def to_dict(self) -> dict:
        return asdict(self)

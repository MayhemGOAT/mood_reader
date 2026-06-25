"""Mood label definitions and helpers."""

from __future__ import annotations

VIBES = [
    "chill",
    "happy",
    "energetic",
    "melancholic",
    "aggressive",
    "romantic",
    "uplifting",
    "dark",
]

# Approximate valence/energy anchors for each vibe (used for weak supervision)
VIBE_ANCHORS: dict[str, tuple[float, float]] = {
    "chill": (0.55, 0.25),
    "happy": (0.85, 0.60),
    "energetic": (0.70, 0.90),
    "melancholic": (0.25, 0.30),
    "aggressive": (0.20, 0.85),
    "romantic": (0.75, 0.35),
    "uplifting": (0.80, 0.70),
    "dark": (0.15, 0.55),
}


def energy_bucket(energy: float) -> str:
    if energy < 0.35:
        return "low"
    if energy < 0.65:
        return "medium"
    return "high"


def valence_bucket(valence: float) -> str:
    if valence < 0.35:
        return "negative"
    if valence < 0.65:
        return "neutral"
    return "positive"


def vibe_from_valence_energy(valence: float, energy: float) -> str:
    """Map continuous valence/energy to nearest vibe anchor."""
    best_vibe = VIBES[0]
    best_dist = float("inf")
    for vibe, (v, e) in VIBE_ANCHORS.items():
        dist = (valence - v) ** 2 + (energy - e) ** 2
        if dist < best_dist:
            best_dist = dist
            best_vibe = vibe
    return best_vibe


# Last.fm mood tags → approximate (valence, energy) when Spotify audio-features unavailable
TAG_MOOD_ANCHORS: dict[str, tuple[float, float]] = {
    "happy": (0.85, 0.60),
    "cheerful": (0.82, 0.55),
    "uplifting": (0.80, 0.70),
    "party": (0.75, 0.85),
    "dance": (0.72, 0.88),
    "energetic": (0.70, 0.90),
    "chill": (0.55, 0.25),
    "relaxing": (0.58, 0.22),
    "mellow": (0.52, 0.28),
    "romantic": (0.75, 0.35),
    "love": (0.78, 0.40),
    "sad": (0.22, 0.28),
    "melancholy": (0.25, 0.30),
    "melancholic": (0.25, 0.30),
    "depressive": (0.18, 0.25),
    "dark": (0.15, 0.55),
    "aggressive": (0.20, 0.85),
    "angry": (0.18, 0.82),
    "heavy": (0.25, 0.75),
}


def valence_energy_from_tags(tags: list[str]) -> tuple[float, float] | None:
    """Infer valence/energy from crowd tags (e.g. Last.fm)."""
    if not tags:
        return None

    matches: list[tuple[float, float]] = []
    for tag in tags:
        key = tag.strip().lower()
        if key in TAG_MOOD_ANCHORS:
            matches.append(TAG_MOOD_ANCHORS[key])

    if not matches:
        return None

    valence = sum(v for v, _ in matches) / len(matches)
    energy = sum(e for _, e in matches) / len(matches)
    return valence, energy


# Fallback when Spotify audio-features API is unavailable (403 on new apps)
GENRE_MOOD_ESTIMATES: dict[str, tuple[float, float]] = {
    "pop": (0.65, 0.55),
    "rock": (0.55, 0.65),
    "hip-hop": (0.50, 0.70),
    "r-n-b": (0.60, 0.45),
    "soul": (0.62, 0.40),
    "latin": (0.72, 0.75),
    "reggaeton": (0.70, 0.80),
    "k-pop": (0.68, 0.72),
    "indie": (0.58, 0.50),
    "folk": (0.55, 0.35),
    "country": (0.60, 0.45),
    "metal": (0.30, 0.85),
    "punk": (0.35, 0.80),
    "blues": (0.40, 0.35),
    "dance": (0.72, 0.88),
    "electronic": (0.55, 0.75),
    "afrobeat": (0.70, 0.78),
    "reggae": (0.65, 0.55),
    "bollywood": (0.70, 0.65),
    "spanish": (0.68, 0.60),
    "french": (0.62, 0.50),
    "german": (0.55, 0.55),
    "jazz": (0.55, 0.40),
    "classical": (0.50, 0.30),
}


def valence_energy_from_genre(genre: str) -> tuple[float, float] | None:
    key = genre.strip().lower().replace("genre:", "")
    return GENRE_MOOD_ESTIMATES.get(key)


MARKET_GENRE_HINT: dict[str, str] = {
    "KR": "k-pop",
    "JP": "pop",
    "ES": "spanish",
    "MX": "latin",
    "BR": "latin",
    "AR": "latin",
    "CO": "latin",
    "FR": "french",
    "DE": "german",
    "IN": "bollywood",
    "NG": "afrobeat",
    "ZA": "afrobeat",
    "JM": "reggae",
    "IT": "pop",
    "TR": "pop",
    "TH": "pop",
    "ID": "pop",
    "PH": "pop",
}


def valence_energy_from_market(market: str) -> tuple[float, float] | None:
    genre = MARKET_GENRE_HINT.get(market.upper())
    if genre:
        return valence_energy_from_genre(genre)
    return valence_energy_from_genre("pop")

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

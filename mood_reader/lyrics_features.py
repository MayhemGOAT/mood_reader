"""Extract mood-relevant features from song lyrics."""

from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from mood_reader.config import load_config

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

_vader = SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _get_emotion_pipeline():
    if os.getenv("MOOD_READER_FAST") == "1":
        return None
    from transformers import pipeline

    cfg = load_config()["lyrics"]
    return pipeline(
        "text-classification",
        model=cfg["emotion_model"],
        top_k=None,
        device=-1,
    )


def _clean_lyrics(text: str) -> str:
    text = re.sub(r"\[.*?\]", " ", text)  # strip [Verse], [Chorus], etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_scores(text: str) -> dict[str, float]:
    lower = text.lower()
    keywords = {
        "love_romance": ["love", "heart", "kiss", "baby", "darling", "romance"],
        "party_energy": ["dance", "party", "night", "club", "fire", "wild"],
        "sadness": ["cry", "tears", "lonely", "pain", "hurt", "goodbye", "miss"],
        "anger": ["fight", "rage", "hate", "burn", "war", "kill"],
        "hope": ["rise", "shine", "dream", "hope", "free", "fly", "light"],
        "darkness": ["dark", "shadow", "death", "blood", "devil", "hell"],
    }
    words = re.findall(r"[a-z']+", lower)
    total = max(len(words), 1)
    scores = {}
    for theme, terms in keywords.items():
        hits = sum(1 for w in words if w in terms)
        scores[f"kw_{theme}"] = hits / total
    return scores


def _emotion_scores(text: str) -> dict[str, float]:
    pipe = _get_emotion_pipeline()
    if pipe is None:
        return _emotion_scores_fast(text)

    cfg = load_config()["lyrics"]
    max_tokens = cfg["max_length"]
    tokenizer = pipe.tokenizer
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return {f"emo_{k}": 0.0 for k in EMOTION_LABELS}

    stride = max(max_tokens - 2, 1)
    chunks: list[str] = []
    for start in range(0, len(token_ids), stride):
        piece = token_ids[start : start + stride]
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))
        if start + stride >= len(token_ids):
            break

    accum = {label: 0.0 for label in EMOTION_LABELS}
    for chunk in chunks:
        if not chunk.strip():
            continue
        results = pipe(chunk, truncation=True, max_length=max_tokens)[0]
        for item in results:
            label = item["label"].lower()
            if label in accum:
                accum[label] += item["score"]
    n = max(len(chunks), 1)
    return {f"emo_{k}": v / n for k, v in accum.items()}


def _emotion_scores_fast(text: str) -> dict[str, float]:
    """Rule-based fallback when transformers are disabled."""
    lower = text.lower()
    patterns = {
        "anger": r"\b(angry|rage|hate|fight|mad)\b",
        "disgust": r"\b(disgust|gross|sick)\b",
        "fear": r"\b(afraid|scared|fear|terror)\b",
        "joy": r"\b(joy|happy|smile|laugh|fun)\b",
        "sadness": r"\b(sad|cry|tears|lonely|blue|hurt)\b",
        "surprise": r"\b(wow|shock|surprise)\b",
    }
    scores = {}
    for label, pattern in patterns.items():
        scores[f"emo_{label}"] = min(len(re.findall(pattern, lower)) / 10.0, 1.0)
    scores["emo_neutral"] = max(0.0, 1.0 - sum(scores.values()))
    return scores


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chars])
        start += max_chars
    return chunks


def extract_lyrics_features(lyrics: str) -> dict[str, float]:
    text = _clean_lyrics(lyrics)
    if not text:
        raise ValueError("Lyrics text is empty")

    vader = _vader.polarity_scores(text)
    words = re.findall(r"[a-z']+", text.lower())

    features: dict[str, float] = {
        "lyrics_word_count": float(len(words)),
        "lyrics_unique_word_ratio": len(set(words)) / max(len(words), 1),
        "lyrics_avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "lyrics_exclamation_ratio": text.count("!") / max(len(text), 1),
        "lyrics_question_ratio": text.count("?") / max(len(text), 1),
        "vader_compound": vader["compound"],
        "vader_pos": vader["pos"],
        "vader_neg": vader["neg"],
        "vader_neu": vader["neu"],
    }
    features.update(_keyword_scores(text))
    features.update(_emotion_scores(text))
    return features


def lyrics_feature_names() -> list[str]:
    base = [
        "lyrics_word_count",
        "lyrics_unique_word_ratio",
        "lyrics_avg_word_len",
        "lyrics_exclamation_ratio",
        "lyrics_question_ratio",
        "vader_compound",
        "vader_pos",
        "vader_neg",
        "vader_neu",
    ]
    kw = [f"kw_{k}" for k in ("love_romance", "party_energy", "sadness", "anger", "hope", "darkness")]
    emo = [f"emo_{k}" for k in EMOTION_LABELS]
    return base + kw + emo

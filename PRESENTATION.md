# Mood Reader — Project Presentation

## Slide 1 — Title
**Mood Reader**  
Classifying song mood from lyrics + audio

Presented by: *[Your Name]*  
Date: *[Today’s Date]*

---

## Slide 2 — What I Built
A multimodal ML system that predicts a song’s:
- **Valence** — negative ↔ positive
- **Energy** — calm ↔ intense
- **Vibe** — one of 8 labels (`chill`, `happy`, `energetic`, `melancholic`, `aggressive`, `romantic`, `uplifting`, `dark`)

Goal: mood prediction from lyrics + optional audio, without needing Spotify at inference time.

---

## Slide 3 — Why This Project
- Music discovery and playlists are often mood-driven
- Lyrics capture emotional meaning; audio captures rhythm and intensity
- Most simple systems use only one signal

I combined both modalities for stronger mood understanding.

---

## Slide 4 — How It Works
```text
Lyrics ──► emotion model + VADER + keywords ──┐
                                               ├──► Gradient Boosting ──► valence / energy / vibe
Audio  ──► librosa (tempo, MFCC, spectral) ──┘
```

Saved models:
- `valence_model.joblib` (regression)
- `energy_model.joblib` (regression)
- `vibe_model.joblib` (classification)

---

## Slide 5 — End-to-End Workflow
CLI pipeline from data to prediction:
- `fetch` / `discover` — collect lyrics and track metadata from APIs
- `merge-kaggle` / `enrich` — scale and clean training data
- `train` / `train-eval` — fit models and generate metrics/charts
- `predict` — score a track by artist/title or raw lyrics

---

## Slide 6 — Engineering Decisions
- **Fast mode** (`MOOD_READER_FAST=1`) skips the heavy HuggingFace emotion model for quick iteration
- **Spotify optional at inference** — works with lyrics-only inputs
- **Kaggle merge path** — grows datasets when Spotify API rate limits hit

These made experimentation practical at real scale.

---

## Slide 7 — What I Shipped (Timeline)
1. Initial multimodal mood classifier
2. Dataset pipelines (Kaggle merge, Spotify enrich, fetch)
3. Lyrics + Spotify training and evaluation pipeline
4. Emotion-model evaluation artifacts and charts
5. README polish (taxonomy, quickstart, CLI reference)

---

## Slide 8 — Example Output
```json
{
  "valence": 0.721,
  "energy": 0.814,
  "valence_label": "positive",
  "energy_label": "high",
  "vibe": "energetic"
}
```

Interpretation: positive, high-energy track — good fit for workout/party playlists.

---

## Slide 9 — Skills Demonstrated
- End-to-end ML workflow: data → features → train → evaluate → predict
- Multimodal fusion of NLP and audio DSP
- Practical tradeoffs: speed vs quality, API limits vs dataset scale
- Usable CLI and clear documentation

---

## Slide 10 — What’s Next
- Broader labeled data across genres and languages
- Confidence scores and better model calibration
- Small API or web UI for interactive demos
- Deeper multimodal fusion for higher accuracy

# Mood Reader — Project Presentation

## Slide 1 — Title
**Mood Reader**  
Classifying song mood from lyrics + audio  

Presented by: *[Your Name]*  
Date: *[Today’s Date]*  

---

## Slide 2 — What I Was Building
I built a **multimodal machine learning project** that predicts a song’s:
- **Valence** (negative ↔ positive)
- **Energy** (calm ↔ intense)
- **Vibe label** (8 classes like `chill`, `happy`, `energetic`, `dark`)

Goal: make mood prediction work with **lyrics + optional audio**, without requiring Spotify at inference time.

---

## Slide 3 — Why This Project
- Playlist curation and music discovery are often mood-driven.
- Lyrics capture emotional meaning; audio captures rhythm/energy.
- Most simple systems use only one signal.

I wanted to combine both modalities to get better mood understanding.

---

## Slide 4 — Core Approach
Two feature pipelines feed a shared model layer:

1. **Lyrics features**
   - Transformer-based emotion signal (DistilRoBERTa)
   - VADER sentiment
   - Theme/keyword indicators

2. **Audio features** (librosa)
   - Tempo and RMS energy
   - MFCCs
   - Spectral/chroma descriptors

Then fused into gradient boosting models for prediction.

---

## Slide 5 — System Architecture
```text
Lyrics ──► emotion model + VADER + keywords ──┐
                                               ├──► Gradient Boosting ──► valence / energy / vibe
Audio  ──► librosa (tempo, MFCC, spectral) ──┘
```

Saved model artifacts:
- `models/valence_model.joblib`
- `models/energy_model.joblib`
- `models/vibe_model.joblib`

---

## Slide 6 — CLI + Data Workflow I Implemented
I implemented an end-to-end CLI workflow:
- `fetch` → collect songs/lyrics/features from APIs
- `discover` → sample tracks and gather data
- `merge-kaggle` → join lyrics with large Spotify feature datasets
- `clean` / `enrich` → quality and metadata backfill
- `train` / `train-eval` → fit and evaluate models
- `predict` → score single songs from artist/title or raw lyrics

This made the project reproducible from data collection to inference.

---

## Slide 7 — Practical Engineering Decisions
- Added a **fast mode** (`MOOD_READER_FAST=1`) to skip heavy HF emotion inference for rapid iteration.
- Kept Spotify optional at inference so model can still run with text-only inputs.
- Added Kaggle merge path to handle Spotify API rate limits during dataset growth.

These decisions made experimentation significantly easier.

---

## Slide 8 — Vibe Taxonomy
I mapped moods onto a circumplex-style valence × energy space with 8 vibe classes:
- `happy`, `energetic`, `uplifting`, `chill`
- `romantic`, `melancholic`, `dark`, `aggressive`

This gave both:
- Continuous outputs (valence, energy)
- Human-friendly categorical output (vibe)

---

## Slide 9 — Milestones (from commit history)
1. **Initial build**  
   Multimodal song mood classifier scaffold and pipeline.

2. **Data pipeline expansion**  
   Kaggle merge, Spotify enrich, fetch checkpoints.

3. **Training/evaluation upgrade**  
   Lyrics + Spotify multimodal training/eval pipeline.

4. **Model artifact improvements**  
   Full emotion model evaluation outputs and charts.

5. **Documentation polish**  
   README badges, taxonomy, quickstart, CLI references.

---

## Slide 10 — Example Prediction Output
```json
{
  "valence": 0.721,
  "energy": 0.814,
  "valence_label": "positive",
  "energy_label": "high",
  "vibe": "energetic",
  "audio_analyzed": true
}
```

Interpretation:
- The song is predicted as positive/high-energy and likely workout/party style.

---

## Slide 11 — What This Project Demonstrates
- End-to-end ML product thinking (data → features → training → inference)
- Combining NLP and audio DSP in one prediction system
- Practical tradeoff handling (speed vs quality, API limits vs dataset scale)
- Usable command-line interface and strong documentation

---

## Slide 12 — Next Steps
- Expand labeled dataset diversity across genres/languages
- Add model calibration + confidence reporting
- Package as a small API/web app for interactive use
- Explore sequence models or multimodal deep fusion for higher accuracy

---

## Slide 13 — Closing
In this project, I built a **production-shaped prototype** for mood-aware music intelligence:
- data ingestion
- multimodal feature engineering
- model training/evaluation
- prediction interface

It is now in a strong state for either:
1) productization, or  
2) deeper ML experimentation.


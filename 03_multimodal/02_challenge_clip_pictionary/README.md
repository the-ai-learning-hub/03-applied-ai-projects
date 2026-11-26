# 🎮 Challenge 3 · CLIP Pictionary

**Can you describe a photo so well that CLIP picks it out of 36?** You get a secret photo, you describe it with words (or with your voice), and CLIP ranks the 36 photos of the table by similarity with your description. ⏱️ **≈ 15 min** · you play alone, on your own laptop (or in Colab).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/03_multimodal/02_challenge_clip_pictionary/Challenge_3_CLIP_Pictionary.ipynb) · 🏠 [Multimodal](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#challenge-3)

## Rules

The table: our 17 pets (`data/pets/cropped`) + 19 photos of `data/cats_and_dogs/validation`, in a fixed shuffled order. Up to **3 descriptions per level**; only the **best** counts.

| Rank of the secret photo | Points |
|---|---|
| #1 | 10 |
| #2 – #3 | 5 |
| #4 – #5 | 2 |
| #6 or lower | 0 |
| short bonus: #1 with ≤ 5 words | +3 |

| Level | Secret photo | Twist |
|---|---|---|
| 1 · Easy | the orange cat (`cat10000`) | — |
| 2 · Look-alikes | a dark Siamese-type cat on a red blanket (`cat9005`) | several similar cats on the table |
| 3 · Forbidden words | the AI-generated dog with a kitten on its head (`dog9016AI`) | no *cat, cats, kitten(s), kitty, kitties, dog(s), puppy, puppies, pup(s), doggy, doggie, pet(s), feline, canine*: a forbidden word = 0 for that try |
| 🎤 Voice bonus | the dog in a purple blanket (`dog9007`) | a WAV → Whisper-tiny → text → CLIP: **+5** if the photo is in the top 3 |

**Medals:** 🥉 Bronze ≥ 12 · 🥈 Silver ≥ 24 · 🥇 Gold ≥ 35 (max 44). The example tries score **33 (Silver)**.

## How to run

- **Locally:** `uv sync` once, open the notebook with the `.venv` kernel, *Run All*, then edit the ✏️ **YOUR TURN** cells (`LEVEL_1`, `LEVEL_2`, `LEVEL_3`, `VOICE_FILE`) and re-run them.
- **Your voice:** record a WAV (or FLAC/OGG; convert `.m4a` phone recordings first), put it in `data/audio/` and set `VOICE_FILE`. In Colab set `UPLOAD_WAV = True`.
- **Google Colab** (if your setup fails): click the badge above, *Runtime → Run all*.
- **Runtime:** ≈ 25 s on a laptop CPU (first run: + CLIP ≈ 600 MB and Whisper-tiny ≈ 150 MB downloads, already there if you ran notebook 05).

Scoring rules: [`src/challenges.py`](../../src/challenges.py) (`CH3_RULES`, `FORBIDDEN_WORDS`).

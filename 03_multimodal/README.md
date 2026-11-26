# 03 · Multimodal: everything is a vector

Text, images and sound all become vectors — and similar meaning means nearby points. One idea that powers semantic search, CLIP and speech-to-text. 🏠 [Lesson 03](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#multimodal)

| Notebook | Question | Time | Open |
|---|---|---|---|
| [05 · Everything is a vector](01_project_everything_is_a_vector/05_Everything_Is_A_Vector.ipynb) | Can one idea connect words, photos and voices? | 60 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/03_multimodal/01_project_everything_is_a_vector/05_Everything_Is_A_Vector.ipynb) |
| [🎮 Challenge 3 · CLIP Pictionary](02_challenge_clip_pictionary/Challenge_3_CLIP_Pictionary.ipynb) | Can you describe a photo so CLIP finds it first? | 15 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/03_multimodal/02_challenge_clip_pictionary/Challenge_3_CLIP_Pictionary.ipynb) |

## 🎯 Objectives

- **Text → vectors:** sentence embeddings of 10,000 tweets (MiniLM), a 2D map of meaning (t-SNE) and **semantic search vs keyword search**.
- **Images ↔ words (CLIP):** search *our own pets* with a sentence, classify cats vs dogs with **zero** training photos and compare with trained models, find similar photos.
- **Sound → image → text:** waveform → spectrogram ("sound becomes an image"), **Whisper** speech-to-text, and the chain *voice → text → sentiment*.

## 📋 Prerequisites

Notebooks 01–04 (TF-IDF, tweet sentiment, CNNs, transfer learning) and Lesson 02 (PCA, K-Means, Logistic Regression).

## 📦 Models (downloaded on first run, CPU is enough)

| Model | Used for | Size |
|---|---|---|
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | sentence embeddings (384 numbers) | ≈ 90 MB |
| [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | photos and captions in one space (512 numbers) | ≈ 600 MB |
| [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | speech → text | ≈ 150 MB |

After the downloads the notebook runs in a few minutes on a laptop CPU (the first run also spends ≈ 1 min embedding the tweets; they are then cached in `data/_cache/`).

## 🎙️ Your own voice

Put any WAV or FLAC file in `data/audio/` (Windows *Sound Recorder* → Settings → Recording format → WAV) and re-run section 5. In Colab, set `RECORD = True` in the recording cell to use your microphone.

## GPU (optional)

Not needed: every model in this folder runs on a laptop CPU.

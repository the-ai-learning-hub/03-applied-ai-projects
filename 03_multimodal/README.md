# 03 · Multimodal: everything is a vector

Text, images and sound all become vectors — and similar meaning means nearby points. One idea that powers semantic search, CLIP, speech-to-text and RAG. 🏠 [Lesson 03](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#multimodal)

| Notebook | Question | Time | Open |
|---|---|---|---|
| [05 · Everything is a vector](01_project_everything_is_a_vector/05_Everything_Is_A_Vector.ipynb) | Can one idea connect words, photos and voices? | 60 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/03_multimodal/01_project_everything_is_a_vector/05_Everything_Is_A_Vector.ipynb) |
| [🎮 Challenge 3 · CLIP Pictionary](02_challenge_clip_pictionary/Challenge_3_CLIP_Pictionary.ipynb) | Can you describe a photo so CLIP finds it first? | 15 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/03_multimodal/02_challenge_clip_pictionary/Challenge_3_CLIP_Pictionary.ipynb) |

## 🎯 Objectives

- **Text → vectors:** sentence embeddings of 10,000 tweets (MiniLM), a 2D map of meaning (t-SNE) and **semantic search vs keyword search**.
- **Images ↔ words (CLIP):** search *our own pets* with a sentence, classify cats vs dogs with **zero** training photos and compare with trained models, find similar photos.
- **Sound → image → text:** waveform → spectrogram ("sound becomes an image"), **Whisper** speech-to-text, and the chain *voice → text → sentiment*.
- **RAG with a local LLM:** retrieve → augment → generate with **Ollama** on your own laptop (no API keys, no data leaves your computer), ending with *voice → text → retrieval → answer*.

## 📋 Prerequisites

Notebooks 01–04 (TF-IDF, tweet sentiment, CNNs, transfer learning) and Lesson 02 (PCA, K-Means, Logistic Regression).

## 📦 Models (downloaded on first run, CPU is enough)

| Model | Used for | Size |
|---|---|---|
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | sentence embeddings (384 numbers) | ≈ 90 MB |
| [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | photos and captions in one space (512 numbers) | ≈ 600 MB |
| [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | speech → text | ≈ 150 MB |
| [`qwen3.5:2b`](https://ollama.com/library/qwen3.5) via Ollama (optional) | answers in the RAG section | ≈ 2.7 GB |

After the downloads the notebook runs in a few minutes on a laptop CPU (the first run also spends ≈ 1 min embedding the tweets; they are then cached in `data/_cache/`).

## 🦙 Ollama setup (optional, for section 6)

The notebook works without Ollama: it then prints the prompt that *would* be sent. To get real answers from a local model:

1. **Install** Ollama
   - Windows: `winget install Ollama.Ollama` (or the installer from <https://ollama.com/download>)
   - macOS: `brew install ollama` (or the app from the website) · Linux: `curl -fsSL https://ollama.com/install.sh | sh`
2. **Download a small model** (once, ≈ 2–3 GB): `ollama pull qwen3.5:2b`
   Alternatives: `llama3.2:3b` (≈ 2 GB) or, with 16 GB of RAM, `qwen3.5:4b` / `gemma4:e4b` (better answers, slower). Change `OLLAMA_MODEL` in the notebook to match.
3. **Check it is running:** the Ollama icon is in the system tray, or run `ollama serve`. Then `ollama run qwen3.5:2b "hello"` should answer.
4. Re-run the RAG cells. The notebook talks to `http://localhost:11434/api/chat` with plain `requests` — no API key, no account.

In **Google Colab** Ollama is not available by default: the RAG cells show the retrieved tweets and the prompt instead.

## 🎙️ Your own voice

Put any WAV or FLAC file in `data/audio/` (Windows *Sound Recorder* → Settings → Recording format → WAV) and re-run section 5. In Colab, set `RECORD = True` in the recording cell to use your microphone.

## GPU (optional)

Not needed. With an NVIDIA GPU, Ollama uses it automatically and answers several times faster.

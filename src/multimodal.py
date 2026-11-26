"""Helpers for notebook 05 · Everything is a vector (text, images and sound as vectors + local RAG).

Everything here runs on a CPU laptop with small models:

* sentence embeddings: ``sentence-transformers/all-MiniLM-L6-v2`` (~90 MB, 384 numbers per text);
* text <-> image: ``openai/clip-vit-base-patch32`` (~600 MB, 512 numbers per image or caption);
* speech -> text: ``openai/whisper-tiny`` (~150 MB);
* answers: a small local LLM served by `Ollama <https://ollama.com>`_ through its REST API
  (``http://localhost:11434``). No API keys, nothing leaves your computer.

Models are loaded lazily (only when a function needs them) and cached in memory, so importing
this module is instant and the tests never download anything.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
WHISPER_MODEL = "openai/whisper-tiny"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:2b"  # alternatives: llama3.2:3b, gemma4:e2b, qwen3.5:4b
SAMPLE_RATE = 16_000  # Whisper (and our WAV files) use 16 kHz

OLLAMA_HELP = """\
🦙 Ollama is not running on this computer, so there is no local LLM to answer.
   The retrieval step above still worked - this is the prompt that *would* be sent.

   To enable the last step (free, local, no API key):
     1. Install Ollama:  Windows -> winget install Ollama.Ollama   (or download from https://ollama.com/download)
                         macOS   -> brew install ollama            Linux -> curl -fsSL https://ollama.com/install.sh | sh
     2. Download a small model (~2-3 GB):   ollama pull {model}
     3. Make sure the Ollama app is running (tray icon) or run:   ollama serve
     4. Re-run this cell.
"""

OLLAMA_COLAB_HELP = """\
☁️ You are in Google Colab: "localhost" here is Google's remote machine, not your laptop,
   so your own Ollama cannot be reached. Retrieval worked; generation is skipped.
   To see real answers, run this notebook locally with Ollama installed (see the folder README).
"""


def in_colab() -> bool:
    """True when running inside a hosted Google Colab runtime."""
    import sys
    return "google.colab" in sys.modules


def _is_local(url: str) -> bool:
    return any(host in url for host in ("localhost", "127.0.0.1", "0.0.0.0"))


# --------------------------------------------------------------------------------------------
# Vectors: normalising, cosine similarity, top-k search, keyword baseline, 2D maps
# --------------------------------------------------------------------------------------------
def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Scale every row to length 1 (then a dot product *is* the cosine similarity)."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of ``A`` and every row of ``B`` (shape ``len(A) x len(B)``)."""
    A = l2_normalize(np.atleast_2d(A))
    B = l2_normalize(np.atleast_2d(B))
    return A @ B.T


def top_k(query_vec: np.ndarray, corpus_vecs: np.ndarray, k: int = 5,
          exclude: Iterable[int] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine scores of the ``k`` corpus rows closest to ``query_vec`` (best first).

    ``exclude`` lets you skip some indices (e.g. the query image itself in "find similar photos").
    """
    scores = cosine_similarity(query_vec, corpus_vecs)[0]
    for i in exclude:
        scores[i] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    idx = np.argsort(-scores, kind="stable")[:k]
    return idx, scores[idx]


_WORD = re.compile(r"[a-z0-9']+")
_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "is", "are", "i",
         "my", "me", "it", "this", "that", "with", "so", "be", "was", "am", "do", "what", "when",
         "about", "most", "people", "you"}


def tokenize(text: str) -> list[str]:
    """Lower-case words without punctuation or very common stop words."""
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def keyword_search(query: str, docs: Sequence[str], k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Old-school search: score = how many of the query's words appear in the document.

    Returns indices and scores (number of shared words), best first; documents with no shared
    word are never returned - that is exactly the weakness semantic search fixes.
    """
    q = set(tokenize(query))
    scores = np.array([len(q & set(tokenize(d))) for d in docs], dtype=float)
    idx = np.argsort(-scores, kind="stable")[:k]
    idx = idx[scores[idx] > 0]
    return idx, scores[idx]


def words(text: str) -> list[str]:
    """All words in order, lower-case, without punctuation (no stop-word removal)."""
    return _WORD.findall(text.lower().replace("’", "'"))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = (substituted + deleted + inserted words) / words in the reference (0 = perfect).

    The standard speech-to-text metric: a word-level edit distance on the ordered word sequences.
    """
    ref, hyp = words(reference), words(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def project_2d(X: np.ndarray, method: str = "pca", seed: int = 42, perplexity: float = 30.0) -> np.ndarray:
    """Squash high-dimensional vectors to 2D for plotting (``"pca"`` or ``"tsne"``).

    PCA keeps the global shape (fast, linear); t-SNE keeps neighbours together (slower, nicer maps).
    """
    X = np.asarray(X, dtype=np.float32)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        Xp = PCA(n_components=min(50, X.shape[1], len(X)), random_state=seed).fit_transform(X)
        return TSNE(n_components=2, perplexity=min(perplexity, (len(X) - 1) / 3), init="pca",
                    random_state=seed).fit_transform(Xp)
    raise ValueError("method must be 'pca' or 'tsne'")


# --------------------------------------------------------------------------------------------
# Text -> vector (sentence-transformers)
# --------------------------------------------------------------------------------------------
@lru_cache(maxsize=2)
def load_text_model(name: str = TEXT_MODEL):
    """Load (once) a sentence-transformers model on the CPU."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")


def embed_texts(texts: Sequence[str], model=None, batch_size: int = 128) -> np.ndarray:
    """One normalised vector per text (MiniLM: 384 numbers)."""
    model = model or load_text_model()
    vecs = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)


# --------------------------------------------------------------------------------------------
# Images and captions -> the same vector space (CLIP)
# --------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_clip(name: str = CLIP_MODEL):
    """Load (once) CLIP's model and processor; returns ``(model, processor)``."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(name).eval()
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor


def _as_tensor(out):
    """``get_*_features`` returns a tensor in older transformers and an output object in newer ones."""
    if hasattr(out, "pooler_output") and not hasattr(out, "shape"):
        return out.pooler_output
    return out


def clip_embed_images(images: Sequence, batch_size: int = 16) -> np.ndarray:
    """CLIP vectors (512 numbers, normalised) for image paths or PIL images."""
    import torch
    from PIL import Image

    model, processor = load_clip()
    vecs = []
    for start in range(0, len(images), batch_size):
        batch = [Image.open(im).convert("RGB") if isinstance(im, (str, Path)) else im.convert("RGB")
                 for im in images[start:start + batch_size]]
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt")
            vecs.append(_as_tensor(model.get_image_features(**inputs)).numpy())
    return l2_normalize(np.concatenate(vecs))


def clip_embed_texts(texts: Sequence[str]) -> np.ndarray:
    """CLIP vectors (512 numbers, normalised) for captions such as ``"a photo of a cat"``."""
    import torch

    model, processor = load_clip()
    with torch.no_grad():
        # CLIP reads at most 77 tokens: longer descriptions are cut instead of crashing
        inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
        vecs = _as_tensor(model.get_text_features(**inputs)).numpy()
    return l2_normalize(vecs)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def zero_shot_probs(image_vecs: np.ndarray, label_vecs: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """Zero-shot classification: softmax over (scaled) cosine similarity image <-> each label prompt.

    CLIP was trained with a scale (inverse temperature) of about 100, so ``scale=100`` turns
    similarities such as 0.26 vs 0.21 into scores like 99% vs 1%. These are *relative* scores over
    the prompts you chose (they always sum to 1 across them), not a calibrated confidence: add or
    change a prompt and every score changes. Rows = images, columns = labels.
    """
    return softmax(scale * cosine_similarity(image_vecs, label_vecs), axis=1)


def clip_zero_shot(images: Sequence, labels: Sequence[str],
                   template: str = "a photo of a {}") -> np.ndarray:
    """Probabilities (rows = images, columns = ``labels``) using prompts like ``"a photo of a cat"``."""
    prompts = [template.format(lab) for lab in labels]
    return zero_shot_probs(clip_embed_images(images), clip_embed_texts(prompts))


# --------------------------------------------------------------------------------------------
# Sound -> numbers -> image (spectrogram) -> text (Whisper)
# --------------------------------------------------------------------------------------------
def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Read a WAV/FLAC/OGG file as mono float32 in [-1, 1], resampled to ``sr`` (no ffmpeg needed)."""
    import soundfile as sf

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32), sr


def mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 80) -> np.ndarray:
    """Log-mel spectrogram in dB (rows = frequency bands, columns = time) - the 'image' Whisper sees."""
    import librosa

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)


def plot_waveform_and_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE, title: str = "",
                                  axes=None):
    """Two stacked panels sharing the time axis: the raw wave and its log-mel spectrogram."""
    import matplotlib.pyplot as plt

    from src.plotting import PALETTE

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.6]})
    else:
        fig = axes[0].figure
    t = np.arange(len(y)) / sr
    axes[0].plot(t, y, color=PALETTE["blue"], lw=0.6)
    axes[0].set(ylabel="amplitude", title=title or "Waveform: one number per 1/16000 s")
    S = mel_spectrogram(y, sr)
    axes[1].imshow(S, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, len(y) / sr, 0, S.shape[0]])
    axes[1].set(xlabel="time (s)", ylabel="mel band (low → high pitch)")
    axes[1].grid(False)
    return fig, axes


@lru_cache(maxsize=1)
def load_whisper(name: str = WHISPER_MODEL):
    """Load (once) a Whisper speech-recognition pipeline on the CPU."""
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=name, device="cpu")


def transcribe(audio: str | Path | np.ndarray, sr: int = SAMPLE_RATE, language: str | None = "en",
               task: str = "transcribe") -> str:
    """Speech -> text with Whisper. ``audio`` is a file path or a 1-D array at ``sr`` Hz.

    We pass the raw numbers (not the file name) so no ffmpeg is needed. ``task="translate"``
    makes Whisper answer in English whatever language you speak; ``language=None`` auto-detects.
    """
    if isinstance(audio, (str, Path)):
        audio, sr = load_audio(audio, sr)
    generate_kwargs = {"task": task}
    if language:
        generate_kwargs["language"] = language
    out = load_whisper()({"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": sr},
                         generate_kwargs=generate_kwargs)
    return out["text"].strip()


# --------------------------------------------------------------------------------------------
# RAG with a local LLM (Ollama REST API, no keys)
# --------------------------------------------------------------------------------------------
def ollama_available(url: str = OLLAMA_URL, timeout: float = 2.0) -> bool:
    """True if an Ollama server answers at ``url`` (never raises)."""
    try:
        import requests
        return requests.get(f"{url}/api/tags", timeout=timeout).ok
    except Exception:
        return False


def ollama_models(url: str = OLLAMA_URL, timeout: float = 2.0) -> list[str]:
    """Names of the models already pulled into the local Ollama (empty list if not running)."""
    try:
        import requests
        r = requests.get(f"{url}/api/tags", timeout=timeout)
        return [m["name"] for m in r.json().get("models", [])] if r.ok else []
    except Exception:
        return []


def build_rag_prompt(question: str, context_docs: Sequence[str]) -> str:
    """Augment step: paste the retrieved documents above the question, with clear instructions."""
    context = "\n".join(f"[{i}] {doc.strip()}" for i, doc in enumerate(context_docs, start=1))
    return (
        "You are a helpful analyst. Answer the question using ONLY the tweets below.\n"
        "If the tweets do not contain the answer, say you don't know.\n"
        "Be brief (3-5 bullet points) and cite tweet numbers like [2].\n\n"
        f"Tweets:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


def ask_ollama(question: str, context_docs: Sequence[str] = (), model: str = OLLAMA_MODEL,
               url: str = OLLAMA_URL, timeout: float = 180.0, temperature: float = 0.2,
               verbose: bool = True) -> str | None:
    """Generate step: send the RAG prompt to a local Ollama model and return its answer.

    Never raises: if Ollama is not running, or the model is missing, it prints friendly
    instructions plus the prompt that *would* have been sent, and returns ``None``.
    In hosted Colab (where ``localhost`` is Google's machine) it says that generation is skipped.
    """
    prompt = build_rag_prompt(question, context_docs) if context_docs else question
    if not ollama_available(url):
        if verbose:
            colab = in_colab() and _is_local(url)
            print(OLLAMA_COLAB_HELP if colab else OLLAMA_HELP.format(model=model))
            print("─" * 30, "prompt that would be sent", "─" * 30)
            print(prompt)
        return None
    try:
        import requests
        r = requests.post(
            f"{url}/api/chat",
            json={"model": model, "stream": False, "think": False,  # skip "thinking" mode: faster answers
                  "messages": [{"role": "user", "content": prompt}],
                  "options": {"temperature": temperature, "seed": 42}},
            timeout=timeout,
        )
        if r.status_code == 404:
            if verbose:
                print(f"🦙 Ollama is running but the model '{model}' is not downloaded yet.\n"
                      f"   Run in a terminal:  ollama pull {model}\n"
                      f"   Models you already have: {ollama_models(url) or 'none'}")
            return None
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as err:  # timeouts, connection drops, unexpected JSON...
        if verbose:
            print(f"🦙 Could not get an answer from Ollama ({type(err).__name__}: {err}).\n"
                  "   Check that the Ollama app is running and try again.")
        return None


# --------------------------------------------------------------------------------------------
# Optional: predictions of the model trained in notebook 04 (if they were saved)
# --------------------------------------------------------------------------------------------
_FILE_KEYS = ("file", "filename", "image", "path", "name")
_PRED_KEYS = ("pred", "predicted", "prediction", "pred_label", "label_pred")


def _find_records(obj) -> list[dict]:
    """Walk any JSON structure and return the first list of dicts that look like per-image predictions."""
    if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
        if any(k in obj[0] for k in _FILE_KEYS) and any(k in obj[0] for k in _PRED_KEYS):
            return obj
    children = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, list) else []
    for child in children:
        found = _find_records(child)
        if found:
            return found
    return []


def load_trained_pet_predictions(results_dir: str | Path):
    """Look for notebook 04's saved predictions on our pets (``*pet*.json`` in ``results_dir``).

    Returns a DataFrame with columns ``file`` (e.g. ``cat9001.jpg``) and ``pred`` (``cat``/``dog``),
    or ``None`` if nothing usable is found. Never raises.
    """
    import json

    import pandas as pd

    try:
        for path in sorted(Path(results_dir).glob("*pet*.json")):
            records = _find_records(json.loads(path.read_text(encoding="utf-8")))
            if not records:
                continue
            fkey = next(k for k in _FILE_KEYS if k in records[0])
            pkey = next(k for k in _PRED_KEYS if k in records[0])
            df = pd.DataFrame({"file": [Path(str(r[fkey])).name for r in records],
                               "pred": [str(r[pkey]).lower().rstrip("s") for r in records]})
            df.attrs["source"] = path.name
            return df
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------------------------
# Small drawing helper: a chain of boxes (used for the pipeline diagrams)
# --------------------------------------------------------------------------------------------
def plot_pipeline(steps: Sequence[str], ax=None, colors: Sequence[str] | None = None, title: str = ""):
    """Draw ``steps`` as rounded boxes from left to right, joined by arrows (text only, no emoji)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from src.plotting import PALETTE

    n = len(steps)
    if ax is None:
        _, ax = plt.subplots(figsize=(min(12, 2.3 * n), 1.8))
    colors = list(colors or [PALETTE["blue"]] * n)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (text, colour) in enumerate(zip(steps, colors)):
        ax.add_patch(FancyBboxPatch((i + 0.07, 0.15), 0.86, 0.7, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=colour, ec=colour, alpha=0.14, lw=0))
        ax.add_patch(FancyBboxPatch((i + 0.07, 0.15), 0.86, 0.7, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="none", ec=colour, lw=1.6))
        ax.text(i + 0.5, 0.5, text, ha="center", va="center", fontsize=10, color=PALETTE["navy"])
        if i < n - 1:
            ax.annotate("", xy=(i + 1.07, 0.5), xytext=(i + 0.93, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=PALETTE["grey"], lw=1.6))
    if title:
        ax.set_title(title)
    return ax


# --------------------------------------------------------------------------------------------
# Offline-friendly model loading
# --------------------------------------------------------------------------------------------
def try_load(what: str, loader, *args, **kwargs):
    """Run a model ``loader`` (e.g. :func:`load_clip`); on failure print a friendly message and return None.

    The first run needs internet to download each model. If that fails (offline, proxy, server
    hiccup) the notebook keeps going: cells that need this model print "skipped" instead of crashing.
    """
    try:
        return loader(*args, **kwargs)
    except Exception as err:  # no internet, blocked download, disk full...
        print(f"⚠️ Could not load {what} ({type(err).__name__}: {str(err)[:120]}).\n"
              "   The first run must download it: check your internet connection and re-run this cell.\n"
              "   Cells that need it will print 'skipped'; the other sections still work.")
        return None

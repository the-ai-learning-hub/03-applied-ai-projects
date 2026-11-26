"""Fast tests for src.multimodal: no model downloads, no network (requests is mocked)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from src import multimodal as mm
from src.paths import data_dir


def test_cosine_similarity_basic():
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    B = np.array([[3.0, 0.0], [1.0, 1.0]])
    S = mm.cosine_similarity(A, B)
    assert S.shape == (2, 2)
    assert S[0, 0] == pytest.approx(1.0)
    assert S[1, 0] == pytest.approx(0.0)
    assert S[0, 1] == pytest.approx(1 / np.sqrt(2), rel=1e-5)


def test_top_k_order_and_exclude():
    corpus = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    idx, scores = mm.top_k(np.array([1.0, 0.0]), corpus, k=2)
    assert list(idx) == [0, 1]
    assert scores[0] >= scores[1]
    idx, _ = mm.top_k(np.array([1.0, 0.0]), corpus, k=2, exclude=[0])
    assert list(idx) == [1, 2]


def test_keyword_search_needs_shared_words():
    docs = ["my flight was delayed", "lovely sunny beach", "the flight crew was great"]
    idx, scores = mm.keyword_search("flight delayed", docs, k=5)
    assert list(idx) == [0, 2]
    assert list(scores) == [2, 1]
    idx, _ = mm.keyword_search("airplane late", docs)
    assert len(idx) == 0  # no shared word -> keyword search finds nothing


def test_project_2d_shapes():
    X = np.random.default_rng(0).normal(size=(40, 16))
    assert mm.project_2d(X, "pca").shape == (40, 2)
    assert mm.project_2d(X, "tsne", perplexity=5).shape == (40, 2)
    with pytest.raises(ValueError):
        mm.project_2d(X, "umap")


def test_zero_shot_probs_rows_sum_to_one():
    imgs = np.array([[1.0, 0.1], [0.1, 1.0]])
    labels = np.array([[1.0, 0.0], [0.0, 1.0]])
    P = mm.zero_shot_probs(imgs, labels)
    assert np.allclose(P.sum(axis=1), 1)
    assert P[0].argmax() == 0 and P[1].argmax() == 1


def test_load_audio_and_spectrogram(tmp_path):
    import soundfile as sf

    sr = 8_000
    t = np.arange(sr) / sr
    sf.write(tmp_path / "tone.wav", 0.5 * np.sin(2 * np.pi * 440 * t), sr)
    y, out_sr = mm.load_audio(tmp_path / "tone.wav")  # resampled to 16 kHz
    assert out_sr == 16_000 and y.ndim == 1
    assert abs(len(y) - 16_000) < 50
    S = mm.mel_spectrogram(y, out_sr)
    assert S.shape[0] == 80 and S.max() == pytest.approx(0.0)


def test_word_error_rate():
    assert mm.word_error_rate("My flight got cancelled.", "my flight got cancelled") == 0
    assert mm.word_error_rate("a b c d", "a x c d") == pytest.approx(0.25)    # 1 substitution
    assert mm.word_error_rate("a b c d", "a c d") == pytest.approx(0.25)      # 1 deletion
    assert mm.word_error_rate("a b", "b a") == pytest.approx(1.0)             # order matters
    assert mm.words("Don’t stop!") == ["don't", "stop"]


def test_try_load_is_friendly(capsys):
    def broken():
        raise OSError("offline")

    assert mm.try_load("a model", broken) is None
    assert "re-run this cell" in capsys.readouterr().out
    assert mm.try_load("ok", lambda x: x + 1, 1) == 2


def test_ollama_colab_message(capsys):
    with mock.patch("requests.get", side_effect=ConnectionError("refused")), \
            mock.patch.object(mm, "in_colab", return_value=True):
        assert mm.ask_ollama("Q?", ["doc"]) is None
    out = capsys.readouterr().out
    assert "Colab" in out and "generation is skipped" in out and "[1] doc" in out


def test_build_rag_prompt_numbers_documents():
    prompt = mm.build_rag_prompt("Why?", ["first tweet", "second tweet"])
    assert "[1] first tweet" in prompt and "[2] second tweet" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_ollama_unavailable_is_friendly(capsys):
    with mock.patch("requests.get", side_effect=ConnectionError("refused")):
        assert mm.ollama_available() is False
        assert mm.ollama_models() == []
        answer = mm.ask_ollama("What do people complain about?", ["my flight is late"])
    assert answer is None
    out = capsys.readouterr().out
    assert "ollama pull" in out and "[1] my flight is late" in out


def test_ask_ollama_success():
    ok = mock.Mock(ok=True, status_code=200)
    ok.json.return_value = {"message": {"role": "assistant", "content": "  Delays [1]. "}}
    with mock.patch("requests.get", return_value=mock.Mock(ok=True)), \
            mock.patch("requests.post", return_value=ok) as post:
        answer = mm.ask_ollama("Q?", ["doc"], model="tiny:1b")
    assert answer == "Delays [1]."
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "tiny:1b" and payload["stream"] is False
    assert "doc" in payload["messages"][0]["content"]


def test_ask_ollama_missing_model(capsys):
    missing = mock.Mock(ok=False, status_code=404)
    tags = mock.Mock(ok=True)
    tags.json.return_value = {"models": [{"name": "other:1b"}]}
    with mock.patch("requests.get", return_value=tags), \
            mock.patch("requests.post", return_value=missing):
        assert mm.ask_ollama("Q?", ["doc"], model="absent:1b") is None
    assert "ollama pull absent:1b" in capsys.readouterr().out


def test_load_trained_pet_predictions(tmp_path):
    import json

    assert mm.load_trained_pet_predictions(tmp_path) is None
    (tmp_path / "pets_predictions.json").write_text(json.dumps(
        {"model": "x", "pets": [{"file": "data/pets/cropped/cats/cat9001.jpg", "pred": "cats"},
                                {"file": "dog9006.jpg", "pred": "dog"}]}), encoding="utf-8")
    df = mm.load_trained_pet_predictions(tmp_path)
    assert list(df["file"]) == ["cat9001.jpg", "dog9006.jpg"]
    assert list(df["pred"]) == ["cat", "dog"]

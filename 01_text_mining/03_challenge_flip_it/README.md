# 🎚️ Challenge 1 · Flip It!

**How few words must you change to make a sentiment AI change its mind?** The model is sure that *"I love this sunny day"* is 85 % positive. Edit the sentence: every word lights up by how hard it pushes the model, a gauge slides between NEGATIVE and POSITIVE as you type, and when the verdict flips… 🎉. ⏱️ **≈ 15 min** · you play alone, on your own laptop (or in Colab).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/03_challenge_flip_it/Challenge_1_Flip_It.ipynb) · 🏠 [Text Mining](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#challenge-1)

## The model

TF-IDF (single words) + Logistic Regression, trained in the notebook on the 39,000 tweets of notebook 02 in ≈ 1.5 s (macro-F1 0.70 on the test tweets). Because it is linear, the score is *bias + one push per word*, so the game can show every word's exact push live.

## Rules

**1 swap** = one word added, removed or replaced (as the model reads it: lower case, `!` and `,` count as words). Your **best** valid flip of each level counts.

| Level | Start sentence | Flip it to | Twist | 🥇 | 🥈 |
|---|---|---|---|---|---|
| 1 · 🌞 Warm-up | *I love this sunny day* (85 % pos) | NEGATIVE | none | 1 | ≤ 2 |
| 2 · 🚑 Rescue mission | *this is the worst headache ever* (98 % neg) | POSITIVE | the obvious word is not the one | 1 | ≤ 3 |
| 3 · 🎭 Liar mode | *this new phone is awesome* (79 % pos) | NEGATIVE | a human must still read it as happy: honour box + obvious sad words banned | 1 | ≤ 3 |
| 4 · ➕ Add-only | *I lost my keys and missed the bus* (99 % neg) | POSITIVE | you may only add words | 3 | ≤ 5 |
| 5 · 💀 Boss | *happy birthday! thank you so much, love you all, have a great day! haha* (99.98 % pos) | NEGATIVE | none | 3 | ≤ 5 |

Any flip within the rules is at least 🥉. **Stars per level:** 🥇 3 · 🥈 2 · 🥉 1. **Medals:** 🥉 Bronze ≥ 5 ⭐ · 🥈 Silver ≥ 9 ⭐ · 🥇 Gold ≥ 13 ⭐ (out of 15). Every gold has been checked to be possible.

## How to run

- **Locally:** `uv sync` once, open the notebook with the project's `.venv` kernel, *Run All*, then play in the **🎚️ Play!** cell (an `ipywidgets` game; type and everything updates).
- **Google Colab** (if your setup fails): click the badge above, *Runtime → Run all*. The first cell clones the repository; the tweets download automatically.
- **No widgets?** The same game in one line: `flip("your sentence", level=1)` (`honest=True` in Liar mode), and `game.scorecard()` for your medal.
- **Runtime:** ≈ 5 s on a laptop CPU, no model download.

The game (`FlipGame`, `FLIP_LEVELS`, scoring and HTML) lives in [`src/challenges.py`](../../src/challenges.py), tested in `src/tests/test_challenges.py`.

"""Shared plotting style and helpers for Lesson 03.

Call :func:`apply_style` once at the top of a notebook. Colours follow the Okabe–Ito palette
(safe for colour-blind readers); helpers also use markers or labels, never colour alone.
Titles should state the takeaway ("TF-IDF beats every lexicon"), not just the variable names.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#: Okabe–Ito palette (colour-blind safe), with names used across the lesson.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#6B7A90",
    "navy": "#0E1A2E",
}
CYCLE = [PALETTE[k] for k in ("blue", "orange", "green", "red", "purple", "sky", "grey")]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

#: Fixed colours for the two classes used across the lesson (same meaning everywhere).
CLASS_COLORS = {
    "neg": PALETTE["red"], "pos": PALETTE["blue"],
    "spam": PALETTE["red"], "ham": PALETTE["blue"],
    "cat": PALETTE["orange"], "dog": PALETTE["sky"],
}


def apply_style() -> None:
    """Apply the lesson's Matplotlib style (light background, readable on a projector)."""
    mpl.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.prop_cycle": mpl.cycler(color=CYCLE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "font.size": 11,
        }
    )


def label_bars(ax: plt.Axes, values: Sequence[float], decimals: int = 2, suffix: str = "",
               horizontal: bool = True) -> plt.Axes:
    """Write each value at the end of its bar."""
    values = list(values)
    span = max(values) if values else 1
    for i, v in enumerate(values):
        text = f"{v:.{decimals}f}{suffix}"
        if horizontal:
            ax.text(v + span * 0.015, i, text, va="center", fontsize=10)
        else:
            ax.text(i, v + span * 0.015, text, ha="center", va="bottom", fontsize=10)
    if horizontal:
        ax.set_xlim(right=span * 1.15)
    else:
        ax.set_ylim(top=span * 1.12)
    return ax


def save_fig(fig: plt.Figure, path: str | Path) -> Path:
    """Save ``fig`` as PNG (folders are created if needed) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def plot_history(history: dict[str, Sequence[float]], ylabel: str = "loss",
                 title: str = "Training curves", ax: plt.Axes | None = None) -> plt.Axes:
    """Plot one line per key in ``history`` (e.g. ``{"train": [...], "validation": [...]}``)."""
    ax = ax or plt.gca()
    for (name, values), marker in zip(history.items(), MARKERS):
        values = list(values)
        every = max(1, len(values) // 12)
        ax.plot(range(1, len(values) + 1), values, marker=marker, markevery=every, ms=5, label=name)
    ax.set(xlabel="epoch", ylabel=ylabel, title=title)
    ax.legend()
    return ax


def plot_confusion(cm: np.ndarray, labels: Sequence[str], ax: plt.Axes | None = None,
                   title: str = "Confusion matrix", normalize: bool = True) -> plt.Axes:
    """Compact confusion matrix: row-normalised colours, with counts *and* % in each cell."""
    ax = ax or plt.gca()
    cm = np.asarray(cm)
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(share if normalize else cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if share[i, j] > 0.6 else PALETTE["navy"]
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center",
                    fontsize=11, color=colour)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    ax.grid(False)
    return ax


def leaderboard(scores: dict[str, float], baseline: float | None = None, metric: str = "macro-F1",
                title: str | None = None, ax: plt.Axes | None = None,
                highlight: str | None = None) -> plt.Axes:
    """Horizontal bar 'leaderboard' sorted by score, with an optional baseline line."""
    ax = ax or plt.gca()
    items = sorted(scores.items(), key=lambda kv: kv[1])
    names, values = zip(*items) if items else ((), ())
    colours = [PALETTE["orange"] if n == highlight else PALETTE["blue"] for n in names]
    ax.barh(range(len(values)), values, color=colours)
    ax.set_yticks(range(len(names)), names)
    label_bars(ax, values, decimals=3)
    if baseline is not None:
        ax.axvline(baseline, color=PALETTE["red"], ls="--", lw=1.5)
        ax.text(baseline, len(values) - 0.4, f" baseline {baseline:.3f}", color=PALETTE["red"],
                fontsize=10, va="bottom")
    ax.set(xlabel=metric, title=title or f"Leaderboard ({metric})")
    ax.grid(axis="y", visible=False)
    return ax

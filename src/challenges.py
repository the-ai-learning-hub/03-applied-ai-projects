"""Scoring rules and the scorecard figure for the three Lesson 03 class challenges.

* Challenge 1 · Fool the Sentiment Model  (``01_text_mining/03_challenge_fool_the_model``)
* Challenge 2 · YOLO Scavenger Hunt       (``02_computer_vision/03_challenge_yolo``)
* Challenge 3 · CLIP Pictionary           (``03_multimodal/02_challenge_clip_pictionary``)

Everything here is plain Python + NumPy/pandas/Matplotlib: no model is loaded, so the tests run in
a second. The notebooks call the models and pass the *answers* (labels, probabilities, detected
classes, ranks) to these functions. Every rule lives in one constant, so the notebook, the README
and the slides can quote the exact same numbers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------------
# Levels (medals)
# --------------------------------------------------------------------------------------------------

#: Minimum points for each medal, per challenge (below Bronze you are a "Rookie").
LEVELS = {
    "fool_the_model": {"Bronze": 10, "Silver": 25, "Gold": 40},
    "yolo_hunt": {"Bronze": 10, "Silver": 25, "Gold": 40},
    "clip_pictionary": {"Bronze": 12, "Silver": 24, "Gold": 35},
}
MEDAL_EMOJI = {"Rookie": "🎯", "Bronze": "🥉", "Silver": "🥈", "Gold": "🥇"}
MEDAL_COLORS = {"Rookie": "#6B7A90", "Bronze": "#A8612B", "Silver": "#8A96A3", "Gold": "#D9A400"}


def medal_for(points: float, thresholds: Mapping[str, int]) -> tuple[str, str | None, int | None]:
    """Medal reached with ``points``, the next medal and how many points are still missing.

    >>> medal_for(27, {"Bronze": 10, "Silver": 25, "Gold": 40})
    ('Silver', 'Gold', 13)
    """
    reached, nxt, missing = "Rookie", None, None
    for name, minimum in sorted(thresholds.items(), key=lambda kv: kv[1]):
        if points >= minimum:
            reached = name
        elif nxt is None:
            nxt, missing = name, int(np.ceil(minimum - points))
    return reached, nxt, missing


def clean_entries(entries: Iterable, max_entries: int, key=lambda e: e) -> list:
    """Drop empty and repeated entries (same text, ignoring case/spaces) and keep the first ``max_entries``.

    ``key`` picks the text of an entry (e.g. ``lambda e: e[0]`` for ``(text, label)`` pairs).
    The cap stops "points farming" by pasting the same trick 50 times.
    """
    seen, kept = set(), []
    for entry in entries:
        text = " ".join(str(key(entry)).lower().split())
        if not text or text in seen:
            continue
        seen.add(text)
        kept.append(entry)
        if len(kept) == max_entries:
            break
    return kept


# --------------------------------------------------------------------------------------------------
# Challenge 1 · Fool the Sentiment Model
# --------------------------------------------------------------------------------------------------

#: Points of Challenge 1 (the notebook, the README and the slides quote these).
CH1_RULES = {
    "max_sentences": 8,        # Round 1: only the first 8 different sentences count
    "per_judge_fooled": 2,     # Round 1: a judge predicts the opposite of your TRUE label
    "both_fooled_bonus": 3,    # Round 1: extra if BOTH judges are fooled by the same sentence
    "max_spam": 6,             # Round 2: only the first 6 different messages count
    "spam_passes": 3,          # Round 2: a real spam (self-declared) gets P(spam) < 0.5
    "stealth_below": 0.10,     # Round 2: ... and if P(spam) < 0.10
    "stealth_bonus": 2,        # ... you also get the stealth bonus
}


def judge_points(true_label: str, pred_a: str, pred_b: str, rules: Mapping = CH1_RULES) -> dict:
    """Round 1 points for one sentence: which judges were fooled and how many points that is."""
    if true_label not in ("pos", "neg"):
        raise ValueError(f"true label must be 'pos' or 'neg', got {true_label!r}")
    fooled_a, fooled_b = pred_a != true_label, pred_b != true_label
    points = rules["per_judge_fooled"] * (fooled_a + fooled_b)
    bonus = rules["both_fooled_bonus"] if (fooled_a and fooled_b) else 0
    return {"fooled_a": bool(fooled_a), "fooled_b": bool(fooled_b), "points": int(points), "bonus": int(bonus),
            "total": int(points + bonus)}


def spam_points(p_spam: float, still_spam: bool, rules: Mapping = CH1_RULES, threshold: float = 0.5) -> dict:
    """Round 2 points for one message: it must *still be spam* (your honest call) and get past the filter."""
    passed = bool(still_spam) and p_spam < threshold
    stealth = passed and p_spam < rules["stealth_below"]
    points = rules["spam_passes"] if passed else 0
    bonus = rules["stealth_bonus"] if stealth else 0
    return {"passed": passed, "stealth": bool(stealth), "points": int(points), "bonus": int(bonus),
            "total": int(points + bonus)}


def term_contributions(text: str, vectorizer, model, positive_class: str | None = None) -> pd.DataFrame:
    """How much each term of ``text`` pushed a linear model (TF-IDF value × model weight).

    Positive ``contribution`` pushes towards ``positive_class`` (default: ``model.classes_[1]``).
    Sorted by absolute effect, strongest first. Works for ``LogisticRegression`` (``coef_``) and
    ``MultinomialNB`` (log-probability difference between the two classes).
    """
    row = vectorizer.transform([text])
    if row.nnz == 0:
        return pd.DataFrame(columns=["term", "tfidf", "weight", "contribution"])
    if hasattr(model, "coef_"):
        weights = np.ravel(model.coef_)
    else:
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    classes = list(model.classes_)
    sign = 1.0 if positive_class in (None, classes[1]) else -1.0
    terms = np.asarray(vectorizer.get_feature_names_out())[row.indices]
    w = sign * weights[row.indices]
    frame = pd.DataFrame({"term": terms, "tfidf": row.data, "weight": w, "contribution": row.data * w})
    order = frame["contribution"].abs().sort_values(ascending=False).index
    return frame.loc[order].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# Challenge 2 · YOLO Scavenger Hunt
# --------------------------------------------------------------------------------------------------

#: Points of Challenge 2.
CH2_RULES = {
    "per_distinct_class": 1,   # every different COCO class YOLO confirmed during your game (once)
    "per_hunt_item": 3,        # every item of the hunt list confirmed (once, on top of the +1)
    "per_fool": 5,             # "Fool YOLO": a confirmed detection of something that is not really there
    "max_fools": 3,            # at most 3 Fool-YOLO claims count
    "min_confidence": 0.35,    # detections below this confidence do not count
    "min_frames": 3,           # ... and a class must be seen in 3 frames in a row to be confirmed
    "time_limit": 60,          # webcam game: 60 seconds
}

#: Default hunt list: COCO class names (YOLO only knows the 80 COCO classes).
HUNT_LIST = ["cup", "bottle", "cell phone", "book", "scissors", "keyboard", "mouse", "laptop",
             "backpack", "potted plant"]


def hunt_score(detections: Mapping[str, Iterable[str]], hunt_list: Sequence[str] = HUNT_LIST,
               fooled: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
               rules: Mapping = CH2_RULES) -> dict:
    """Score a scavenger hunt.

    ``detections`` maps each photo name to the classes YOLO found in it (already filtered by
    confidence). ``fooled`` lists ``(photo name, class)`` claims (or a ``{photo: class}`` dict): a class
    YOLO "saw" that is not really there. A claim counts only if YOLO really detected that class in that
    photo, each claim once, at most ``rules["max_fools"]`` claims.
    Returns the points per rule, the found/missing hunt items and the valid/invalid fool claims.
    """
    found = sorted({c for classes in detections.values() for c in classes})
    hunt_found = [h for h in hunt_list if h in found]
    hunt_missing = [h for h in hunt_list if h not in found]
    valid, invalid = [], []
    claims = fooled.items() if isinstance(fooled, Mapping) else (fooled or [])
    for photo, cls in dict.fromkeys(tuple(c) for c in claims):
        ok = cls in set(detections.get(photo, ())) and len(valid) < rules["max_fools"]
        (valid if ok else invalid).append((photo, cls))
    breakdown = {
        "distinct objects": rules["per_distinct_class"] * len(found),
        "hunt-list items": rules["per_hunt_item"] * len(hunt_found),
        "Fool YOLO": rules["per_fool"] * len(valid),
    }
    return {"breakdown": breakdown, "total": int(sum(breakdown.values())), "found": found,
            "hunt_found": hunt_found, "hunt_missing": hunt_missing, "fools_valid": valid,
            "fools_invalid": invalid}


# --------------------------------------------------------------------------------------------------
# Challenge 3 · CLIP Pictionary
# --------------------------------------------------------------------------------------------------

#: Points of Challenge 3.
CH3_RULES = {
    "rank_points": {1: 10, 2: 5, 3: 5, 4: 2, 5: 2},   # rank of the secret photo → points
    "short_bonus": 3,          # rank #1 with a description of at most `short_words` words
    "short_words": 5,
    "max_tries": 3,            # per level, only your best try (of the first 3) counts
    "voice_bonus_top3": 5,     # voice round: +5 if the spoken description puts the photo in the top 3
}

#: Level 3 words you may not use (singular, plural and friends).
FORBIDDEN_WORDS = ["cat", "cats", "kitten", "kittens", "kitty", "kitties", "dog", "dogs", "puppy", "puppies",
                   "pup", "pups", "doggy", "doggie", "pet", "pets", "feline", "canine"]

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
#: CLIP's text encoder reads at most 77 tokens; we keep descriptions to 40 words to stay well inside.
MAX_DESCRIPTION_WORDS = 40


def word_count(text: str) -> int:
    """Number of words (runs of letters/digits) in ``text``."""
    return len(_WORD_RE.findall(text or ""))


def forbidden_used(text: str, forbidden: Iterable[str] = FORBIDDEN_WORDS) -> list[str]:
    """Forbidden words found in ``text`` as whole words (case-insensitive), in order of appearance."""
    banned = {w.lower() for w in forbidden}
    return [w for w in (t.lower() for t in _WORD_RE.findall(text or "")) if w in banned]


def first_words(text: str, n: int = MAX_DESCRIPTION_WORDS) -> str:
    """``text`` cut after its first ``n`` words (whitespace-separated); unchanged if shorter."""
    words = str(text).split()
    return text if len(words) <= n else " ".join(words[:n])


def best_try(attempts: Sequence[Mapping]) -> dict | None:
    """The try with the most points (ties: the better rank); ``None`` if there is no try."""
    if not attempts:
        return None
    return dict(max(attempts, key=lambda a: (a["total"], -a["rank"])))


def rank_of(target: int, scores: Sequence[float]) -> int:
    """1-based rank of item ``target`` when ``scores`` are sorted from best to worst (ties: stable)."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return int(np.where(order == target)[0][0]) + 1


def pictionary_points(rank: int, n_words: int, forbidden_hits: Sequence[str] = (),
                      rules: Mapping = CH3_RULES) -> dict:
    """Points for one description: rank points + short-description bonus; a forbidden word scores 0."""
    if forbidden_hits:
        return {"points": 0, "bonus": 0, "total": 0, "foul": True}
    points = rules["rank_points"].get(int(rank), 0)
    bonus = rules["short_bonus"] if (rank == 1 and n_words <= rules["short_words"]) else 0
    return {"points": int(points), "bonus": int(bonus), "total": int(points + bonus), "foul": False}


# --------------------------------------------------------------------------------------------------
# The scorecard figure
# --------------------------------------------------------------------------------------------------

def plot_scorecard(total: float, breakdown: Mapping[str, float], thresholds: Mapping[str, int],
                   title: str, subtitle: str = "", accent: str = "#0072B2", footnote: str = ""):
    """A big, friendly scorecard: total + medal + progress meter (left), points per rule (right).

    ``breakdown`` maps each rule ("distinct objects", "Round 1 · fooled TF-IDF"…) to its points;
    one colour for all bars (they are one series), values printed at the end of each bar.
    Returns the Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyBboxPatch

    ink, muted, track = "#0E1A2E", "#5B6778", "#E3E8EF"
    medal, nxt, missing = medal_for(total, thresholds)
    gold = max(thresholds.values())

    fig = plt.figure(figsize=(12, 4.6))
    left = fig.add_axes([0.0, 0.0, 0.42, 1.0])
    left.set_xlim(0, 1); left.set_ylim(0, 1); left.axis("off")
    left.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.9, boxstyle="round,pad=0,rounding_size=0.04",
                                  fc="#F5F7FA", ec="none", transform=left.transAxes))
    left.text(0.09, 0.85, title.upper(), fontsize=11, color=muted, fontweight="bold", va="center")
    big = left.text(0.09, 0.56, f"{total:g}", fontsize=66, color=ink, fontweight="bold", va="center")
    bbox = big.get_window_extent(fig.canvas.get_renderer()).transformed(left.transAxes.inverted())
    left.text(bbox.x1 + 0.02, bbox.y0 + 0.03, "points", fontsize=15, color=muted, va="bottom")
    # medal: a coloured disc with the level written under it (never colour alone)
    ratio = (fig.get_figheight() * 1.0) / (fig.get_figwidth() * 0.42)       # keep the disc round
    left.add_patch(Ellipse((0.77, 0.58), 0.21 * ratio, 0.21, color=MEDAL_COLORS[medal]))
    left.add_patch(Ellipse((0.77, 0.58), 0.15 * ratio, 0.15, fill=False, lw=2, ec="white"))
    left.text(0.77, 0.58, medal[0], fontsize=22, color="white", fontweight="bold", ha="center", va="center")
    left.text(0.77, 0.41, medal.upper(), fontsize=12, color=ink, fontweight="bold", ha="center")
    # progress meter towards Gold, with a tick per medal
    x0, x1, y = 0.09, 0.91, 0.24
    frac = min(total / gold, 1.0) if gold else 1.0
    left.add_patch(FancyBboxPatch((x0, y - 0.018), x1 - x0, 0.036, boxstyle="round,pad=0,rounding_size=0.018",
                                  fc=track, ec="none"))
    if frac > 0:
        left.add_patch(FancyBboxPatch((x0, y - 0.018), (x1 - x0) * frac, 0.036,
                                      boxstyle="round,pad=0,rounding_size=0.018", fc=accent, ec="none"))
    for name, minimum in thresholds.items():
        xm = x0 + (x1 - x0) * minimum / gold
        left.plot([xm, xm], [y - 0.04, y + 0.04], color=ink, lw=1.2)
        left.text(xm, y - 0.075, f"{name}\n{minimum}", fontsize=8.5, color=muted, ha="center", va="top")
    status = f"{missing} more for {nxt}" if nxt else "Top level reached!"
    left.text(0.09, 0.33, status, fontsize=11, color=ink, va="center")
    if subtitle:
        left.text(0.09, 0.73, subtitle, fontsize=10, color=muted, va="center")

    right = fig.add_axes([0.66, 0.2, 0.31, 0.64])
    labels = list(breakdown.keys())[::-1]
    values = [float(breakdown[k]) for k in labels]
    right.barh(range(len(values)), values, color=accent, height=0.62)
    right.set_yticks(range(len(labels)), labels, fontsize=10.5)
    top = max(values + [1.0])
    for i, v in enumerate(values):
        right.text(v + top * 0.02, i, f"+{v:g}" if v else "0", va="center", fontsize=11, color=ink,
                   fontweight="bold")
    right.set_xlim(0, top * 1.18)
    right.set_xlabel("points")
    right.grid(axis="y", visible=False)
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", length=0)
    right.set_title("Where your points came from", fontsize=12.5)
    if footnote:
        fig.text(0.47, 0.04, footnote, fontsize=9, color=muted)
    return fig


class HuntTracker:
    """Keeps the state of a scavenger hunt, frame after frame (no model inside: easy to test).

    A class is **confirmed** the first time it is detected with confidence ≥ ``min_conf`` in
    ``min_frames`` consecutive frames of the same source (a webcam flickers: one lucky frame
    does not count). For each confirmed class we remember when (seconds since the start) and where
    (``"webcam"`` or a photo name) it was found.
    """

    def __init__(self, hunt_list: Sequence[str] = HUNT_LIST, min_conf: float = CH2_RULES["min_confidence"],
                 min_frames: int = CH2_RULES["min_frames"]):
        self.hunt_list, self.min_conf, self.min_frames = list(hunt_list), float(min_conf), int(min_frames)
        self.streak: dict[str, int] = {}
        self.confirmed: dict[str, dict] = {}          # first confirmation of each class
        self.found_in: dict[str, set[str]] = {}       # every (source, class) confirmed, for Fool-YOLO claims
        self._last_source = None

    def update(self, detections: Iterable[tuple[str, float]], t: float, source: str = "webcam") -> list[str]:
        """Feed the ``(class, confidence)`` pairs of one frame; returns the classes confirmed just now."""
        if source != self._last_source:          # a new photo: streaks start again
            self.streak, self._last_source = {}, source
        present = {cls for cls, conf in detections if conf >= self.min_conf}
        self.streak = {cls: self.streak.get(cls, 0) + 1 for cls in present}
        for cls, n in self.streak.items():
            if n >= self.min_frames:
                self.found_in.setdefault(source, set()).add(cls)
        new = sorted(c for c, n in self.streak.items() if n >= self.min_frames and c not in self.confirmed)
        for cls in new:
            self.confirmed[cls] = {"t": float(t), "source": source}
        return new

    @property
    def seen(self) -> list[str]:
        """Confirmed classes, in the order they were found."""
        return list(self.confirmed)

    @property
    def hunt_found(self) -> list[str]:
        """Hunt-list items confirmed so far, in the order they were found."""
        return [c for c in self.confirmed if c in self.hunt_list]

    @property
    def done(self) -> bool:
        """True when every item of the hunt list has been found."""
        return all(h in self.confirmed for h in self.hunt_list)

    def detections_by_source(self) -> dict[str, set[str]]:
        """``{source: classes confirmed in that source}``, ready for :func:`hunt_score`."""
        return {src: set(classes) for src, classes in self.found_in.items()}

    def table(self) -> pd.DataFrame:
        """One row per confirmed class: where and when it was found, and whether it was on the list."""
        rows = [{"object": c, "on hunt list": c in self.hunt_list, "found in": i["source"],
                 "time (s)": round(i["t"], 1)} for c, i in self.confirmed.items()]
        return pd.DataFrame(rows, columns=["object", "on hunt list", "found in", "time (s)"])


# Colours for the video overlay (OpenCV uses BGR order): Okabe–Ito orange / sky blue, navy ink.
_BGR_HUNT, _BGR_OTHER, _BGR_INK, _BGR_FOUND = (0, 159, 230), (233, 180, 86), (46, 26, 14), (115, 158, 0)


def draw_hud(frame: np.ndarray, detections: Sequence[tuple[str, float, Sequence[float]]], tracker: HuntTracker,
             fps: float | None = None, time_left: float | None = None, checklist: bool = True) -> np.ndarray:
    """Draw boxes, the hunt checklist, the timer and the FPS on a BGR frame (returns a new array).

    Hunt-list objects get orange boxes, everything else sky-blue boxes; found items are ticked
    ``[x]`` with the second they were found. ASCII text only (OpenCV fonts have no emoji).
    """
    import cv2

    img = np.ascontiguousarray(frame).copy()
    h, w = img.shape[:2]
    s = max(0.5, max(w, h) / 640)                  # text scale: same look on a 640 px webcam and a 1024 px photo
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cls, conf, box in detections:
        if conf < tracker.min_conf:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        colour = _BGR_HUNT if cls in tracker.hunt_list else _BGR_OTHER
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, max(2, int(3 * s)))
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.7 * s, max(1, int(2 * s)))
        ty = max(th + 8, y1)
        cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 8, ty), colour, -1)
        cv2.putText(img, label, (x1 + 4, ty - 5), font, 0.7 * s, _BGR_INK, max(1, int(2 * s)), cv2.LINE_AA)
    if not checklist:
        return img
    # checklist panel, semi-transparent, top-left
    line = int(24 * s)
    panel_w, panel_h = int(230 * s), line * (len(tracker.hunt_list) + 1) + int(14 * s)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (min(panel_w, w - 1), min(panel_h, h - 1)), (255, 255, 255), -1)
    img = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
    cv2.putText(img, f"HUNT {len(tracker.hunt_found)}/{len(tracker.hunt_list)}", (int(10 * s), line),
                font, 0.75 * s, _BGR_INK, max(1, int(2 * s)), cv2.LINE_AA)
    for k, item in enumerate(tracker.hunt_list, start=2):
        info = tracker.confirmed.get(item)
        text = f"[x] {item} {info['t']:.0f}s" if info else f"[ ] {item}"
        cv2.putText(img, text, (int(10 * s), line * k), font, 0.6 * s, _BGR_FOUND if info else _BGR_INK,
                    max(1, int((2 if info else 1) * s)), cv2.LINE_AA)
    corner = []
    if time_left is not None:
        corner.append(f"{max(0, int(np.ceil(time_left)))} s left")
    if fps:
        corner.append(f"{fps:.1f} FPS")
    for k, text in enumerate(corner):
        (tw, th), _ = cv2.getTextSize(text, font, 0.8 * s, max(1, int(2 * s)))
        y = int(12 * s) + (k + 1) * (th + int(16 * s))
        cv2.rectangle(img, (w - tw - int(24 * s), y - th - int(8 * s)), (w - 4, y + int(8 * s)), (255, 255, 255), -1)
        cv2.putText(img, text, (w - tw - int(14 * s), y), font, 0.8 * s, _BGR_INK, max(1, int(2 * s)), cv2.LINE_AA)
    return img


def crop_box(frame: np.ndarray, box: Sequence[float], margin: float = 0.15) -> np.ndarray:
    """Cut a box out of a frame, with a small margin around it (clipped to the image)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - dx)), int(max(0, y1 - dy))
    x2, y2 = int(min(w, x2 + dx)), int(min(h, y2 + dy))
    return frame[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)].copy()


def run_hunt(detect, frames: Iterable[tuple[np.ndarray, str]], tracker: HuntTracker,
             time_limit: float | None = None, display=None, stop_when_done: bool = True) -> dict:
    """The game loop, for a webcam or for photos: detect → update the tracker → snapshot → draw → show.

    * ``detect(frame_bgr)`` returns ``[(class, confidence, (x1, y1, x2, y2)), ...]`` (e.g. YOLO);
    * ``frames`` yields ``(frame_bgr, source)`` pairs (``source`` = ``"webcam"`` or a photo name);
    * ``display(annotated_bgr)`` shows a frame and returns ``False`` to stop (e.g. key Q).

    The frame generator is always closed at the end, even after an error or Ctrl+C: that is what
    releases the camera. Returns ``snapshots`` (``{class: RGB crop}``, taken when each class is
    confirmed), ``last_frames`` (``{source: annotated RGB frame}``), the smoothed ``fps`` and the
    number of ``frames`` processed.
    """
    import time

    snapshots, last_frames, n, fps, start = {}, {}, 0, None, None
    try:
        for frame, source in frames:
            if start is None:                      # the clock starts with the first frame (cameras take a while to open)
                start = prev = time.perf_counter()
            t = time.perf_counter() - start
            if time_limit is not None and t > time_limit:
                break
            dets = detect(frame)
            new = tracker.update([(c, p) for c, p, _ in dets], t, source)
            for cls in new:
                best = max((d for d in dets if d[0] == cls), key=lambda d: d[1])
                snapshots[cls] = crop_box(frame, best[2])[..., ::-1]
            now = time.perf_counter()
            inst = 1.0 / max(now - prev, 1e-6)
            fps = inst if fps is None else 0.85 * fps + 0.15 * inst      # smoothed: easier to read
            prev = now
            n += 1
            left = None if time_limit is None else time_limit - t
            live = time_limit is not None          # photos: boxes only (no checklist, timer or FPS)
            hud = draw_hud(frame, dets, tracker, fps=fps if live else None, time_left=left, checklist=live)
            last_frames[source] = hud[..., ::-1]
            if display is not None and display(hud) is False:
                break
            if stop_when_done and tracker.done:
                break
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
    return {"snapshots": snapshots, "last_frames": last_frames, "fps": fps, "frames": n}


def webcam_frames(index: int = 0, width: int = 640, height: int = 480):
    """Yield ``(frame_bgr, "webcam")`` from a camera; the camera is released when the generator closes.

    Raises ``RuntimeError`` if no camera can be opened (the notebook then switches to photos).
    """
    import sys

    import cv2

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY   # DirectShow opens fast on Windows
    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"No camera found at index {index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if n == 0:                               # opened but blind (busy, blocked by privacy settings…)
                    raise RuntimeError(f"The camera at index {index} opened but sent no image.")
                break
            n += 1
            yield cv2.flip(frame, 1), "webcam"          # mirror image: easier to aim objects at the camera
    finally:
        cap.release()


def photo_frames(paths: Iterable, repeats: int = CH2_RULES["min_frames"]):
    """Yield each photo ``repeats`` times as ``(frame_bgr, photo name)`` (a still photo = a video that never moves)."""
    from PIL import Image, ImageOps

    for path in paths:
        with Image.open(path) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        for _ in range(repeats):
            yield bgr, getattr(path, "name", str(path))


# --------------------------------------------------------------------------------------------------
# Figures used by the challenge notebooks
# --------------------------------------------------------------------------------------------------

#: Fill colours of the result boards: a fooled judge (you scored) vs a judge that caught you.
BOARD_COLORS = {"fooled": "#E69F00", "caught": "#E3E8EF", "info": "#F5F7FA"}


def _short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def plot_board(texts: Sequence[str], cells: Sequence[Sequence[tuple[str, str]]], col_titles: Sequence[str],
               points: Sequence[int], title: str, text_title: str = "your sentence", wrap: int = 60):
    """A results board: one row per entry, one coloured card per judge, and the points of the row.

    ``cells[i][j]`` is ``(label, state)`` with ``state`` in :data:`BOARD_COLORS` (``"fooled"`` cards are
    orange and say so in words; ``"caught"`` cards are light grey).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n, k = len(texts), len(col_titles)
    fig, ax = plt.subplots(figsize=(12, 0.62 * n + 1.3))
    ax.set_xlim(0, 1); ax.set_ylim(n, -0.9); ax.axis("off")
    x_text, x_cols, x_pts = 0.0, 0.5, 0.96
    col_w = (x_pts - x_cols - 0.02) / k
    ax.text(x_text, -0.5, text_title.upper(), fontsize=9, color="#5B6778", fontweight="bold", va="center")
    for j, t in enumerate(col_titles):
        ax.text(x_cols + col_w * (j + 0.5), -0.5, t.upper(), fontsize=9, color="#5B6778", fontweight="bold",
                ha="center", va="center")
    ax.text(x_pts, -0.5, "PTS", fontsize=9, color="#5B6778", fontweight="bold", ha="center", va="center")
    for i, (text, row, pts) in enumerate(zip(texts, cells, points)):
        ax.text(x_text, i + 0.5, _short(text, wrap), fontsize=10, color="#0E1A2E", va="center")
        for j, (label, state) in enumerate(row):
            x0 = x_cols + col_w * j + 0.006
            ax.add_patch(FancyBboxPatch((x0, i + 0.14), col_w - 0.012, 0.72,
                                        boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=20,
                                        fc=BOARD_COLORS[state], ec="none"))
            ax.text(x0 + (col_w - 0.012) / 2, i + 0.5, label, fontsize=9.5, ha="center", va="center",
                    color="#0E1A2E", fontweight="bold" if state == "fooled" else "normal")
        ax.text(x_pts, i + 0.5, f"+{pts}" if pts else "0", fontsize=12, ha="center", va="center",
                color="#0E1A2E", fontweight="bold")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=4)
    fig.tight_layout()
    return fig


def plot_term_pushes(tables: Sequence[pd.DataFrame], titles: Sequence[str], toward: str, away: str,
                     colors: tuple[str, str], top: int = 6, ncols: int = 3, suptitle: str = ""):
    """Small multiples: for each text, the ``top`` terms that pushed a linear model the most.

    Each table comes from :func:`term_contributions`; bars to the right push towards ``toward``
    (first colour), bars to the left towards ``away`` (second colour). Same x-scale everywhere.
    """
    import matplotlib.pyplot as plt

    n = len(tables)
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.5 * nrows + 0.6), squeeze=False)
    lim = max([t["contribution"].abs().max() for t in tables if len(t)] + [0.1]) * 1.15
    for ax, table, title in zip(axes.ravel(), tables, titles):
        part = table.head(top).iloc[::-1]
        ax.barh(range(len(part)), part["contribution"], height=0.62,
                color=[colors[0] if v > 0 else colors[1] for v in part["contribution"]])
        ax.set_yticks(range(len(part)), part["term"], fontsize=9.5)
        ax.axvline(0, color="#0E1A2E", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_title(_short(title, 40), fontsize=10, fontweight="normal")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        if len(part) == 0:
            ax.text(0, 0.5, "no known word", ha="center", va="center", transform=ax.transAxes, color="#5B6778")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(f"← pushes to {away}    ·    pushes to {toward} →", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def thumbnail(path, size: int = 220):
    """A square RGB thumbnail (centre crop) of an image file, as a NumPy array."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(ImageOps.fit(im, (size, size)))


def plot_numbered_grid(paths: Sequence, ncols: int = 9, highlight: int | None = None, size: float = 1.3,
                       title: str = ""):
    """All photos of a pool in a grid, each with its number (1, 2, 3…); ``highlight`` = 0-based index to frame."""
    import matplotlib.pyplot as plt

    n = len(paths)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.12), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k >= n:
            ax.axis("off")
            continue
        ax.imshow(thumbnail(paths[k], 160))
        ax.set_title(f"#{k + 1}", fontsize=9, loc="center", fontweight="normal", pad=2)
        if highlight == k:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#E69F00"); sp.set_linewidth(4)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_tries(pool: Sequence, target: int, tries: Sequence[dict], top: int = 5, title: str = ""):
    """One row per description: CLIP's top-``top`` photos, the secret photo framed in orange.

    Each try is a dict with ``text``, ``scores`` (one cosine similarity per pool photo), ``rank``,
    ``total`` and ``foul``. If the secret photo is not in the top ``top``, it is added at the end of the row
    with its rank, so you always see where it landed.
    """
    import matplotlib.pyplot as plt

    n = max(len(tries), 1)
    fig, axes = plt.subplots(n, top + 2, figsize=(12, 1.95 * n + 0.5), squeeze=False,
                             gridspec_kw={"width_ratios": [2.4] + [1] * (top + 1)})
    for row, attempt in zip(axes, tries):
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        info = row[0]
        info.axis("off")
        verdict = ("FOUL: forbidden word" if attempt["foul"] else
                   f"rank #{attempt['rank']}  ·  +{attempt['total']} pts")
        info.text(0, 0.62, "\n".join(_wrap(f"“{attempt['text']}”", 30)[:3]), fontsize=9.5, va="center",
                  color="#0E1A2E", transform=info.transAxes)
        info.text(0, 0.12, verdict, fontsize=10.5, fontweight="bold", va="center", transform=info.transAxes,
                  color="#A8612B" if attempt["foul"] else "#0E1A2E")
        order = np.argsort(-np.asarray(attempt["scores"]), kind="stable")
        shown = list(order[:top])
        extra = target not in shown
        for c, idx in enumerate(shown + ([target] if extra else [])):
            ax = row[1 + c]
            ax.imshow(thumbnail(pool[idx], 150))
            ax.set_title(f"#{idx + 1}" + ("" if c < top else f" · rank {attempt['rank']}"), fontsize=8.5,
                         loc="center", fontweight="normal", pad=2)
            if idx == target:
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color("#E69F00"); sp.set_linewidth(4)
        if extra:
            row[top + 1].set_xlabel("secret photo", fontsize=8.5)
        for ax in row[1 + len(shown) + (1 if extra else 0):]:
            if not ax.images:
                ax.axis("off")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]

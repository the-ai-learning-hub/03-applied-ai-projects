"""Scoring rules, figures and interactive helpers for the three Lesson 03 class challenges.

* Challenge 1 · Flip It!                 (``01_text_mining/03_challenge_flip_it``)
* Challenge 2 · YOLO Live                 (``02_computer_vision/03_challenge_yolo``)
* Challenge 3 · CLIP Pictionary           (``03_multimodal/02_challenge_clip_pictionary``)

Everything here is plain Python + NumPy/pandas/Matplotlib: no model is loaded, so the tests run in
a second. The notebooks call the models and pass the *answers* (labels, probabilities, detected
classes, ranks) to these functions. Challenge 1 also gets its interactive game here
(:class:`FlipGame`, ``ipywidgets`` + HTML), and Challenge 2 its drawing, tracking, video and webcam
machinery (OpenCV), so the notebooks stay short. Every rule lives in one
constant, so the notebook, the README and the slides can quote the exact same numbers.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.text import clean_tweet

# --------------------------------------------------------------------------------------------------
# Levels (medals)
# --------------------------------------------------------------------------------------------------

#: Minimum points for each medal, per challenge (below Bronze you are a "Rookie").
LEVELS = {
    "flip_it": {"Bronze": 5, "Silver": 9, "Gold": 13},          # stars (3 per level, 5 levels)
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
# Challenge 1 · Flip It!
# --------------------------------------------------------------------------------------------------

#: The five levels of Challenge 1. ``start``: the sentence you get; ``goal``: the verdict you must reach;
#: ``gold`` / ``silver``: most word swaps for that medal (any flip is at least Bronze);
#: ``rule``: ``None``, ``"liar"`` (a human must still read the *start* sentiment) or ``"add_only"``.
FLIP_LEVELS = [
    {"name": "Warm-up", "emoji": "🌞", "start": "I love this sunny day", "goal": "neg",
     "gold": 1, "silver": 2, "rule": None,
     "mission": "Make the model say NEGATIVE."},
    {"name": "Rescue mission", "emoji": "🚑", "start": "this is the worst headache ever", "goal": "pos",
     "gold": 1, "silver": 3, "rule": None,
     "mission": "Make the model say POSITIVE. One swap is possible, but the word is not the one you think."},
    {"name": "Liar mode", "emoji": "🎭", "start": "this new phone is awesome", "goal": "neg",
     "gold": 1, "silver": 3, "rule": "liar",
     "mission": "Make the model say NEGATIVE while a human still reads a HAPPY sentence."},
    {"name": "Add-only", "emoji": "➕", "start": "I lost my keys and missed the bus", "goal": "pos",
     "gold": 3, "silver": 5, "rule": "add_only",
     "mission": "Make the model say POSITIVE. You may only ADD words: the bad news cannot be deleted."},
    {"name": "Boss · 99.98 % sure", "emoji": "💀", "start":
     "happy birthday! thank you so much, love you all, have a great day! haha", "goal": "neg",
     "gold": 3, "silver": 5, "rule": None,
     "mission": "The model has never been so sure. Make it say NEGATIVE."},
]

#: Liar mode: words too obviously negative to add (a human would not read the sentence as happy any more).
LIAR_BANNED = frozenset({
    "sad", "hate", "hates", "hated", "sucks", "suck", "worst", "awful", "terrible", "horrible", "depressing",
    "depressed", "died", "dead", "die", "ugh", "cry", "crying", "angry", "upset", "boring", "bored", "annoyed",
    "disappointed", "disappointing", "lame", "unfortunately", "sadly", "bummer", "pain", "hurts", "hurt",
    "broken", "broke", "fail", "failed", "rip", "shitty", "crap", "useless", "ugly", "stupid", "waste",
    "wrong", "sorry", "headache", "unhappy", "miserable", "worse", "bad",
})

#: Stars per medal in a level, and the colours of the game (the slide deck's dark palette).
FLIP_STARS = {"Gold": 3, "Silver": 2, "Bronze": 1}
FLIP_THEME = {"page": "#08111f", "slide": "#0e1a2e", "card": "#13223b", "card2": "#172a48",
              "line": "#1d3152", "line2": "#2a4570", "ink": "#f1f5fb", "body": "#b4c2d8", "muted": "#94a6c2",
              "pos": "#9bd67e", "neg": "#ff7a7a", "yellow": "#ffd43b", "violet": "#c4a2ff", "blue": "#6aaef0"}
_VERDICT = {"pos": "POSITIVE", "neg": "NEGATIVE"}


def word_edits(a: Sequence[str], b: Sequence[str]) -> int:
    """Word swaps between two token lists: each word added, removed or replaced counts 1 (Levenshtein).

    >>> word_edits(["i", "love", "it"], ["i", "hate", "it", "!"])
    2
    """
    previous = list(range(len(b) + 1))
    for i, word_a in enumerate(a, 1):
        current = [i]
        for j, word_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word_a != word_b)))
        previous = current
    return previous[-1]


def is_add_only(start: Sequence[str], tokens: Sequence[str]) -> bool:
    """``True`` if ``tokens`` keeps every word of ``start``, in the same order (words were only added)."""
    remaining = iter(tokens)
    return all(word in remaining for word in start)


def new_word_positions(start: Sequence[str], tokens: Sequence[str]) -> set[int]:
    """Positions in ``tokens`` of the words that are not in ``start`` (added or replaced)."""
    from difflib import SequenceMatcher

    kept = set()
    for block in SequenceMatcher(None, list(start), list(tokens), autojunk=False).get_matching_blocks():
        kept.update(range(block.b, block.b + block.size))
    return set(range(len(tokens))) - kept


def word_pushes(text: str, vectorizer, model, clean=clean_tweet, pos_label: str = "pos") -> dict:
    """Read ``text`` like the model does and say how hard each word pushes towards ``pos_label``.

    Works for a word-level (unigram) ``TfidfVectorizer`` + a binary linear model (``LogisticRegression``):
    the model's score is ``bias + Σ tfidf(word) × weight(word)``, so each word's push is exact.
    A word that appears twice shares its push between its two copies; an unknown word pushes 0.
    Returns ``tokens``, ``pushes`` (NumPy array), ``known`` (bool per token), ``bias``, ``logit`` and
    ``p_pos`` (probability of ``pos_label``).
    """
    if getattr(vectorizer, "ngram_range", (1, 1)) != (1, 1):
        raise ValueError("word_pushes needs a word-level vectorizer (ngram_range=(1, 1))")
    cleaned = clean(text) if clean else str(text)
    tokens = list(vectorizer.build_analyzer()(cleaned))
    classes = list(model.classes_)
    sign = 1.0 if pos_label == classes[1] else -1.0
    weights = sign * np.ravel(model.coef_)
    bias = float(sign * np.ravel(model.intercept_)[0])
    row = vectorizer.transform([cleaned])
    value = dict(zip(row.indices, row.data))
    vocab = vectorizer.vocabulary_
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    pushes, known = [], []
    for t in tokens:
        f = vocab.get(t)
        known.append(f is not None)
        pushes.append(0.0 if f is None else value.get(f, 0.0) * weights[f] / counts[t])
    pushes = np.array(pushes, dtype=float)
    logit = bias + float(pushes.sum())
    return {"tokens": tokens, "pushes": pushes, "known": known, "bias": bias, "logit": logit,
            "p_pos": float(1.0 / (1.0 + np.exp(-logit)))}


def flip_medal(swaps: int, level: Mapping) -> tuple[str, int]:
    """Medal and stars for a flip in ``swaps`` word swaps (Gold ≤ ``level['gold']``, Silver ≤ ``level['silver']``)."""
    medal = "Gold" if swaps <= level["gold"] else "Silver" if swaps <= level["silver"] else "Bronze"
    return medal, FLIP_STARS[medal]


def flip_result(level: Mapping, start: Sequence[str], tokens: Sequence[str], p_pos: float,
                honest: bool = False, banned: Iterable[str] = LIAR_BANNED) -> dict:
    """Judge one attempt of a level: did the verdict flip, how many swaps, is it within the rules?

    ``start`` / ``tokens``: the words of the start sentence and of the attempt (as the model reads them).
    ``honest``: Liar mode only, the player's word that a human still reads the start sentiment.
    Returns ``verdict``, ``flipped``, ``swaps``, ``problems`` (list of strings), ``valid``, ``medal``, ``stars``.
    """
    verdict = "pos" if p_pos >= 0.5 else "neg"
    swaps = word_edits(start, tokens)
    problems = []
    if level.get("rule") == "add_only" and not is_add_only(start, tokens):
        problems.append("Add-only: every original word must stay, in the same order.")
    if level.get("rule") == "liar":
        added = set(tokens) - set(start)
        too_obvious = sorted(added & set(banned))
        if too_obvious:
            problems.append(f"Liar mode: {', '.join(too_obvious)} would make a human read it as negative.")
        elif verdict == level["goal"] and not honest:
            problems.append("Liar mode: tick the honour box if a human still reads it as happy.")
    flipped = verdict == level["goal"]
    valid = flipped and not problems
    medal, stars = flip_medal(swaps, level) if valid else (None, 0)
    return {"verdict": verdict, "flipped": flipped, "swaps": swaps, "problems": problems, "valid": valid,
            "medal": medal, "stars": stars}


def flip_summary(best: Mapping[int, Mapping], levels: Sequence[Mapping] = FLIP_LEVELS) -> dict:
    """Total of a game: ``best`` maps a level number (1, 2, …) to its best valid result (``swaps``, ``stars``).

    Returns ``stars``, ``max_stars``, ``flips``, ``swaps``, ``medal`` (from :data:`LEVELS` ``["flip_it"]``)
    and a one-line ``share`` text.
    """
    stars = sum(int(r["stars"]) for r in best.values())
    swaps = sum(int(r["swaps"]) for r in best.values())
    medal, nxt, missing = medal_for(stars, LEVELS["flip_it"])
    max_stars = FLIP_STARS["Gold"] * len(levels)
    share = (f"🎚️ I flipped the AI {len(best)}/{len(levels)} times in {swaps} word swaps · "
             f"{stars}/{max_stars} ⭐ · {MEDAL_EMOJI[medal]} {medal} · #FlipIt")
    return {"stars": stars, "max_stars": max_stars, "flips": len(best), "swaps": swaps, "medal": medal,
            "next": nxt, "missing": missing, "share": share}


# ---- Flip It! · HTML pieces (inline styles, so they also look right where <style> tags are removed) ----

_FLIP_CSS = """<style>
@keyframes flipit-pop{0%{transform:scale(.4);opacity:0}60%{transform:scale(1.08);opacity:1}100%{transform:scale(1)}}
@keyframes flipit-burst{0%{transform:translate(-50%,-50%) scale(.3) rotate(0deg);opacity:1}
 100%{transform:translate(calc(-50% + var(--dx)),calc(-50% + var(--dy))) scale(1.25) rotate(var(--r));opacity:0}}
@keyframes flipit-glow{0%,100%{box-shadow:0 0 0 0 rgba(255,212,59,.0)}50%{box-shadow:0 0 28px 4px rgba(255,212,59,.45)}}
.flipit-app{background:#08111f;border:1px solid #1d3152;border-radius:18px;padding:14px 16px 16px;max-width:980px}
.flipit-app input[type="text"]{background:#0b1526!important;color:#f1f5fb!important;font-size:20px!important;
 height:48px!important;border:2px solid #2a4570!important;border-radius:12px!important;padding:0 14px!important;
 font-family:"Inter Tight",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif!important}
.flipit-app input[type="text"]:focus{border-color:#c4a2ff!important;outline:none!important;box-shadow:0 0 0 4px rgba(196,162,255,.18)!important}
.flipit-app .widget-toggle-button,.flipit-app .jupyter-button{border-radius:10px!important;font-weight:700!important}
.flipit-app .widget-checkbox label,.flipit-app .widget-checkbox .widget-label-basic span{color:#f1f5fb!important;font-size:15px!important}
.flipit-app .widget-html,.flipit-app .widget-html-content{color:#f1f5fb}
</style>"""

_FONT = "font-family:'Inter Tight',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif"   # single quotes:
_MONO = "font-family:'JetBrains Mono',ui-monospace,Consolas,'SF Mono',monospace"           # inside style="…"


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _confidence(p_pos: float) -> tuple[str, str]:
    """('94 %', 'POSITIVE'): the confidence in the current verdict, with 2 decimals above 99 %."""
    verdict = "pos" if p_pos >= 0.5 else "neg"
    conf = p_pos if verdict == "pos" else 1 - p_pos
    text = f"{conf * 100:.2f} %" if conf >= 0.99 else f"{conf * 100:.0f} %"
    return text, _VERDICT[verdict]


def render_word_chips(tokens: Sequence[str], pushes: Sequence[float], known: Sequence[bool],
                      new: Iterable[int] = ()) -> str:
    """The sentence as the model reads it: one chip per word, green = pushes to POSITIVE, red = NEGATIVE.

    The number under each word is its push; a dashed chip is a word the model never saw (push 0);
    a yellow underline marks a word you added or changed.
    """
    th, new = FLIP_THEME, set(new)
    scale = max(1.0, max((abs(p) for p in pushes), default=0.0))
    chips = []
    for i, (tok, push, ok) in enumerate(zip(tokens, pushes, known)):
        strength = min(1.0, abs(push) / scale)
        if not ok:
            bg, border, ink, label = "transparent", f"1.5px dashed {th['muted']}", th["muted"], "unknown"
        else:
            col = th["pos"] if push > 0 else th["neg"]
            bg = _rgba(col, 0.10 + 0.78 * strength) if abs(push) > 1e-9 else _rgba(th["muted"], 0.12)
            border = f"1.5px solid {_rgba(col, 0.35 + 0.65 * strength)}"
            ink = th["page"] if strength > 0.55 else th["ink"]
            label = f"{push:+.2f}"
        mark = f"box-shadow:inset 0 -4px 0 {th['yellow']};" if i in new else ""
        chips.append(
            f'<span title="push {push:+.3f}" style="display:inline-flex;flex-direction:column;align-items:center;'
            f'margin:4px 3px;padding:6px 11px 5px;border-radius:11px;background:{bg};border:{border};{mark}'
            f'min-width:26px;vertical-align:top"><span style="{_FONT};font-weight:700;font-size:19px;'
            f'line-height:1.25;color:{ink}">{html.escape(tok)}</span><span style="{_MONO};font-size:11px;'
            f'color:{ink};opacity:.85">{label}</span></span>')
    if not chips:
        chips.append(f'<span style="color:{th["muted"]};{_FONT};font-size:16px">(no words: the model only has its bias)</span>')
    return "".join(chips)


def render_gauge(p_pos: float, start_p: float | None = None, uid: str = "0") -> str:
    """The big NEG ◀━━●━━▶ POS thermometer; the knob slides from ``start_p`` to ``p_pos``."""
    th = FLIP_THEME
    col = th["pos"] if p_pos >= 0.5 else th["neg"]
    conf, word = _confidence(p_pos)
    left = f"{min(98.5, max(1.5, p_pos * 100)):.2f}%"    # keep the knob inside the bar
    anim = ""
    if start_p is not None:
        anim = f"<style>@keyframes flipit-knob-{uid}{{from{{left:{min(98.5, max(1.5, start_p * 100)):.2f}%}}}}</style>"
    return (
        f'{anim}<div style="{_FONT};margin:6px 0 2px">'
        f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px">'
        f'<span style="font-size:40px;font-weight:800;color:{col};line-height:1">{conf}</span>'
        f'<span style="font-size:20px;font-weight:800;letter-spacing:.12em;color:{col}">{word}</span>'
        f'<span style="margin-left:auto;{_MONO};font-size:13px;color:{th["muted"]}">P(positive) = {p_pos:.4f}</span></div>'
        f'<div style="position:relative;height:20px;border-radius:10px;'
        f'background:linear-gradient(90deg,{th["neg"]} 0%,{_rgba(th["neg"], .25)} 46%,{_rgba(th["pos"], .25)} 54%,{th["pos"]} 100%)">'
        f'<div style="position:absolute;left:50%;top:-7px;bottom:-7px;width:2px;margin-left:-1px;background:{th["ink"]};opacity:.55"></div>'
        f'<div style="position:absolute;left:{left};top:50%;width:30px;height:30px;margin:-15px 0 0 -15px;'
        f'border-radius:50%;background:{th["ink"]};border:5px solid {col};box-shadow:0 0 0 6px {_rgba(col, .25)},0 0 22px {col};'
        f'animation:flipit-knob-{uid} .55s cubic-bezier(.3,1.45,.5,1)"></div></div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:9px;{_MONO};font-size:12px;'
        f'font-weight:700;letter-spacing:.14em"><span style="color:{th["neg"]}">◀ NEGATIVE</span>'
        f'<span style="color:{th["muted"]}">flip line 50 %</span><span style="color:{th["pos"]}">POSITIVE ▶</span></div></div>')


def render_level_intro(number: int, level: Mapping, start_p: float, total: int = len(FLIP_LEVELS)) -> str:
    """The header card of a level: name, start sentence, the model's verdict, the mission and the medals."""
    th = FLIP_THEME
    conf, word = _confidence(start_p)
    goal_col = th["pos"] if level["goal"] == "pos" else th["neg"]
    extra = ""
    if level.get("rule") == "liar":
        extra = (f'<div style="margin-top:8px;font-size:13px;color:{th["muted"]}">🚫 too obvious, not allowed: '
                 f'{", ".join(sorted(LIAR_BANNED)[:18])}…</div>')
    return (
        f'<div style="{_FONT};background:linear-gradient(135deg,{th["slide"]} 0%,{th["card2"]} 100%);'
        f'border:1px solid {th["line"]};border-left:6px solid {th["violet"]};border-radius:14px;padding:14px 18px;color:{th["ink"]}">'
        f'<div style="{_MONO};font-size:12px;font-weight:700;letter-spacing:.16em;color:{th["violet"]}">'
        f'LEVEL {number}/{total} · {html.escape(level["name"].upper())}</div>'
        f'<div style="font-size:24px;font-weight:800;margin:6px 0 4px">{level["emoji"]} “{html.escape(level["start"])}”</div>'
        f'<div style="font-size:15px;color:{th["body"]}">The model says <b style="color:{th["ink"]}">{conf} {word}</b>. '
        f'<b style="color:{goal_col}">{html.escape(level["mission"])}</b></div>'
        f'<div style="margin-top:8px;font-size:14px;color:{th["body"]}">🥇 ≤ {level["gold"]} swap{"s" * (level["gold"] > 1)} · '
        f'🥈 ≤ {level["silver"]} swaps · 🥉 any flip · every word added, removed or replaced = 1 swap</div>{extra}</div>')


def render_status(result: Mapping, level: Mapping) -> str:
    """One line under the gauge: the swap counter, the medal you would get, the rule problems."""
    th = FLIP_THEME
    swaps = result["swaps"]
    pill = (f'<span style="display:inline-block;padding:5px 14px;border-radius:999px;background:{th["card2"]};'
            f'border:1px solid {th["line2"]};font-weight:800;font-size:18px">🔁 {swaps} swap{"" if swaps == 1 else "s"}</span>')
    if result["valid"]:
        msg = (f'<b style="color:{th["yellow"]}">FLIPPED! {MEDAL_EMOJI[result["medal"]]} {result["medal"]} · '
               f'{"⭐" * result["stars"]}</b>')
    elif result["flipped"]:
        msg = f'<b style="color:{th["yellow"]}">Flipped… but not by the rules.</b>'
    else:
        medal, _ = flip_medal(max(swaps, 1), level)
        msg = (f'<span style="color:{th["body"]}">Not flipped yet: push it to <b>{_VERDICT[level["goal"]]}</b>. '
               f'(Flip now = {MEDAL_EMOJI[medal]})</span>' if swaps else
               f'<span style="color:{th["body"]}">Edit the sentence above: every keystroke is scored live.</span>')
    problems = "".join(f'<div style="margin-top:6px;color:{th["neg"]};font-weight:600">⚠️ {html.escape(p)}</div>'
                       for p in result["problems"])
    return (f'<div style="{_FONT};margin-top:12px;font-size:16px;display:flex;align-items:center;gap:14px;'
            f'flex-wrap:wrap">{pill}{msg}</div>{problems}')


def render_celebration(result: Mapping, seed: int = 0) -> str:
    """A confetti burst (pure CSS, no JavaScript) and the medal, shown when a level is won."""
    if not result.get("valid"):
        return ""
    th = FLIP_THEME
    rng = np.random.default_rng(seed)
    bits = []
    for k in range(34):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(120, 360)
        dx, dy = dist * np.cos(angle), dist * np.sin(angle) * 0.55 - 30
        emoji = ["🎉", "✨", "🎊", "⭐", "🎚️", "💥", "🥳", "🌟"][k % 8]
        bits.append(f'<span style="position:absolute;left:50%;top:50%;font-size:{rng.uniform(18, 34):.0f}px;'
                    f'--dx:{dx:.0f}px;--dy:{dy:.0f}px;--r:{rng.uniform(-540, 540):.0f}deg;pointer-events:none;'
                    f'animation:flipit-burst {rng.uniform(1.0, 1.9):.2f}s cubic-bezier(.15,.7,.3,1) '
                    f'{rng.uniform(0, .25):.2f}s both">{emoji}</span>')
    swaps = result["swaps"]
    return (
        f'<div style="position:relative;overflow:visible;margin:14px 0 4px;text-align:center;{_FONT};'
        f'padding:16px 12px;border-radius:16px;color:{th["page"]};'
        f'background:linear-gradient(110deg,{th["yellow"]} 0%,#ffe680 45%,{th["violet"]} 100%);'
        f'animation:flipit-pop .55s cubic-bezier(.3,1.5,.5,1) both,flipit-glow 1.6s ease-in-out .4s 2">'
        f'{"".join(bits)}<div style="font-size:30px;font-weight:900;letter-spacing:.02em">'
        f'🎉 FLIPPED in {swaps} swap{"" if swaps == 1 else "s"}! {MEDAL_EMOJI[result["medal"]]}</div>'
        f'<div style="font-size:17px;font-weight:700;margin-top:4px">{result["medal"]} · {"⭐" * result["stars"]}'
        f'{"☆" * (FLIP_STARS["Gold"] - result["stars"])} · the AI now says {_VERDICT[result["verdict"]]}</div></div>')


def render_scoreboard(best: Mapping[int, Mapping], levels: Sequence[Mapping] = FLIP_LEVELS) -> str:
    """The final scorecard: stars per level, best sentence, total medal and a one-line result to share."""
    th = FLIP_THEME
    s = flip_summary(best, levels)
    rows = []
    for n, level in enumerate(levels, 1):
        r = best.get(n)
        stars = (f'<span style="color:{th["yellow"]}">{"★" * r["stars"]}</span>'
                 f'<span style="color:{th["line2"]}">{"★" * (3 - r["stars"])}</span>') if r else \
            f'<span style="color:{th["line2"]}">★★★</span>'
        what = (f'{r["swaps"]} swap{"" if r["swaps"] == 1 else "s"} · <i>“{html.escape(r["text"])}”</i>' if r
                else f'<span style="color:{th["muted"]}">not flipped yet</span>')
        rows.append(f'<div style="display:flex;gap:14px;align-items:center;padding:8px 0;border-top:1px solid {th["line"]}">'
                    f'<span style="width:230px;font-weight:700">{level["emoji"]} {n} · {html.escape(level["name"])}</span>'
                    f'<span style="width:62px;font-size:18px;letter-spacing:1px">{stars}</span>'
                    f'<span style="flex:1;color:{th["body"]}">{what}</span></div>')
    hint = f' · {s["missing"]} more ⭐ for {s["next"]}' if s["next"] else " · top level!"
    return (
        f'<div style="{_FONT};background:{th["slide"]};border:1px solid {th["line"]};border-radius:16px;'
        f'padding:16px 18px;color:{th["ink"]}">'
        f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">'
        f'<span style="font-size:46px">{MEDAL_EMOJI[s["medal"]]}</span><div>'
        f'<div style="{_MONO};font-size:12px;font-weight:700;letter-spacing:.16em;color:{th["violet"]}">FLIP IT! · SCORECARD</div>'
        f'<div style="font-size:26px;font-weight:800">{s["stars"]} / {s["max_stars"]} ⭐ · {s["medal"]}</div>'
        f'<div style="color:{th["body"]};font-size:14px">{s["flips"]}/{len(levels)} levels flipped · '
        f'{s["swaps"]} swaps in total{hint} · medals: 🥉 {LEVELS["flip_it"]["Bronze"]} ⭐ · '
        f'🥈 {LEVELS["flip_it"]["Silver"]} ⭐ · 🥇 {LEVELS["flip_it"]["Gold"]} ⭐</div></div></div>'
        f'{"".join(rows)}'
        f'<div style="margin-top:12px;padding:10px 14px;border-radius:10px;background:{th["page"]};'
        f'border:1px dashed {th["line2"]};{_MONO};font-size:14px;color:{th["yellow"]};user-select:all">'
        f'{html.escape(s["share"])}</div></div>')


class FlipGame:
    """Challenge 1 · Flip It! Change as few words as possible to flip a sentiment model's verdict.

    ``FlipGame(vectorizer, model)`` takes a fitted word-level ``TfidfVectorizer`` and a binary linear model.
    * :meth:`play` returns the interactive game (``ipywidgets``: type, and everything updates live);
    * :meth:`flip` is the no-widget version: ``game.flip("my sentence", level=1)`` shows the same card;
    * :meth:`scorecard` shows your stars, medal and a line to share.
    The best valid result of each level is kept in ``game.best`` (fewest swaps wins).
    """

    def __init__(self, vectorizer, model, levels: Sequence[Mapping] = FLIP_LEVELS, clean=clean_tweet,
                 pos_label: str = "pos"):
        self.vectorizer, self.model, self.levels = vectorizer, model, list(levels)
        self.clean, self.pos_label = clean, pos_label
        self.best: dict[int, dict] = {}
        self._starts = [self.read(lv["start"]) for lv in self.levels]
        self._ui: dict = {}
        self._count = 0

    def read(self, text: str) -> dict:
        """The model's reading of ``text``: tokens, push of each word, bias, logit and P(positive)."""
        return word_pushes(text, self.vectorizer, self.model, clean=self.clean, pos_label=self.pos_label)

    def check(self, text: str, level: int, honest: bool = False, record: bool = True) -> tuple[dict, dict]:
        """Read and judge ``text`` for ``level`` (1-based); keep it in :attr:`best` if it is a better valid flip."""
        lv, start = self.levels[level - 1], self._starts[level - 1]
        reading = self.read(text)
        result = flip_result(lv, start["tokens"], reading["tokens"], reading["p_pos"], honest=honest)
        if record and result["valid"]:
            old = self.best.get(level)
            if old is None or result["swaps"] < old["swaps"]:
                self.best[level] = {"text": " ".join(str(text).split()), "swaps": result["swaps"],
                                    "stars": result["stars"], "medal": result["medal"]}
        return reading, result

    def card(self, text: str, level: int | None = None, honest: bool = False, start_p: float | None = None,
             celebrate: bool = True, record: bool = True) -> str:
        """HTML of one attempt: words coloured by push, the gauge, the swap counter and the medal."""
        self._count += 1
        uid = str(self._count)
        th = FLIP_THEME
        if level is None:  # just look at how the model reads a sentence
            reading = self.read(text)
            return (f'{_FLIP_CSS}<div style="{_FONT};background:{th["slide"]};border:1px solid {th["line"]};'
                    f'border-radius:16px;padding:14px 18px;color:{th["ink"]}">'
                    f'{render_word_chips(reading["tokens"], reading["pushes"], reading["known"])}'
                    f'{render_gauge(reading["p_pos"], 0.5, uid)}</div>')
        start = self._starts[level - 1]
        reading, result = self.check(text, level, honest=honest, record=record)
        new = new_word_positions(start["tokens"], reading["tokens"])
        party = render_celebration(result, seed=self._count) if celebrate else ""
        return (f'{_FLIP_CSS}<div style="{_FONT};background:{th["slide"]};border:1px solid {th["line"]};'
                f'border-radius:16px;padding:14px 18px;color:{th["ink"]}">'
                f'<div style="{_MONO};font-size:12px;font-weight:700;letter-spacing:.16em;color:{th["muted"]};'
                f'margin-bottom:4px">THE MODEL READS · {self.levels[level - 1]["emoji"]} LEVEL {level}</div>'
                f'{render_word_chips(reading["tokens"], reading["pushes"], reading["known"], new)}'
                f'{render_gauge(reading["p_pos"], start["p_pos"] if start_p is None else start_p, uid)}'
                f'{render_status(result, self.levels[level - 1])}{party}</div>')

    def flip(self, text: str, level: int | None = None, honest: bool = False):
        """No-widget version of the game: show (and score, if ``level`` is given) one attempt as HTML.

        ``honest=True`` is your word of honour for Liar mode (a human still reads the sentence as happy).
        """
        from IPython.display import HTML

        return HTML(self.card(text, level, honest=honest))

    def intro(self, level: int):
        """HTML card that presents a level (for the no-widget version)."""
        from IPython.display import HTML

        return HTML(_FLIP_CSS + render_level_intro(level, self.levels[level - 1], self._starts[level - 1]["p_pos"],
                                                   len(self.levels)))

    def scorecard(self):
        """Your stars, medal and a one-line result to share, as HTML."""
        from IPython.display import HTML

        return HTML(_FLIP_CSS + render_scoreboard(self.best, self.levels))

    def word_board(self, words: Sequence[str]):
        """HTML chips with the weight of each word in ``words`` (green = positive, red = negative)."""
        from IPython.display import HTML

        vocab, coef = self.vectorizer.vocabulary_, np.ravel(self.model.coef_)
        sign = 1.0 if self.pos_label == list(self.model.classes_)[1] else -1.0
        weights = [sign * coef[vocab[w]] if w in vocab else 0.0 for w in words]
        return HTML(_FLIP_CSS + f'<div style="background:{FLIP_THEME["slide"]};border-radius:14px;padding:10px 12px">'
                    + render_word_chips(list(words), weights, [w in vocab for w in words]) + "</div>")

    # ---- the interactive version -----------------------------------------------------------------
    def play(self, level: int = 1):
        """The interactive game (``ipywidgets``): pick a level, type, watch the gauge move, flip it!"""
        import ipywidgets as w

        th = FLIP_THEME
        title = w.HTML(
            f'{_FLIP_CSS}<div style="{_FONT};display:flex;align-items:center;gap:14px;margin-bottom:6px">'
            f'<span style="font-size:40px">🎚️</span><div><div style="font-size:30px;font-weight:900;'
            f'color:{th["ink"]};line-height:1.1">Flip <span style="color:{th["yellow"]}">It!</span></div>'
            f'<div style="color:{th["body"]};font-size:14px">Change as few words as possible to flip the AI\'s verdict.'
            f'</div></div></div>')
        picker = w.ToggleButtons(options=[(f'{n} {lv["emoji"]}', n) for n, lv in enumerate(self.levels, 1)],
                                 value=level, style={"button_width": "74px"})
        intro, live, party, board = w.HTML(), w.HTML(), w.HTML(), w.HTML()
        text = w.Text(continuous_update=True, placeholder="type your version of the sentence…",
                      layout=w.Layout(width="100%"))
        honest = w.Checkbox(value=False, indent=False,
                            description="✋ On my honour: a human still reads this sentence as HAPPY")
        reset = w.Button(description="↺ Reset", style={"button_color": th["card2"], "text_color": th["ink"]})
        nxt = w.Button(description="Next level ▶", style={"button_color": th["violet"], "text_color": th["page"]})
        state = {"p": None, "won": False, "busy": False}

        def refresh(*_):
            if state["busy"]:
                return
            n = picker.value
            start = self._starts[n - 1]
            reading, result = self.check(text.value, n, honest=honest.value)
            self._count += 1
            new = new_word_positions(start["tokens"], reading["tokens"])
            live.value = (
                f'<div style="{_FONT};background:{th["slide"]};border:1px solid {th["line"]};border-radius:16px;'
                f'padding:12px 18px 16px;color:{th["ink"]}"><div style="{_MONO};font-size:12px;font-weight:700;'
                f'letter-spacing:.16em;color:{th["muted"]};margin-bottom:4px">THE MODEL READS</div>'
                f'{render_word_chips(reading["tokens"], reading["pushes"], reading["known"], new)}'
                f'{render_gauge(reading["p_pos"], state["p"], str(self._count))}'
                f'{render_status(result, self.levels[n - 1])}</div>')
            state["p"] = reading["p_pos"]
            if result["valid"] and not state["won"]:
                party.value = render_celebration(result, seed=self._count)
            elif not result["valid"]:
                party.value = ""
            state["won"] = result["valid"]
            board.value = render_scoreboard(self.best, self.levels)

        def on_text(change):
            if honest.value:  # a new sentence needs a new word of honour
                state["busy"] = True
                honest.value = False
                state["busy"] = False
            refresh()

        def load(*_):
            n = picker.value
            lv = self.levels[n - 1]
            intro.value = render_level_intro(n, lv, self._starts[n - 1]["p_pos"], len(self.levels))
            honest.layout.display = "" if lv.get("rule") == "liar" else "none"
            state.update(busy=True, p=self._starts[n - 1]["p_pos"], won=False)
            honest.value = False
            text.value = lv["start"]
            state["busy"] = False
            party.value = ""
            refresh()

        def go_next(_):
            picker.value = picker.value % len(self.levels) + 1

        text.observe(on_text, names="value")
        honest.observe(refresh, names="value")
        picker.observe(load, names="value")
        reset.on_click(load)
        nxt.on_click(go_next)
        load()
        app = w.VBox([title, picker, intro, text, honest, live, party, w.HBox([reset, nxt]), board])
        app.add_class("flipit-app")
        self._ui = {"app": app, "picker": picker, "intro": intro, "text": text, "honest": honest, "live": live,
                    "party": party, "board": board, "reset": reset, "next": nxt}
        return app


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


# --------------------------------------------------------------------------------------------------
# Challenge 2 · YOLO Live: images, video, webcam
# --------------------------------------------------------------------------------------------------
# The notebook loads the model and passes a ``detect(frame_bgr) -> [(class, confidence, (x1, y1, x2, y2)), ...]``
# function to these helpers, so everything below runs (and is tested) without downloading a model.

#: The five sizes of YOLO26 (detection, COCO val2017, 640 px), from https://docs.ultralytics.com/models/yolo26/
#: (CPU = ONNX on an Intel Xeon in the Ultralytics benchmark, T4 = NVIDIA T4 GPU with TensorRT).
YOLO26_SIZES = [
    {"model": "yolo26n.pt", "size": "nano", "params (M)": 2.4, "mAP 50-95": 40.9, "CPU ms": 38.9, "T4 GPU ms": 1.7},
    {"model": "yolo26s.pt", "size": "small", "params (M)": 9.5, "mAP 50-95": 48.6, "CPU ms": 87.2, "T4 GPU ms": 2.5},
    {"model": "yolo26m.pt", "size": "medium", "params (M)": 20.4, "mAP 50-95": 53.1, "CPU ms": 220.0, "T4 GPU ms": 4.7},
    {"model": "yolo26l.pt", "size": "large", "params (M)": 24.8, "mAP 50-95": 55.0, "CPU ms": 286.2, "T4 GPU ms": 6.2},
    {"model": "yolo26x.pt", "size": "extra-large", "params (M)": 55.7, "mAP 50-95": 57.5, "CPU ms": 525.8,
     "T4 GPU ms": 11.8},
]

#: The tasks of the same family: same backbone, a different "head" and a different output.
YOLO_TASKS = [
    {"task": "detect", "weights": "yolo26n.pt", "output": "boxes + class + confidence"},
    {"task": "segment", "weights": "yolo26n-seg.pt", "output": "boxes + a pixel mask per object"},
    {"task": "pose", "weights": "yolo26n-pose.pt", "output": "people + 17 body keypoints (skeleton)"},
    {"task": "obb", "weights": "yolo26n-obb.pt", "output": "rotated boxes (aerial photos: ships, planes)"},
    {"task": "classify", "weights": "yolo26n-cls.pt", "output": "one label for the whole image (like notebook 04)"},
]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

#: Box colours (Okabe–Ito, colour-blind safe), given to the classes in the order they first appear.
OVERLAY_COLORS = ["#56B4E9", "#E69F00", "#009E73", "#F0E442", "#CC79A7", "#D55E00", "#0072B2", "#FFFFFF"]
#: Lines on a white chart: the two overlay colours that are too light on white get a darker twin.
_LINE_ON_WHITE = {"#F0E442": "#B5A600", "#FFFFFF": "#6B7A90"}
_INK = (24, 16, 8)            # BGR of the dark text on the label chips
_PANEL = (46, 26, 14)         # BGR of the navy HUD panels


def list_media(folder) -> tuple[list, list]:
    """``(images, videos)`` directly inside ``folder`` (sorted by name; hidden files and README skipped)."""
    from pathlib import Path

    folder = Path(folder)
    if not folder.is_dir():
        return [], []
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    return ([p for p in files if p.suffix.lower() in IMAGE_SUFFIXES],
            [p for p in files if p.suffix.lower() in VIDEO_SUFFIXES])


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    """``"#E69F00"`` → ``(0, 159, 230)`` (OpenCV uses blue-green-red order)."""
    c = color.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


class ClassColors(dict):
    """``{class: "#hex"}`` that gives the next colour to each new class (the same class keeps its colour)."""

    def __missing__(self, cls):
        self[cls] = OVERLAY_COLORS[len(self) % len(OVERLAY_COLORS)]
        return self[cls]

    def bgr(self, cls) -> tuple[int, int, int]:
        return hex_to_bgr(self[cls])


def detections_from_result(result) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """An Ultralytics ``Results`` object → ``[(class, confidence, (x1, y1, x2, y2)), ...]`` (plain Python)."""
    boxes = result.boxes
    return [(result.names[int(c)], float(p), tuple(float(v) for v in b))
            for c, p, b in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]


def count_classes(detections, min_conf: float = 0.0) -> dict[str, int]:
    """``{class: how many boxes}`` above ``min_conf``, most frequent first (ties: alphabetical)."""
    counts: dict[str, int] = {}
    for cls, conf, _ in detections:
        if conf >= min_conf:
            counts[cls] = counts.get(cls, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def counts_text(counts: Mapping[str, int], max_items: int = 4) -> str:
    """``{"car": 3, "person": 1}`` → ``"3 car · 1 person"`` (ASCII ``|`` separator for OpenCV)."""
    items = [f"{n} {c}" for c, n in list(counts.items())[:max_items]]
    rest = len(counts) - max_items
    return "  |  ".join(items + ([f"+{rest} more"] if rest > 0 else [])) or "nothing found"


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes: 0 = no overlap, 1 = the same box."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class SimpleTracker:
    """A tiny multi-object tracker: it links the boxes of one frame to the boxes of the previous frames.

    Rule (greedy IoU matching): a new box continues the track of the **same class** whose last box overlaps it
    most (IoU ≥ ``iou_min``); boxes that match nothing start a new track with a new ID; a track that is not
    seen for ``max_missed`` frames is closed. Every track keeps the last ``trail`` centres of its box
    (the motion trail). Real trackers (ByteTrack, BoT-SORT in ``model.track``) add a motion model on top.
    """

    def __init__(self, iou_min: float = 0.25, max_missed: int = 12, trail: int = 40):
        self.iou_min, self.max_missed, self.trail = float(iou_min), int(max_missed), int(trail)
        self.tracks: dict[int, dict] = {}
        self.history: dict[int, dict] = {}   # every track ever opened: class and how many frames it was seen
        self._next = 1

    def update(self, detections) -> list[tuple[int, str, float, tuple]]:
        """Feed one frame's ``[(class, confidence, box), ...]``; returns ``[(track id, class, confidence, box)]``."""
        pairs = sorted(((box_iou(t["box"], d[2]), tid, j) for tid, t in self.tracks.items()
                        for j, d in enumerate(detections) if t["cls"] == d[0]), reverse=True)
        used_t, used_d, out = set(), set(), []
        for iou, tid, j in pairs:
            if iou < self.iou_min:
                break
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j)
            self._step(tid, detections[j])
            out.append((tid, *detections[j]))
        for j, det in enumerate(detections):
            if j not in used_d:
                tid, self._next = self._next, self._next + 1
                self.tracks[tid] = {"cls": det[0], "trail": [], "missed": 0}
                self.history[tid] = {"cls": det[0], "hits": 0}
                self._step(tid, det)
                used_t.add(tid)
                out.append((tid, *det))
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return out

    def _step(self, tid: int, det) -> None:
        cls, conf, box = det
        t = self.tracks[tid]
        t.update(box=tuple(box), conf=float(conf), missed=0)
        t["trail"] = (t["trail"] + [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)])[-self.trail:]
        self.history[tid]["hits"] += 1

    def unique_counts(self, min_hits: int = 5) -> dict[str, int]:
        """How many **different** objects of each class were tracked for at least ``min_hits`` frames."""
        counts: dict[str, int] = {}
        for h in self.history.values():
            if h["hits"] >= min_hits:
                counts[h["cls"]] = counts.get(h["cls"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def add_to_heatmap(heat: np.ndarray, boxes, weight: float = 1.0) -> np.ndarray:
    """Add one "visit" per box to ``heat`` (an ``H × W`` float array), as a disc at the box centre.

    Summed over a video, it shows **where** objects spend their time (lanes, sidewalks…). Returns ``heat``.
    """
    import cv2

    h, w = heat.shape[:2]
    for x1, y1, x2, y2 in boxes:
        cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        r = max(2, int(0.25 * min(x2 - x1, y2 - y1)))
        if 0 <= cx < w and 0 <= cy < h:
            stamp = np.zeros_like(heat)
            cv2.circle(stamp, (cx, cy), r, weight, -1)
            heat += stamp
    return heat


def heatmap_overlay(frame_bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """The heatmap (blurred, log scale, "inferno" colours) blended over a darkened frame (BGR)."""
    import cv2

    if heat.max() <= 0:
        return frame_bgr.copy()
    k = max(3, (min(heat.shape) // 20) | 1)
    smooth = cv2.GaussianBlur(heat.astype(np.float32), (k, k), 0)
    norm = np.log1p(smooth) / np.log1p(smooth.max())
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    base = (frame_bgr * 0.45).astype(np.uint8)
    a = (alpha * np.clip(norm * 1.6, 0, 1))[..., None]
    return (base * (1 - a) + colour * a).astype(np.uint8)


def count_timeline(rows: Sequence[Mapping[str, int]], times: Sequence[float]) -> pd.DataFrame:
    """One row per processed frame (index = time in seconds), one column per class, 0 when absent."""
    df = pd.DataFrame(list(rows), index=pd.Index(list(times), name="time (s)")).fillna(0).astype(int)
    order = df.sum().sort_values(ascending=False).index if len(df.columns) else []
    return df[list(order)]


# ---- drawing (OpenCV, BGR) ----

def _text(img, text, org, scale, colour, thick=1):
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _panel(img, x1, y1, x2, y2, alpha=0.72, colour=_PANEL):
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, colour), alpha, 0)


def draw_box(img: np.ndarray, box, label: str, colour, conf: float | None = None, scale: float = 1.0) -> None:
    """One stylish box, in place: thin frame, thick corner brackets, a label chip and a confidence bar."""
    import cv2

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    t = max(1, int(round(1.5 * scale)))
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, t, cv2.LINE_AA)
    arm = max(6, int(min(x2 - x1, y2 - y1) * 0.22))
    for (cx, cy), (dx, dy) in (((x1, y1), (1, 1)), ((x2, y1), (-1, 1)), ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), colour, 3 * t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), colour, 3 * t, cv2.LINE_AA)
    fs = 0.5 * scale
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t))
    pad = int(4 * scale)
    bar = int(4 * scale) if conf is not None else 0
    top = y1 - th - 2 * pad - bar
    if top < 0:                                  # no room above the box: put the chip inside it
        top = y1
    cv2.rectangle(img, (x1, top), (x1 + tw + 2 * pad, top + th + 2 * pad + bar), colour, -1)
    _text(img, label, (x1 + pad, top + th + pad), fs, _INK, max(1, t))
    if conf is not None:                         # confidence bar: dark track, white fill
        yb = top + th + 2 * pad
        cv2.rectangle(img, (x1, yb), (x1 + tw + 2 * pad, yb + bar), _INK, -1)
        cv2.rectangle(img, (x1, yb), (x1 + int((tw + 2 * pad) * max(0.0, min(1.0, conf))), yb + bar),
                      (255, 255, 255), -1)


def draw_trail(img: np.ndarray, points, colour, scale: float = 1.0) -> None:
    """A motion trail, in place: older points thinner, the newest a dot."""
    import cv2

    pts = [(int(x), int(y)) for x, y in points]
    n = len(pts)
    for k in range(1, n):
        cv2.line(img, pts[k - 1], pts[k], colour, max(1, int(round(scale * (1 + 3 * k / n)))), cv2.LINE_AA)
    if pts:
        cv2.circle(img, pts[-1], max(2, int(4 * scale)), colour, -1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, title: str, lines: Sequence[str] = (), fps: float | None = None,
             latency_ms: float | None = None, scale: float = 1.0) -> None:
    """The heads-up display, in place: a navy panel top-left (title + lines) and an FPS meter top-right."""
    import cv2

    h, w = img.shape[:2]
    lh, pad = int(22 * scale), int(10 * scale)
    width = max([cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, 1)[0][0] for s in lines] +
                [cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)[0][0]]) + 2 * pad
    _panel(img, 0, 0, width, pad + lh * (1 + len(lines)) + pad // 2)
    _text(img, title, (pad, pad + int(16 * scale)), 0.6 * scale, hex_to_bgr("#F0E442"), max(1, int(2 * scale)))
    for k, s in enumerate(lines, start=1):
        _text(img, s, (pad, pad + int(16 * scale) + k * lh), 0.5 * scale, (245, 245, 245), 1)
    if fps is not None:
        txt = f"{fps:4.1f} FPS" + (f"  {latency_ms:3.0f} ms" if latency_ms is not None else "")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(1, int(2 * scale)))
        _panel(img, w - tw - 2 * pad, 0, w, th + 2 * pad + int(8 * scale))
        _text(img, txt, (w - tw - pad, th + pad), 0.6 * scale, (255, 255, 255), max(1, int(2 * scale)))
        # speed bar: full at 30 FPS; green ≥ 15, orange ≥ 8, red below
        frac = max(0.0, min(1.0, fps / 30))
        col = hex_to_bgr("#009E73" if fps >= 15 else "#E69F00" if fps >= 8 else "#D55E00")
        y = th + pad + int(4 * scale)
        cv2.rectangle(img, (w - tw - pad, y), (w - pad, y + int(4 * scale)), (90, 90, 90), -1)
        cv2.rectangle(img, (w - tw - pad, y), (w - tw - pad + int(tw * frac), y + int(4 * scale)), col, -1)


def draw_detections(frame_bgr: np.ndarray, detections, min_conf: float = 0.25, colors: ClassColors | None = None,
                    title: str | None = None, show_conf: bool = True) -> np.ndarray:
    """Every box above ``min_conf`` (stylish), plus a HUD with the object count per class. Returns a new BGR image."""
    img = np.ascontiguousarray(frame_bgr).copy()
    colors = ClassColors() if colors is None else colors
    s = max(0.6, max(img.shape[:2]) / 900)
    kept = sorted((d for d in detections if d[1] >= min_conf), key=lambda d: d[1])   # most confident drawn last
    for cls, conf, box in kept:
        draw_box(img, box, f"{cls} {conf:.2f}" if show_conf else cls, colors.bgr(cls),
                 conf if show_conf else None, s)
    counts = count_classes(kept)
    draw_hud(img, title or f"{len(kept)} objects  (confidence >= {min_conf:.2f})",
             [f"{n:>3}  {c}" for c, n in list(counts.items())[:6]], scale=s)
    return img


def draw_tracks(frame_bgr: np.ndarray, tracks, tracker: SimpleTracker, colors: ClassColors) -> np.ndarray:
    """Boxes with their track ID (``car #12``) and the motion trail of each track. Returns a new BGR image."""
    img = np.ascontiguousarray(frame_bgr).copy()
    s = max(0.6, max(img.shape[:2]) / 900)
    for tid, cls, conf, box in tracks:
        if tid in tracker.tracks:
            draw_trail(img, tracker.tracks[tid]["trail"], colors.bgr(cls), s)
    for tid, cls, conf, box in tracks:
        draw_box(img, box, f"{cls} #{tid}", colors.bgr(cls), conf, s)
    return img


# ---- streams: a video file, a webcam, or anything that yields BGR frames ----

def resize_frame(frame: np.ndarray, width: int | None) -> np.ndarray:
    """Shrink a frame to ``width`` pixels (keeps the aspect ratio; never enlarges)."""
    import cv2

    if not width or frame.shape[1] <= width:
        return frame
    return cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)


def video_frames(path, width: int | None = 960, max_seconds: float | None = None):
    """Yield ``(frame_bgr, t_seconds)`` from a video file (resized to ``width``); the file is closed at the end."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open the video {path}.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        k = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = k / fps
            if max_seconds is not None and t > max_seconds:
                break
            yield resize_frame(frame, width), t
            k += 1
    finally:
        cap.release()


def video_info(path) -> dict:
    """Frames per second, number of frames, duration (s) and size of a video file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {"fps": round(fps, 2), "frames": n, "seconds": round(n / fps, 1),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def webcam_frames(index: int = 0, width: int = 640, height: int = 480, mirror: bool = True):
    """Yield ``(frame_bgr, t_seconds)`` from a camera; the camera is released when the generator closes.

    Raises ``RuntimeError`` if no camera can be opened or it sends no image (busy, blocked by privacy settings…).
    """
    import sys
    import time

    import cv2

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY   # DirectShow opens fast on Windows
    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"No camera found at index {index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        n, start = 0, time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if n == 0:
                    raise RuntimeError(f"The camera at index {index} opened but sent no image.")
                break
            n += 1
            yield (cv2.flip(frame, 1) if mirror else frame), time.perf_counter() - start
    finally:
        cap.release()


def run_stream(detect, frames, min_conf: float = 0.3, title: str = "YOLO", display=None,
               time_limit: float | None = None, keep_every: int = 2, keep_width: int = 640,
               max_kept: int = 400, tracker: SimpleTracker | None = None) -> dict:
    """Run a detector on a stream of frames: track, draw the HUD, show, and collect statistics.

    * ``detect(frame_bgr)`` returns ``[(class, confidence, (x1, y1, x2, y2)), ...]`` (e.g. YOLO);
    * ``frames`` yields ``(frame_bgr, t_seconds)`` (:func:`video_frames`, :func:`webcam_frames`…);
    * ``display(annotated_bgr)`` shows a frame and returns ``False`` to stop (e.g. key Q).

    The frame generator is always closed at the end, even after an error or Ctrl+C: that is what releases the
    camera. Returns the kept annotated frames (RGB, every ``keep_every``-th, for the replay), the count
    timeline, the heatmap, the unique objects per class, the latency of each frame and the average FPS.
    """
    import time

    tracker = tracker or SimpleTracker()
    colors = ClassColors()
    kept, rows, times, latencies, heat, first, discovered = [], [], [], [], None, None, {}
    fps, n, prev, start = None, 0, None, time.perf_counter()
    try:
        for frame, t in frames:
            if time_limit is not None and t > time_limit:
                break
            t0 = time.perf_counter()
            dets = [d for d in detect(frame) if d[1] >= min_conf]
            latency = (time.perf_counter() - t0) * 1000
            tracks = tracker.update(dets)
            if heat is None:
                heat, first = np.zeros(frame.shape[:2], np.float32), frame.copy()
            add_to_heatmap(heat, [d[2] for d in dets])
            counts = count_classes(dets)
            for cls in counts:
                discovered.setdefault(cls, round(t, 1))
            rows.append(counts); times.append(round(t, 3)); latencies.append(latency)
            now = time.perf_counter()
            if prev is not None:
                inst = 1.0 / max(now - prev, 1e-6)
                fps = inst if fps is None else 0.9 * fps + 0.1 * inst          # smoothed: easier to read
            prev = now
            img = draw_tracks(frame, tracks, tracker, colors)
            unique = tracker.unique_counts()
            lines = [f"now: {counts_text(counts, 3)}",
                     f"seen so far: {sum(unique.values())} objects, {len(discovered)} classes",
                     f"t = {t:4.1f} s" + (f" / {time_limit:.0f} s" if time_limit else "")]
            draw_hud(img, title, lines, fps=fps, latency_ms=latency, scale=max(0.6, max(img.shape[:2]) / 900))
            if n % keep_every == 0 and len(kept) < max_kept:
                kept.append(np.ascontiguousarray(resize_frame(img, keep_width)[..., ::-1]))
            n += 1
            if display is not None and display(img) is False:
                break
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
    elapsed = time.perf_counter() - start
    return {"frames": n, "kept": kept, "keep_every": keep_every, "timeline": count_timeline(rows, times),
            "heat": heat, "first_frame": first, "unique": tracker.unique_counts(), "discovered": discovered,
            "latency_ms": latencies, "fps": (n / elapsed) if n and elapsed > 0 else None, "colors": colors,
            "tracker": tracker}


def local_display(mode: str = "window", title: str = "YOLO live - press Q to stop"):
    """``(show, close)`` for :func:`run_stream` on your own computer.

    ``"window"``: an OpenCV window (smoothest; click it and press Q or Esc to stop). If no window can be opened
    (e.g. a remote server), it falls back to ``"inline"``: the video plays inside the notebook in an
    ``ipywidgets`` image (stop with the Interrupt button). ``close()`` always closes the window.
    """
    import cv2

    if mode == "window":
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("OpenCV cannot open a window here: showing the video inside the notebook instead.")
            mode = "inline"
    if mode == "inline":
        import ipywidgets as w
        from IPython.display import display

        screen = w.Image(format="jpeg", layout=w.Layout(max_width="800px"))
        display(screen)

        def show(img):
            screen.value = bgr_to_jpeg(img, 70)
            return True

        return show, lambda: None

    def show(img):
        cv2.imshow(title, img)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), ord("Q"), 27)

    def close():
        try:
            cv2.destroyWindow(title)
            cv2.waitKey(1)
        except cv2.error:
            pass

    return show, close


# ---- showing results in the notebook ----

def bgr_to_jpeg(img_bgr: np.ndarray, quality: int = 82) -> bytes:
    """Encode a BGR image as JPEG bytes (small outputs keep the notebook light)."""
    import cv2

    return cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def show_bgr(img_bgr: np.ndarray, width: int | None = 960, quality: int = 82):
    """Display a BGR image inline as a JPEG."""
    from IPython.display import Image, display

    display(Image(bgr_to_jpeg(resize_frame(img_bgr, width), quality)))


def fig_to_jpeg(fig, dpi: int = 90, quality: int = 82) -> bytes:
    """A Matplotlib figure as JPEG bytes (closes the figure)."""
    import io

    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=dpi, facecolor="white", pil_kwargs={"quality": quality})
    plt.close(fig)
    return buf.getvalue()


def show_fig(fig, dpi: int = 90, quality: int = 82):
    """Display a figure full of photos as a JPEG (much lighter than a PNG)."""
    from IPython.display import Image, display

    display(Image(fig_to_jpeg(fig, dpi, quality)))


def encode_clip(frames_rgb: Sequence[np.ndarray], fps: float, folder, crf: int = 48) -> list[tuple]:
    """Write the frames as a small web video; returns ``[(path, mime type)]`` (empty if no encoder works).

    WebM/VP9 first (it plays in VS Code, Chrome, Edge, Firefox and recent Safari; ``crf`` = quality, higher =
    smaller file): with the ``ffmpeg`` program if there is one (e.g. Colab), else with OpenCV's own writer.
    Last resort: MP4/H.264 with OpenCV (plays in browsers, but not inside VS Code).
    """
    import shutil
    import subprocess
    from pathlib import Path

    import cv2

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not len(frames_rgb):
        return []
    h, w = frames_rgb[0].shape[:2]
    h, w = h - h % 2, w - w % 2                       # codecs want even sizes
    webm, mp4 = folder / "clip.webm", folder / "clip.mp4"
    for old in (webm, mp4):
        old.unlink(missing_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        raw = b"".join(np.ascontiguousarray(f[:h, :w]).tobytes() for f in frames_rgb)
        cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf),
               "-deadline", "realtime", "-cpu-used", "8", "-pix_fmt", "yuv420p", str(webm)]
        try:
            subprocess.run(cmd, input=raw, check=True, capture_output=True)
            if webm.exists() and webm.stat().st_size > 1000:
                return [(webm, "video/webm")]
        except (subprocess.CalledProcessError, OSError):
            pass
    crf_prop = getattr(cv2, "VIDEOWRITER_PROP_CRF", None)
    attempts = [(webm, "video/webm", "VP90", cv2.CAP_FFMPEG, [crf_prop, crf] if crf_prop is not None else [])]
    attempts += [(mp4, "video/mp4", "avc1", api, []) for api in
                 ([cv2.CAP_FFMPEG] + ([cv2.CAP_MSMF] if hasattr(cv2, "CAP_MSMF") else []))]
    for path, mime, fourcc, api, params in attempts:
        try:
            writer = cv2.VideoWriter(str(path), api, cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h), params)
        except cv2.error:
            continue
        if not writer.isOpened():
            writer.release()
            continue
        for f in frames_rgb:
            writer.write(np.ascontiguousarray(f[:h, :w, ::-1]))
        writer.release()
        if path.exists() and path.stat().st_size > 1000:
            return [(path, mime)]
    return []


def clip_html(frames_rgb: Sequence[np.ndarray], fps: float, folder, width: int = 720,
              webp_fps: float = 8, max_width: int = 520, quality: int = 48) -> str:
    """HTML that plays the frames inline as an animated WebP ``<img>``.

    Animated WebP plays natively inside VS Code notebooks, GitHub previews, Google Colab and every modern
    browser (unlike base64 ``<video>`` tags, which VS Code's webview CSP and GitHub often block or strip).
    Also writes ``clip.webp`` into ``folder`` when ``folder`` is given.
    """
    import base64
    import io
    from pathlib import Path

    from PIL import Image

    if not len(frames_rgb):
        return ""
    step = max(1, round(fps / webp_fps))
    imgs = []
    for f in frames_rgb[::step]:
        im = Image.fromarray(f)
        if max_width and im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.Resampling.BILINEAR)
        imgs.append(im)
    buf = io.BytesIO()
    imgs[0].save(buf, format="WEBP", save_all=True, append_images=imgs[1:], duration=int(1000 * step / fps),
                 loop=0, quality=quality, method=4)
    raw = buf.getvalue()
    if folder is not None:
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "clip.webp").write_bytes(raw)
    return (f'<img width="{width}" style="max-width:100%;border-radius:10px" '
            f'src="data:image/webp;base64,{base64.b64encode(raw).decode()}">')


# ---- figures ----

def plot_media_grid(images_bgr: Sequence[np.ndarray], titles: Sequence[str], ncols: int = 4, width: float = 16):
    """A grid of (already annotated) BGR images with a title under each one. Returns the figure."""
    import matplotlib.pyplot as plt

    n = len(images_bgr)
    nrows = int(np.ceil(n / ncols))
    cell_h = width / ncols * 0.68
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, nrows * cell_h + 0.3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img, title in zip(axes.ravel(), images_bgr, titles):
        ax.imshow(img[..., ::-1])
        ax.set_title(title, fontsize=10.5, loc="left", fontweight="normal", pad=3)
    fig.tight_layout()
    return fig


def plot_detection_dashboard(frame_bgr: np.ndarray, detections, min_conf: float = 0.25, title: str = ""):
    """The "count the objects" dashboard: annotated photo (left), how many of each class and every box's
    confidence against the threshold (right). Returns the figure."""
    import matplotlib.pyplot as plt

    colors = ClassColors()
    for cls, _, _ in sorted(detections, key=lambda d: -d[1]):
        colors[cls]                                       # same colour order as the boxes
    img = draw_detections(frame_bgr, detections, min_conf, colors)
    fig = plt.figure(figsize=(15, 6.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1.15], wspace=0.08, hspace=0.42)
    ax_img = fig.add_subplot(grid[:, 0])
    ax_img.imshow(img[..., ::-1])
    ax_img.axis("off")
    counts = count_classes(detections, min_conf)
    ax_c = fig.add_subplot(grid[0, 1])
    names = list(counts)[:8][::-1]
    ax_c.barh(names, [counts[c] for c in names], color=[colors[c] for c in names], height=0.6,
              edgecolor="#0E1A2E", linewidth=0.6)
    for k, c in enumerate(names):
        ax_c.text(counts[c] + 0.1, k, str(counts[c]), va="center", fontsize=11, fontweight="bold", color="#0E1A2E")
    ax_c.set_title(f"How many? {sum(counts.values())} objects", fontsize=12.5)
    ax_c.grid(axis="y", visible=False)
    ax_c.set_xlim(0, max(counts.values() or [1]) * 1.25)
    ax_c.set_xlabel("boxes")
    ax_p = fig.add_subplot(grid[1, 1])
    dets = sorted(detections, key=lambda d: -d[1])[:40]
    xs = np.arange(len(dets))
    ax_p.bar(xs, [d[1] for d in dets], color=[colors[d[0]] if d[1] >= min_conf else "#D5DBE3" for d in dets],
             edgecolor="#0E1A2E", linewidth=0.4, width=0.8)
    ax_p.axhline(min_conf, color="#0E1A2E", lw=1.5, ls="--")
    ax_p.text(len(dets) - 0.5, min_conf + 0.03, f"threshold {min_conf:.2f}", ha="right", fontsize=9.5)
    ax_p.set_ylim(0, 1.05)
    ax_p.set_xticks([])
    ax_p.set_ylabel("confidence")
    ax_p.set_title("Every box YOLO proposed (grey = below the threshold)", fontsize=11.5)
    ax_p.grid(axis="x", visible=False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    return fig


def plot_threshold_strip(frame_bgr: np.ndarray, detections, thresholds=(0.1, 0.3, 0.6), width: float = 16):
    """The same photo at several confidence thresholds, side by side (a static version of the slider)."""
    colors = ClassColors()
    for cls, _, _ in sorted(detections, key=lambda d: -d[1]):
        colors[cls]
    imgs = [draw_detections(frame_bgr, detections, th, colors) for th in thresholds]
    titles = [f"threshold {th:.2f} → {sum(d[1] >= th for d in detections)} boxes" for th in thresholds]
    return plot_media_grid(imgs, titles, ncols=len(thresholds), width=width)


def threshold_explorer(frame_bgr: np.ndarray, detections, value: float = 0.25, width: int = 900):
    """An interactive slider: move the confidence threshold and watch the boxes appear and disappear.

    ``detections`` should come from one prediction at a **low** threshold (e.g. 0.05): the slider only filters
    them, so it is instant (no new prediction). Returns the ``ipywidgets`` box to display.
    """
    import ipywidgets as w

    colors = ClassColors()
    for cls, _, _ in sorted(detections, key=lambda d: -d[1]):
        colors[cls]
    small = resize_frame(frame_bgr, width)
    k = small.shape[1] / frame_bgr.shape[1]
    dets = [(c, p, tuple(v * k for v in b)) for c, p, b in detections]
    slider = w.FloatSlider(value=value, min=0.05, max=0.95, step=0.05, description="threshold",
                           continuous_update=True, readout_format=".2f", layout=w.Layout(width="60%"))
    image = w.Image(format="jpeg", layout=w.Layout(max_width=f"{width}px"))

    def redraw(*_):
        image.value = bgr_to_jpeg(draw_detections(small, dets, slider.value, colors))

    slider.observe(redraw, names="value")
    redraw()
    return w.VBox([slider, image])


def plot_video_report(result: Mapping, title: str = "", top: int = 5, smooth_s: float = 0.5):
    """Left: the heatmap of where objects appeared. Right: objects per frame over time (top classes,
    averaged over ``smooth_s`` seconds so the flicker of single frames does not hide the trend)."""
    import matplotlib.pyplot as plt

    colors = result["colors"]
    fig, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    if result["heat"] is not None:
        ax_h.imshow(heatmap_overlay(result["first_frame"], result["heat"])[..., ::-1])
    ax_h.axis("off")
    ax_h.set_title("Where objects appeared (brighter = more often)", fontsize=12)
    df = result["timeline"]
    step = float(np.median(np.diff(df.index))) if len(df) > 1 else 1.0
    smooth = df.rolling(max(1, round(smooth_s / max(step, 1e-6))), min_periods=1, center=True).mean()
    classes = list(df.columns[:top])
    for cls in classes:
        ax_t.plot(smooth.index, smooth[cls], lw=2, color=_LINE_ON_WHITE.get(colors[cls], colors[cls]),
                  label=f"{cls} ({result['unique'].get(cls, 0)} different)")
    ax_t.set_xlabel("time in the video (s)")
    ax_t.set_ylabel(f"objects in the frame ({smooth_s:g} s average)")
    ax_t.set_title("How many objects in each frame", fontsize=12)
    ax_t.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10)
    ax_t.set_ylim(bottom=0)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---- Google Colab: the webcam lives in the browser, so JavaScript sends frames to Python ----

_COLAB_JS = r"""
var yoloDiv = null, yoloVideo = null, yoloImg = null, yoloStream = null, yoloCanvas = null, yoloStop = false;
async function yoloStart(width) {
  try {
    yoloStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user'}});
  } catch (err) {
    return 'ERROR:' + err.name;
  }
  yoloDiv = document.createElement('div');
  const btn = document.createElement('button');
  btn.textContent = 'Stop';
  btn.onclick = () => { yoloStop = true; };
  yoloVideo = document.createElement('video');
  yoloVideo.style.display = 'none';
  yoloVideo.setAttribute('playsinline', '');
  yoloVideo.srcObject = yoloStream;
  await yoloVideo.play();
  yoloImg = document.createElement('img');
  yoloImg.style.width = width + 'px';
  yoloImg.style.borderRadius = '10px';
  yoloDiv.appendChild(btn);
  yoloDiv.appendChild(document.createElement('br'));
  yoloDiv.appendChild(yoloImg);
  document.body.appendChild(yoloDiv);
  yoloCanvas = document.createElement('canvas');
  yoloCanvas.width = width;
  yoloCanvas.height = Math.round(width * yoloVideo.videoHeight / yoloVideo.videoWidth);
  google.colab.output.setIframeHeight(document.documentElement.scrollHeight + yoloCanvas.height, true);
  return 'OK';
}
function yoloEnd() {
  if (yoloStream) { yoloStream.getTracks().forEach((t) => t.stop()); }
  if (yoloDiv) { yoloDiv.remove(); }
  yoloDiv = null; yoloStream = null; yoloStop = false;
  return '';
}
function yoloFrame(annotated) {
  if (yoloStop || !yoloDiv) { return yoloEnd(); }
  if (annotated) { yoloImg.src = annotated; }
  const ctx = yoloCanvas.getContext('2d');
  ctx.translate(yoloCanvas.width, 0); ctx.scale(-1, 1);          // mirror image
  ctx.drawImage(yoloVideo, 0, 0, yoloCanvas.width, yoloCanvas.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return yoloCanvas.toDataURL('image/jpeg', 0.8);
}
"""


class ColabWebcam:
    """Google Colab only: the webcam lives in the browser, so a small JavaScript loop sends frames to Python.

    ``frames()`` yields ``(frame_bgr, t)`` like :func:`webcam_frames`; ``show`` is the ``display`` callback of
    :func:`run_stream` and sends the annotated frame back to the page with the next request. Each frame makes a
    round trip browser → Colab server → browser, so expect a few FPS: the model is fast, the network is not.
    Nothing is saved to disk; the frames do travel to Google's runtime (that is where Python runs).
    """

    def __init__(self, width: int = 640, max_seconds: float = 30):
        self.width, self.max_seconds, self.annotated = int(width), float(max_seconds), ""

    def frames(self):
        import base64
        import json
        import time

        import cv2
        from google.colab.output import eval_js
        from IPython.display import Javascript, display

        display(Javascript(_COLAB_JS))
        status = eval_js(f"yoloStart({self.width})")
        if status != "OK":
            raise RuntimeError(f"No webcam access in the browser ({str(status)[6:] or 'no answer'}): "
                               "allow the camera for colab.research.google.com.")
        start = time.perf_counter()
        try:
            while time.perf_counter() - start < self.max_seconds:
                data = eval_js(f"yoloFrame({json.dumps(self.annotated)})")
                self.annotated = ""
                if not data:                     # the Stop button
                    break
                buf = np.frombuffer(base64.b64decode(data.split(",", 1)[1]), np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame, time.perf_counter() - start
        finally:
            eval_js("yoloEnd()")

    def show(self, img_bgr: np.ndarray) -> bool:
        import base64

        self.annotated = "data:image/jpeg;base64," + base64.b64encode(bgr_to_jpeg(img_bgr, 75)).decode()
        return True


# --------------------------------------------------------------------------------------------------
# Figures used by the challenge notebooks
# --------------------------------------------------------------------------------------------------

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

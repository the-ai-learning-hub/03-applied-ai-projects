"""Scoring rules and the scorecard figure for the three Lesson 03 class challenges.

* Challenge 1 · Flip It!                 (``01_text_mining/03_challenge_flip_it``)
* Challenge 2 · YOLO Scavenger Hunt       (``02_computer_vision/03_challenge_yolo``)
* Challenge 3 · CLIP Pictionary           (``03_multimodal/02_challenge_clip_pictionary``)

Everything here is plain Python + NumPy/pandas/Matplotlib: no model is loaded, so the tests run in
a second. The notebooks call the models and pass the *answers* (labels, probabilities, detected
classes, ranks) to these functions. Challenge 1 also gets its interactive game here
(:class:`FlipGame`, ``ipywidgets`` + HTML), so the notebook stays short. Every rule lives in one
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

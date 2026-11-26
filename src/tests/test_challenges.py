"""Fast tests for src/challenges.py (no model download)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import challenges as ch


def test_medal_for_levels():
    t = {"Bronze": 10, "Silver": 25, "Gold": 40}
    assert ch.medal_for(0, t) == ("Rookie", "Bronze", 10)
    assert ch.medal_for(10, t) == ("Bronze", "Silver", 15)
    assert ch.medal_for(27, t) == ("Silver", "Gold", 13)
    assert ch.medal_for(99, t) == ("Gold", None, None)


def test_clean_entries_dedupes_and_caps():
    entries = [("Hello  world", "pos"), ("hello world", "neg"), ("", "pos"), ("a", "pos"), ("b", "pos")]
    kept = ch.clean_entries(entries, max_entries=2, key=lambda e: e[0])
    assert kept == [("Hello  world", "pos"), ("a", "pos")]


def _tiny_model():
    texts = ["i love this", "love it so much", "great day", "i hate this", "sad day", "hate it so much",
             "worst day", "love love great"]
    labels = ["pos", "pos", "pos", "neg", "neg", "neg", "neg", "pos"]
    vec = TfidfVectorizer(sublinear_tf=True)
    model = LogisticRegression(C=10).fit(vec.fit_transform(texts), labels)
    return vec, model


TINY_LEVELS = [
    {"name": "a", "emoji": "1", "start": "i love this", "goal": "neg", "gold": 1, "silver": 2, "rule": None,
     "mission": "m"},
    {"name": "b", "emoji": "2", "start": "great day", "goal": "neg", "gold": 1, "silver": 2, "rule": "liar",
     "mission": "m"},
    {"name": "c", "emoji": "3", "start": "i hate this", "goal": "pos", "gold": 1, "silver": 3, "rule": "add_only",
     "mission": "m"},
]


def test_word_edits_and_add_only():
    assert ch.word_edits([], []) == 0
    assert ch.word_edits(["i", "love", "it"], ["i", "hate", "it"]) == 1
    assert ch.word_edits(["i", "love", "it"], ["i", "hate", "it", "!"]) == 2
    assert ch.word_edits(["a", "b"], []) == 2
    assert ch.is_add_only(["i", "hate", "this"], ["oh", "i", "hate", "this", "lol"])
    assert not ch.is_add_only(["i", "hate", "this"], ["i", "this", "hate"])
    assert ch.new_word_positions(["i", "love", "it"], ["i", "do", "not", "love", "it"]) == {1, 2}


def test_word_pushes_add_up_to_the_model_score():
    vec, model = _tiny_model()
    for text in ["I love this, I love it", "sad sad day", "qwerty unknown", ""]:
        r = ch.word_pushes(text, vec, model)
        assert np.isclose(r["logit"], model.decision_function(vec.transform([ch.clean_tweet(text)]))[0])
        assert np.isclose(r["p_pos"], model.predict_proba(vec.transform([ch.clean_tweet(text)]))[0, 1])
        assert len(r["tokens"]) == len(r["pushes"]) == len(r["known"])
    r = ch.word_pushes("love hate qwerty", vec, model)
    push = dict(zip(r["tokens"], r["pushes"]))
    assert push["love"] > 0 > push["hate"] and push["qwerty"] == 0 and r["known"] == [True, True, False]
    flipped = ch.word_pushes("love hate", vec, model, pos_label="neg")
    assert np.isclose(flipped["pushes"][0], -push["love"])
    with pytest.raises(ValueError):
        ch.word_pushes("x", TfidfVectorizer(ngram_range=(1, 2)), model)


def test_flip_medal_and_result_rules():
    lv = TINY_LEVELS[0]
    assert ch.flip_medal(1, lv) == ("Gold", 3)
    assert ch.flip_medal(2, lv) == ("Silver", 2)
    assert ch.flip_medal(7, lv) == ("Bronze", 1)
    res = ch.flip_result(lv, ["i", "love", "this"], ["i", "hate", "this"], p_pos=0.2)
    assert res["valid"] and res["swaps"] == 1 and res["stars"] == 3
    assert not ch.flip_result(lv, ["i", "love", "this"], ["i", "love", "this"], p_pos=0.9)["flipped"]
    # Liar mode: needs the honour box and no obvious negative word
    liar = TINY_LEVELS[1]
    assert not ch.flip_result(liar, ["great", "day"], ["sick", "day"], 0.1)["valid"]
    assert ch.flip_result(liar, ["great", "day"], ["sick", "day"], 0.1, honest=True)["valid"]
    bad = ch.flip_result(liar, ["great", "day"], ["sad", "day"], 0.1, honest=True)
    assert not bad["valid"] and "sad" in bad["problems"][0]
    # Add-only: deleting or replacing a start word breaks the rule
    add = TINY_LEVELS[2]
    assert ch.flip_result(add, ["i", "hate", "this"], ["i", "hate", "this", "love"], 0.8)["valid"]
    assert not ch.flip_result(add, ["i", "hate", "this"], ["i", "love", "this"], 0.8)["valid"]


def test_flip_summary_and_share_line():
    s = ch.flip_summary({1: {"swaps": 1, "stars": 3}, 2: {"swaps": 4, "stars": 1}}, TINY_LEVELS)
    assert (s["stars"], s["max_stars"], s["flips"], s["swaps"]) == (4, 9, 2, 5)
    assert s["medal"] == "Rookie" and "2/3 times in 5 word swaps" in s["share"]
    assert ch.flip_summary({}, TINY_LEVELS)["stars"] == 0


def test_flip_levels_are_well_formed():
    for lv in ch.FLIP_LEVELS:
        assert lv["goal"] in ("pos", "neg") and lv["gold"] <= lv["silver"]
        assert lv["rule"] in (None, "liar", "add_only")
    assert ch.LEVELS["flip_it"]["Gold"] <= 3 * len(ch.FLIP_LEVELS)


def test_flip_game_records_best_and_renders_html():
    vec, model = _tiny_model()
    game = ch.FlipGame(vec, model, levels=TINY_LEVELS)
    assert "FLIPPED" not in game.card("i love this", level=1)          # nothing changed yet
    html_out = game.card("i hate hate this sad", level=1)                # 3 swaps: Bronze
    assert "FLIPPED" in html_out and game.best[1]["swaps"] == 3
    game.card("i hate this", level=1)                                     # 1 swap: better, Gold
    assert game.best[1]["swaps"] == 1 and game.best[1]["stars"] == 3
    game.card("i hate hate this sad", level=1)                            # worse: best is kept
    assert game.best[1]["swaps"] == 1
    assert "qwerty" in game.card("qwerty love", level=None) and 1 in game.best
    assert "FLIP IT!" in game.scorecard().data and "1/3 times" in game.scorecard().data
    assert "love" in game.word_board(["love", "hate", "nope"]).data


def test_flip_game_widget_updates_live():
    vec, model = _tiny_model()
    game = ch.FlipGame(vec, model, levels=TINY_LEVELS)
    game.play()
    ui = game._ui
    assert ui["text"].continuous_update and ui["text"].value == "i love this"
    before = ui["live"].value
    ui["text"].value = "i hate this"                                       # typing triggers the callback
    assert ui["live"].value != before and "FLIPPED" in ui["live"].value
    assert "FLIPPED" in ui["party"].value and game.best[1]["swaps"] == 1
    ui["picker"].value = 2                                                 # Liar level: needs the honour box
    assert ui["text"].value == "great day" and ui["honest"].layout.display == ""
    ui["text"].value = "sad day"
    assert 2 not in game.best
    ui["next"].click()
    assert ui["picker"].value == 3 and ui["honest"].layout.display == "none"
    ui["text"].value = "i love this"                                       # replaced a word: not add-only
    assert 3 not in game.best and "Add-only" in ui["live"].value
    ui["reset"].click()
    assert ui["text"].value == "i hate this"


def test_hunt_score():
    detections = {"a.jpg": ["cup", "laptop", "person"], "b.jpg": ["cup", "cat"]}
    res = ch.hunt_score(detections, hunt_list=["cup", "laptop", "book"],
                        fooled={"b.jpg": "cat", "a.jpg": "giraffe"})
    assert res["found"] == ["cat", "cup", "laptop", "person"]
    assert res["breakdown"] == {"distinct objects": 4, "hunt-list items": 6, "Fool YOLO": 5}
    assert res["total"] == 15
    assert res["hunt_missing"] == ["book"]
    assert res["fools_invalid"] == [("a.jpg", "giraffe")]


def test_hunt_score_accepts_pairs_and_dedupes():
    det = {"s.jpg": ["cat", "bed"]}
    res = ch.hunt_score(det, hunt_list=[], fooled=[("s.jpg", "cat"), ("s.jpg", "bed"), ("s.jpg", "cat")])
    assert res["fools_valid"] == [("s.jpg", "cat"), ("s.jpg", "bed")]
    assert res["breakdown"]["Fool YOLO"] == 10


def test_hunt_score_caps_fools():
    det = {f"{i}.jpg": ["cat"] for i in range(5)}
    res = ch.hunt_score(det, hunt_list=[], fooled={f"{i}.jpg": "cat" for i in range(5)})
    assert len(res["fools_valid"]) == ch.CH2_RULES["max_fools"]


def test_word_tools():
    assert ch.word_count("an orange cat, by the door!") == 6
    assert ch.forbidden_used("A fluffy Kitten next to a DOG") == ["kitten", "dog"]
    assert ch.forbidden_used("catalogue of dogged hot-dogs") == ["dogs"]  # whole words only


def test_rank_and_points():
    scores = [0.2, 0.9, 0.5, 0.1]
    assert ch.rank_of(1, scores) == 1 and ch.rank_of(2, scores) == 2 and ch.rank_of(3, scores) == 4
    assert ch.pictionary_points(1, 4)["total"] == 13          # 10 + short bonus
    assert ch.pictionary_points(1, 9)["total"] == 10
    assert ch.pictionary_points(3, 2)["total"] == 5
    assert ch.pictionary_points(5, 2)["total"] == 2
    assert ch.pictionary_points(6, 2)["total"] == 0
    assert ch.pictionary_points(1, 2, ["cat"]) == {"points": 0, "bonus": 0, "total": 0, "foul": True}


def test_plot_scorecard_returns_figure():
    import matplotlib.pyplot as plt
    fig = ch.plot_scorecard(27, {"a": 20, "b": 7, "c": 0}, ch.LEVELS["yolo_hunt"], "Test", subtitle="x")
    assert fig.axes
    plt.close(fig)


def test_tracker_needs_consecutive_frames():
    tr = ch.HuntTracker(hunt_list=["cup", "book"], min_conf=0.4, min_frames=3)
    assert tr.update([("cup", 0.9)], 0.0) == []
    assert tr.update([("cup", 0.9)], 0.1) == []
    assert tr.update([], 0.2) == []                      # flicker: the streak starts again
    assert tr.update([("cup", 0.9), ("cat", 0.3)], 0.3) == []
    tr.update([("cup", 0.9)], 0.4)
    assert tr.update([("cup", 0.9)], 0.5) == ["cup"]
    assert tr.update([("cup", 0.9)], 0.6) == []          # confirmed only once
    assert tr.seen == ["cup"] and tr.hunt_found == ["cup"] and not tr.done
    assert tr.confirmed["cup"]["t"] == 0.5
    assert list(tr.table().columns) == ["object", "on hunt list", "found in", "time (s)"]


def test_tracker_resets_between_photos():
    tr = ch.HuntTracker(hunt_list=["cup"], min_frames=2)
    tr.update([("cup", 0.9)], 0, "a.jpg")
    assert tr.update([("cup", 0.9)], 0, "b.jpg") == []
    assert tr.update([("cup", 0.9)], 0, "b.jpg") == ["cup"]
    assert tr.detections_by_source() == {"b.jpg": {"cup"}} and tr.done
    tr.update([("cup", 0.9)], 0, "c.jpg"); tr.update([("cup", 0.9)], 0, "c.jpg")
    assert tr.detections_by_source() == {"b.jpg": {"cup"}, "c.jpg": {"cup"}}
    assert tr.confirmed["cup"]["source"] == "b.jpg"


def _fake_frames(n, closed):
    try:
        for _ in range(n):
            yield np.zeros((120, 160, 3), np.uint8), "webcam"
    finally:
        closed.append(True)


def test_run_hunt_with_fake_camera_and_detector():
    closed = []
    def detect(frame):
        return [("cup", 0.8, (10, 10, 60, 80)), ("person", 0.9, (0, 0, 160, 120))]
    tr = ch.HuntTracker(hunt_list=["cup", "book"], min_frames=3)
    shown = []
    def display(img):
        shown.append(img.shape)
        return len(shown) < 5                             # "press Q" after 5 frames
    out = ch.run_hunt(detect, _fake_frames(100, closed), tr, time_limit=60, display=display)
    assert closed == [True]                                # camera released
    assert out["frames"] == 5 and shown[0] == (120, 160, 3)
    assert set(out["snapshots"]) == {"cup", "person"} and out["snapshots"]["cup"].ndim == 3
    assert set(tr.seen) == {"cup", "person"}


def test_run_hunt_stops_when_all_found_and_on_errors():
    closed = []
    tr = ch.HuntTracker(hunt_list=["cup"], min_frames=1)
    out = ch.run_hunt(lambda f: [("cup", 0.9, (0, 0, 5, 5))], _fake_frames(50, closed), tr)
    assert out["frames"] == 1 and closed == [True]
    closed2 = []
    def boom(frame):
        raise KeyError("model crashed")
    with pytest.raises(KeyError):
        ch.run_hunt(boom, _fake_frames(5, closed2), ch.HuntTracker())
    assert closed2 == [True]


def test_photo_frames_and_crop(tmp_path):
    from PIL import Image
    path = tmp_path / "p.jpg"
    Image.new("RGB", (40, 30), (255, 0, 0)).save(path)
    frames = list(ch.photo_frames([path], repeats=2))
    assert len(frames) == 2 and frames[0][1] == "p.jpg" and frames[0][0][0, 0, 2] > 200   # BGR: red last
    crop = ch.crop_box(frames[0][0], (10, 10, 20, 20), margin=0.5)
    assert crop.shape == (20, 20, 3)


def test_tries_and_grid_figures(tmp_path):
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image
    paths = []
    for i in range(7):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (30, 20), (i * 30, 0, 0)).save(p)
        paths.append(p)
    tries = [{"text": "t", "scores": np.arange(7.0), "rank": 7, "total": 0, "foul": False},
             {"text": "u", "scores": -np.arange(7.0), "rank": 1, "total": 13, "foul": False}]
    plt.close(ch.plot_tries(paths, 0, tries, title="x"))
    plt.close(ch.plot_numbered_grid(paths, ncols=4, highlight=2))


def test_run_hunt_with_a_camera_that_sends_nothing():
    closed = []
    out = ch.run_hunt(lambda f: [], _fake_frames(0, closed), ch.HuntTracker(), time_limit=60)
    assert out["frames"] == 0 and out["fps"] is None and closed == [True]


class _BlindCapture:
    """A fake cv2.VideoCapture that opens but never delivers a frame."""
    released = False

    def __init__(self, *args):
        pass

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        return False, None

    def release(self):
        _BlindCapture.released = True


def test_webcam_frames_raises_when_camera_sends_no_image(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", _BlindCapture)
    with pytest.raises(RuntimeError, match="no image"):
        ch.run_hunt(lambda f: [], ch.webcam_frames(0), ch.HuntTracker(), time_limit=60)
    assert _BlindCapture.released


def test_clean_entries_all_blank_gives_empty_round():
    assert ch.clean_entries(["", "   ", "\n"], 3) == []
    assert ch.clean_entries([("  ", "pos")], 8, key=lambda e: e[0]) == []


def test_best_try_and_first_words():
    assert ch.best_try([]) is None
    tries = [{"total": 5, "rank": 3}, {"total": 5, "rank": 2}, {"total": 0, "rank": 9}]
    assert ch.best_try(tries) == {"total": 5, "rank": 2}
    assert ch.first_words("a b c d", 2) == "a b"
    assert ch.first_words("a b", 5) == "a b"
    long_text = " ".join(["word"] * 100)
    assert ch.word_count(ch.first_words(long_text)) == ch.MAX_DESCRIPTION_WORDS


def test_flip_html_styles_have_no_double_quotes_inside():
    # a double quote inside style="…" would end the attribute and silently drop the whole style
    assert '"' not in ch._FONT and '"' not in ch._MONO

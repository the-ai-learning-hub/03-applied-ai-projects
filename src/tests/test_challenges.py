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


@pytest.mark.parametrize("pred_a,pred_b,total", [("neg", "neg", 0), ("pos", "neg", 2), ("pos", "pos", 7)])
def test_judge_points(pred_a, pred_b, total):
    assert ch.judge_points("neg", pred_a, pred_b)["total"] == total


def test_judge_points_rejects_bad_label():
    with pytest.raises(ValueError):
        ch.judge_points("negative", "pos", "pos")


def test_spam_points():
    assert ch.spam_points(0.3, True)["total"] == 3
    assert ch.spam_points(0.05, True)["total"] == 5          # stealth bonus
    assert ch.spam_points(0.05, False)["total"] == 0         # not spam any more: no points
    assert ch.spam_points(0.9, True)["total"] == 0           # caught


def test_term_contributions_sign_and_order():
    texts = ["free prize now", "free money call", "see you at lunch", "lunch at home"]
    labels = ["spam", "spam", "ham", "ham"]
    vec = TfidfVectorizer()
    model = LogisticRegression(C=10).fit(vec.fit_transform(texts), labels)
    table = ch.term_contributions("free lunch", vec, model, positive_class="spam")
    assert set(table["term"]) == {"free", "lunch"}
    contrib = dict(zip(table["term"], table["contribution"]))
    assert contrib["free"] > 0 > contrib["lunch"]
    flipped = ch.term_contributions("free lunch", vec, model, positive_class="ham")
    assert np.isclose(dict(zip(flipped["term"], flipped["contribution"]))["free"], -contrib["free"])
    assert ch.term_contributions("zzz", vec, model).empty


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


def test_board_and_pushes_and_tries_figures(tmp_path):
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image
    fig = ch.plot_board(["s1", "s2"], [[("FOOLED", "fooled")], [("caught", "caught")]], ["judge"], [2, 0], "t")
    plt.close(fig)
    table = pd.DataFrame({"term": ["a", "b"], "tfidf": [1, 1], "weight": [1, -1], "contribution": [1.0, -1.0]})
    fig = ch.plot_term_pushes([table, table.iloc[:0]], ["x", "y"], "pos", "neg", ("blue", "red"))
    plt.close(fig)
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

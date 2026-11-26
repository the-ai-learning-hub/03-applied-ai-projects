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
    fig = ch.plot_scorecard(27, {"a": 20, "b": 7, "c": 0}, ch.LEVELS["clip_pictionary"], "Test", subtitle="x")
    assert fig.axes
    plt.close(fig)


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


# ---- Challenge 2 · YOLO Live (no model: a fake detector stands in for YOLO) ----

def test_count_classes_and_text():
    dets = [("car", 0.9, (0, 0, 1, 1)), ("person", 0.8, (0, 0, 1, 1)), ("car", 0.5, (0, 0, 1, 1)),
            ("bus", 0.2, (0, 0, 1, 1))]
    assert ch.count_classes(dets) == {"car": 2, "bus": 1, "person": 1}
    assert list(ch.count_classes(dets, 0.3)) == ["car", "person"]
    assert ch.counts_text({"car": 2, "person": 1}) == "2 car  |  1 person"
    assert ch.counts_text({"a": 1, "b": 1, "c": 1}, max_items=2).endswith("+1 more")
    assert ch.counts_text({}) == "nothing found"


def test_box_iou():
    assert ch.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert ch.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert ch.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert ch.box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_simple_tracker_keeps_ids_and_trails():
    tr = ch.SimpleTracker(iou_min=0.3, max_missed=2, trail=3)
    ids = []
    for k in range(5):                                       # a car moving right, a person standing still
        out = tr.update([("car", 0.9, (10 + 4 * k, 10, 40 + 4 * k, 30)), ("person", 0.8, (100, 0, 120, 60))])
        ids.append({cls: tid for tid, cls, _, _ in out})
    assert all(i == ids[0] for i in ids) and len(set(ids[0].values())) == 2
    car = tr.tracks[ids[0]["car"]]
    assert len(car["trail"]) == 3 and car["trail"][-1] == (41.0, 20.0)   # last 3 centres only
    assert tr.unique_counts(min_hits=5) == {"car": 1, "person": 1}
    assert tr.unique_counts(min_hits=6) == {}


def test_simple_tracker_never_mixes_classes_and_closes_lost_tracks():
    tr = ch.SimpleTracker(iou_min=0.3, max_missed=1)
    (a,) = tr.update([("car", 0.9, (0, 0, 10, 10))])
    (b,) = tr.update([("truck", 0.9, (0, 0, 10, 10))])       # same place, other class: a new ID
    assert a[0] != b[0]
    tr.update([])
    tr.update([])                                            # missed twice > max_missed: closed
    assert tr.tracks == {}
    (c,) = tr.update([("car", 0.9, (0, 0, 10, 10))])
    assert c[0] not in (a[0], b[0]) and len(tr.history) == 3


def test_heatmap_accumulates_at_box_centres():
    heat = np.zeros((100, 200), np.float32)
    ch.add_to_heatmap(heat, [(80, 40, 120, 60)])
    ch.add_to_heatmap(heat, [(80, 40, 120, 60), (500, 500, 600, 600)])   # a box outside the frame is ignored
    assert heat[50, 100] == 2 and heat[0, 0] == 0 and heat.sum() > 0
    frame = np.full((100, 200, 3), 200, np.uint8)
    over = ch.heatmap_overlay(frame, heat)
    assert over.shape == frame.shape and over.dtype == np.uint8 and over[0, 0, 0] < 200   # background darkened
    assert (ch.heatmap_overlay(frame, np.zeros((100, 200), np.float32)) == frame).all()


def test_count_timeline():
    df = ch.count_timeline([{"car": 2}, {}, {"car": 1, "person": 3}], [0.0, 0.04, 0.08])
    assert list(df.columns) == ["car", "person"] and df.index.name == "time (s)"
    assert df.loc[0.04].tolist() == [0, 0] and df["person"].sum() == 3
    assert ch.count_timeline([], []).empty


def test_class_colors_are_stable():
    colors = ch.ClassColors()
    first = colors["car"]
    assert colors["person"] != first and colors["car"] == first
    assert ch.hex_to_bgr("#E69F00") == (0, 159, 230) and colors.bgr("car") == ch.hex_to_bgr(first)


def test_draw_detections_and_tracks():
    frame = np.zeros((120, 160, 3), np.uint8)
    dets = [("cup", 0.9, (10, 40, 60, 100)), ("book", 0.1, (70, 40, 150, 110))]
    img = ch.draw_detections(frame, dets, min_conf=0.5)
    assert img.shape == frame.shape and frame.sum() == 0 and img.sum() > 0      # a new image, input untouched
    assert img[100, 10].any() and not img[110, 150].any()                       # below threshold: not drawn
    tr = ch.SimpleTracker()
    tracks = tr.update(dets)
    assert ch.draw_tracks(frame, tracks, tr, ch.ClassColors()).shape == frame.shape


def test_detections_from_result_with_a_fake_result():
    class _T(list):
        def tolist(self):
            return list(self)

    class _Boxes:
        cls, conf, xyxy = _T([2.0, 0.0]), _T([0.8, 0.6]), _T([[1, 2, 3, 4], [5, 6, 7, 8]])

    class _Result:
        names, boxes = {0: "person", 2: "car"}, _Boxes()

    assert ch.detections_from_result(_Result()) == [("car", 0.8, (1.0, 2.0, 3.0, 4.0)),
                                                    ("person", 0.6, (5.0, 6.0, 7.0, 8.0))]


def _fake_stream(n, closed, size=(120, 160)):
    try:
        for k in range(n):
            yield np.zeros((*size, 3), np.uint8), k / 10
    finally:
        closed.append(True)


def _moving_car(frame):
    return [("car", 0.9, (10, 10, 50, 40)), ("ghost", 0.1, (0, 0, 5, 5))]   # the ghost is below min_conf


def test_run_stream_collects_everything_and_closes_the_source():
    closed = []
    out = ch.run_stream(_moving_car, _fake_stream(10, closed), min_conf=0.3, keep_every=3, keep_width=80)
    assert closed == [True] and out["frames"] == 10
    assert len(out["kept"]) == 4 and out["kept"][0].shape == (60, 80, 3)        # frames 0, 3, 6, 9, resized
    assert list(out["timeline"].columns) == ["car"] and len(out["timeline"]) == 10
    assert out["unique"] == {"car": 1} and out["discovered"] == {"car": 0.0}
    assert out["heat"].shape == (120, 160) and len(out["latency_ms"]) == 10 and out["fps"] > 0


def test_run_stream_stops_on_display_time_limit_and_errors():
    closed = []
    shown = []
    out = ch.run_stream(_moving_car, _fake_stream(50, closed), display=lambda img: shown.append(1) or len(shown) < 4)
    assert out["frames"] == 4 and closed == [True]                                # "press Q" after 4 frames
    closed2 = []
    out = ch.run_stream(_moving_car, _fake_stream(50, closed2), time_limit=0.25)
    assert out["frames"] == 3 and closed2 == [True]                               # t = 0.0, 0.1, 0.2
    closed3 = []

    def boom(frame):
        raise KeyError("model crashed")

    with pytest.raises(KeyError):
        ch.run_stream(boom, _fake_stream(5, closed3))
    assert closed3 == [True]
    out = ch.run_stream(_moving_car, _fake_stream(0, []))
    assert out["frames"] == 0 and out["fps"] is None and out["heat"] is None


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
        ch.run_stream(_moving_car, ch.webcam_frames(0), time_limit=60)
    assert _BlindCapture.released


def test_video_frames_and_info_on_a_tiny_video(tmp_path):
    import cv2
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for k in range(12):
        writer.write(np.full((48, 64, 3), k * 20, np.uint8))
    writer.release()
    info = ch.video_info(path)
    assert info["width"] == 64 and info["height"] == 48 and info["fps"] == 10.0
    frames = list(ch.video_frames(path, width=32, max_seconds=0.5))
    assert len(frames) == 6 and frames[0][0].shape == (24, 32, 3) and frames[-1][1] == pytest.approx(0.5)
    with pytest.raises(RuntimeError):
        list(ch.video_frames(tmp_path / "missing.mp4"))


def test_list_media(tmp_path):
    for name in ("b.JPG", "a.png", "clip.mp4", "notes.txt", ".hidden.jpg", "README.md"):
        (tmp_path / name).write_bytes(b"x")
    images, videos = ch.list_media(tmp_path)
    assert [p.name for p in images] == ["a.png", "b.JPG"] and [p.name for p in videos] == ["clip.mp4"]
    assert ch.list_media(tmp_path / "nope") == ([], [])


def test_yolo_tables_are_well_formed():
    assert [r["size"] for r in ch.YOLO26_SIZES] == ["nano", "small", "medium", "large", "extra-large"]
    maps = [r["mAP 50-95"] for r in ch.YOLO26_SIZES]
    assert maps == sorted(maps)                               # bigger = more accurate ...
    cpu = [r["CPU ms"] for r in ch.YOLO26_SIZES]
    assert cpu == sorted(cpu)                                 # ... and slower
    assert {t["task"] for t in ch.YOLO_TASKS} == {"detect", "segment", "pose", "obb", "classify"}


def test_challenge2_figures_and_clip(tmp_path):
    import matplotlib.pyplot as plt
    frame = np.full((90, 160, 3), 120, np.uint8)
    dets = [("car", 0.9, (10, 10, 60, 50)), ("car", 0.2, (70, 10, 100, 40)), ("person", 0.6, (100, 20, 130, 80))]
    plt.close(ch.plot_detection_dashboard(frame, dets, 0.3, title="t"))
    plt.close(ch.plot_threshold_strip(frame, dets))
    out = ch.run_stream(_moving_car, _fake_stream(8, []), keep_every=1)
    plt.close(ch.plot_video_report(out, title="t"))
    assert ch.fig_to_jpeg(plt.figure())[:2] == b"\xff\xd8"
    html = ch.clip_html(out["kept"], 10, tmp_path)
    assert html.startswith("<video") or html.startswith("<img")
    assert ch.threshold_explorer(frame, dets).children[1].value[:2] == b"\xff\xd8"

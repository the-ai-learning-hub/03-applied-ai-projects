# 🎮 Challenge 2 · YOLO Scavenger Hunt

**How many objects can YOLO find on your desk in 60 seconds?** A real-time object detector (YOLO26 nano) runs **live on your webcam**, on a normal laptop CPU, with boxes, a hunt checklist, a timer and the FPS on screen. ⏱️ **≈ 15 min** · you play alone, on your own laptop.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/02_computer_vision/03_challenge_yolo/Challenge_2_YOLO_Scavenger_Hunt.ipynb) · 🏠 [Computer Vision](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#challenge-2)

## Rules

Hunt list: `cup` · `bottle` · `cell phone` · `book` · `scissors` · `keyboard` · `mouse` · `laptop` · `backpack` · `potted plant` (all COCO classes; you can change the list).
An object is **confirmed** when YOLO sees it with confidence **≥ 0.35** in **3 frames in a row** (a photo counts as 3 identical frames).

| Rule | Points |
|---|---|
| each different object class confirmed (person, cup, chair…) | +1 |
| each hunt-list item confirmed (on top of the +1) | +3 |
| **Fool YOLO**: a confirmed detection of something not really there (a drawing, a photo on a screen, a reflection…), self-declared in `FOOLED` | +5 (max 3 claims) |
| webcam game length | 60 s (press **Q** to stop early; it also stops when the whole list is found) |

**Medals:** 🥉 Bronze ≥ 10 · 🥈 Silver ≥ 25 · 🥇 Gold ≥ 40. The 3 example photos score **23 (Bronze)**.

## How to run

1. **Locally (webcam, recommended):** `uv sync` once, open the notebook with the `.venv` kernel and *Run All* (it plays on the example photos). Then in the ✏️ **YOUR TURN** cell set `LIVE = True` and run the cells from there.
   - `DISPLAY = "window"` opens an OpenCV window (smoothest; click it and press **Q** to stop).
   - `DISPLAY = "inline"` shows the video inside the notebook (for VS Code setups where a window is awkward; stop with *Interrupt*).
   - Wrong camera? Try `CAMERA_INDEX = 1`. No camera at all? The game switches to photo mode by itself.
2. **Photo mode (no webcam):** take photos of your desk, room or bag with your phone and copy them into [`my_photos/`](my_photos/). Keep `LIVE = False`.
3. **Google Colab** (if your setup fails): Python in Colab cannot read the webcam live, so use photo mode: in section 3 set `TAKE_PHOTOS = 3` (webcam snapshots in the browser) or `UPLOAD = True`. ⚠️ In Colab each snapshot or upload is **sent to Google's hosted runtime** and saved there as a JPEG in `my_photos/`; it is deleted when the runtime ends (or delete it yourself from the Files panel).

**Runtime:** ≈ 15 s for the whole notebook in photo mode; the webcam game runs at ≈ 20–25 FPS on a recent laptop CPU (640×480, inference at 480 px). First run: + download of the weights (≈ 5 MB) into `data/_cache/yolo/`.

**Privacy:** in **local live mode** webcam frames are processed on your laptop and kept in memory only (nothing is uploaded or saved to disk). Photo mode reads the files you put in `my_photos/`; in Colab, snapshots and uploads live on Google's runtime (see above).

## Files

- [`my_photos/`](my_photos/): 3 example photos so the notebook runs out of the box (two photos of our pets, and a composite: our cats on a drawn laptop screen, the "Fool YOLO" example). Add your own photos here; please do not commit personal photos.
- Scoring rules and the game loop: [`src/challenges.py`](../../src/challenges.py) (`CH2_RULES`, `HuntTracker`, `run_hunt`).

## Licence note

The `ultralytics` package and the YOLO26 weights are licensed under **AGPL-3.0** ([Ultralytics licensing](https://www.ultralytics.com/license)): free for learning and open-source work; a closed-source commercial product needs an Ultralytics Enterprise licence.

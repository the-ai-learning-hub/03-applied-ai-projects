# 🎮 Challenge 2 · YOLO Live

**🖼️ Images → 🎬 Video → 📷 Your webcam.** A real-time object detector (YOLO26 nano, 2.4 M parameters, a 5 MB file) on a normal laptop CPU: boxes, masks and skeletons on photos, tracking with IDs, trails, counts and a heatmap on a video, and finally your own webcam, live, with an FPS meter.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/02_computer_vision/03_challenge_yolo/Challenge_2_YOLO_Live.ipynb) · 🏠 [Computer Vision](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#challenge-2)

## The three stages

| Stage | What YOLO gets | What you see | Try this |
|---|---|---|---|
| 🧠 **Meet YOLO** | — | the five sizes (n, s, m, l, x: accuracy vs speed, from the [Ultralytics docs](https://docs.ultralytics.com/models/yolo26/)) and the five tasks (detect, segment, pose, OBB, classify) | choose `MODEL` (`yolo26n.pt` by default; try `yolo26s.pt`) |
| 🖼️ **1 · Images** | 8 test photos + yours | a gallery with counts, a "count the objects" dashboard, a **threshold slider**, detect vs **segment** vs **pose** on one photo, and what YOLO gets wrong (a kangaroo, tiny cars at 640 vs 1280 px) | find the threshold where the boxes are "just right"; catch YOLO making a mistake |
| 🎬 **2 · Video** | a 12 s traffic video, a street full of people, or yours | the annotated video with **track IDs, motion trails** and an FPS meter, a **heatmap** of where objects appear and the **count over time** per class | count the different cars; explain why the tracker counts too many |
| 📷 **3 · Webcam** | you, live, 30 s | the same HUD on your webcam: boxes, IDs, classes seen so far, FPS and latency | watch the boxes, track IDs and frame rate update as objects move |

## How to run

1. **Locally (recommended, the webcam runs smoothly):** `uv sync` once, open the notebook with the `.venv` kernel and *Run All*. Then in section 4 set `LIVE = True` and run the last cells.
   - `DISPLAY = "window"` opens an OpenCV window (click it and press **Q** to stop); `DISPLAY = "inline"` plays inside the notebook (stop with *Interrupt*).
   - Wrong camera? Try `CAMERA_INDEX = 1`. No camera? It replays the street video instead.
2. **Your own photos and videos:** copy them into [`my_media/`](my_media/) (everything there except its README is ignored by git). The image gallery includes your photos. Choose one of your videos in section 3 with `VIDEO = my_videos[0]`.
3. **Google Colab:** *Run All* works the same. For your files set `UPLOAD = True` in the upload cell. For the webcam set `LIVE = True`: the camera runs **in your browser** (allow it), each frame goes to Colab and back, so expect only a few FPS. ⚠️ In Colab, uploads and webcam frames are **sent to Google's hosted runtime** (uploads are stored there until the runtime ends).

**Runtime (laptop CPU, `yolo26n`):** ≈ 45 s for the whole notebook; ≈ 20–35 ms per image, so ≈ 25–40 FPS on a 640–960 px video (measured on the authors' laptop: 28 FPS on the traffic video, YOLO alone 20 ms per frame). First run: + the downloads below.

**Downloads (once):** model weights into `data/_cache/yolo/`: `yolo26n.pt` 5.5 MB, `yolo26n-seg.pt` 6.7 MB, `yolo26n-pose.pt` 7.9 MB (`yolo26s.pt` ≈ 20 MB if you try it). The two test videos (≈ 4.2 MB) into `data/yolo_media/` from the lesson's data release.

**Privacy:** in **local** live mode the webcam frames stay in your laptop's memory (nothing is saved or uploaded). Do not film or upload other people without their permission, and do not commit personal photos.

## Files

- [`Challenge_2_YOLO_Live.ipynb`](Challenge_2_YOLO_Live.ipynb): the notebook.
- [`media/`](media/): the 8 test photos (see the credits below).
- [`my_media/`](my_media/): your own photos and videos (git-ignored).
- The machinery (drawing, the simple IoU tracker, heatmap, video loop, webcam in OpenCV and in Colab, the threshold slider): [`src/challenges.py`](../../src/challenges.py) (`run_stream`, `SimpleTracker`, `draw_detections`, `plot_video_report`, `ColabWebcam`…), tested in `src/tests/test_challenges.py`.

<a id="credits"></a>
## Credits and licences of the test media

All test media come from [Wikimedia Commons](https://commons.wikimedia.org/) under open licences. We resized the photos (≤ 1280 px, JPEG) and cut and re-encoded the videos (12 s, 1280 × 720, H.264, no sound); these are the only changes. Material under CC BY-SA keeps its licence (share-alike).

**Photos** (`media/`, committed):

| File | Original on Wikimedia Commons | Author | Licence |
|---|---|---|---|
| `street_london.jpg` | [Busy Busy Regent Street](https://commons.wikimedia.org/wiki/File:Busy_Busy_Regent_Street.jpg) | Soham Banerjee | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) |
| `dinner_table.jpg` | [Formal Place Setting](https://commons.wikimedia.org/wiki/File:Formal_Place_Setting.jpg) | Hopefulromntic | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) |
| `desk.jpg` | [Desk Setup (Unsplash)](https://commons.wikimedia.org/wiki/File:Desk_Setup_(Unsplash).jpg) | Luke Chesser | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `market_singapore.jpg` | [Sungei Road Thieves Market Singapore](https://commons.wikimedia.org/wiki/File:Sungei_Road_Thieves_Market_Singapore._(8069648206).jpg) | Bernard Spragg. NZ | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `elephants_amboseli.jpg` | [Elephants Herd Amboseli](https://commons.wikimedia.org/wiki/File:Elephants_Herd_Amboseli_(7234335792).jpg) | Benh LIEU SONG | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0) |
| `skateboarder_brussels.jpg` | [Skateboarder jumping off a ledge at Skatepark des Ursulines in Brussels](https://commons.wikimedia.org/wiki/File:Skateboarder_jumping_off_a_ledge_at_Skatepark_des_Ursulines_in_Brussels_(DSCF4486).jpg) | Trougnouf | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `parking_lot_tel_aviv.jpg` | [Tel Aviv parking lot](https://commons.wikimedia.org/wiki/File:Tel_Aviv_parking_lot.jpg) | Dolev | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) |
| `kangaroos.jpg` | [Eastern Grey kangaroo, Majura Nature Reserve ACT 02](https://commons.wikimedia.org/wiki/File:Eastern_Grey_kangaroo,_Majura_Nature_Reserve_ACT_02.jpg) | Thennicke | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |

**Videos** (`data/yolo_media/`, downloaded from the data release):

| File | Original on Wikimedia Commons | Author | Licence |
|---|---|---|---|
| `traffic_chiang_mai.mp4` (first 12 s) | [A intersection with traffic in Chiang Mai, Thailand](https://commons.wikimedia.org/wiki/File:A_intersection_with_traffic_in_Chiang_Mai,_Thailand.ogv) | Amada44 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `pedestrians_madrid.mp4` (12 s from 0:03) | [Calle preciados](https://commons.wikimedia.org/wiki/File:Calle_preciados.ogv) | Osvaldo Gago (fotografar.net) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) |

## Licence note

The `ultralytics` package and the YOLO26 weights are licensed under **AGPL-3.0** ([Ultralytics licensing](https://www.ultralytics.com/license)): free for learning and open-source work; a closed-source commercial product needs an Ultralytics Enterprise licence.

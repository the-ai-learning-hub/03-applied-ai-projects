# 📖 Data dictionary · Lesson 03

Every dataset used in Lesson 03: what it is, where it comes from, its licence, its columns and its known issues.
The datasets are not stored in git: each one is a zip file in the GitHub release [`data-v1`](https://github.com/the-ai-learning-hub/03-applied-ai-projects/releases/tag/data-v1), downloaded and unzipped into this folder the first time a notebook needs it (`src/paths.py`). Other downloads go to `data/_cache/`. Everything except this file and `pets/` (the test pets, committed) is git-ignored.

**Contents**

- [SMS Spam Collection (`sms_spam/`)](#sms-spam-collection-sms_spam)
- [English tweets with sentiment (`tweets/`)](#english-tweets-with-sentiment-tweets)
- [Sentiment lexicons (`lexicons/`)](#sentiment-lexicons-lexicons)
- [Cats and dogs photos (`cats_and_dogs/`)](#cats-and-dogs-photos-cats_and_dogs)
- [Test pets (`pets/`)](#test-pets-pets)
- [YOLO test media (`yolo_media/` + `03_challenge_yolo/media/`)](#yolo-test-media-yolo_media--03_challenge_yolomedia)
- [Fashion-MNIST (downloaded at runtime)](#fashion-mnist-downloaded-at-runtime--_cachefashionmnist)

---

## SMS Spam Collection (`sms_spam/`)

| | |
|---|---|
| **What** | 5,574 real English SMS messages, each tagged as `ham` (normal) or `spam` |
| **Source** | Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CC84> · <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> |
| **Licence** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **File** | `sms_spam/sms_spam.csv` (UTF-8, comma-separated, with header). Converted from the original tab-separated `SMSSpamCollection` file without changing any message |
| **One row =** | one SMS message |
| **Used in** | `01_text_mining/01_mini_project_sms_spam/01_Text_Mining_Basics_SMS_Spam.ipynb` |

| Column | Type | Description |
|---|---|---|
| `label` | text | `ham` (4,827 · 86.6%) or `spam` (747 · 13.4%) — the target |
| `text` | text | the raw message (2 to 910 characters, median 62) |

**Known issues:** 403 messages are exact duplicates of another message (mostly short ham like "Sorry, I'll call later" and repeated spam campaigns); we keep them, but split train/test *after* removing duplicates so the same text is never in both. Messages are informal (abbreviations such as "u", "2", "lar") and some contain HTML entities (`&lt;#&gt;`).

---

## English tweets with sentiment (`tweets/`)

| | |
|---|---|
| **What** | Short English tweets (2009), each labelled with its sentiment: `pos` or `neg` |
| **Source** | Provided in the Text Mining course at ISCTE – University Institute of Lisbon |
| **Files** | `tweets/tweets_train.csv` (39,936 rows) and `tweets/tweets_test.csv` (9,985 rows), UTF-8 with header |
| **One row =** | one tweet |
| **Used in** | `01_text_mining/02_project_tweets/02.0`–`02.4_*.ipynb` and `01_text_mining/03_challenge_flip_it/Challenge_1_Flip_It.ipynb` (training data of the Flip It! model; `tweets_test.csv` only for its macro-F1) |

| Column | Type | Description |
|---|---|---|
| `tweet` | integer | tweet identifier (unique; no id appears in both files) |
| `text` | text | the tweet text. User mentions were already anonymised as `USER` and links as `URL` |
| `class` | text | `pos` (≈ 83%) or `neg` (≈ 17%) — the target (train 82.8% pos, test 83.2% pos) |

**Known issues**

- **Empty texts:** 237 train and 60 test rows have an empty (or whitespace-only) `text`.
- **Duplicates:** 616 train and 112 test texts are repeated inside the same file (identical text, sometimes with different labels).
- **Train/test overlap:** 281 test texts (compared after lower-casing and trimming, empty ones included; 221 non-empty) also appear in the training file. A model can "remember" them, so the notebook removes them from the test set before evaluating.
- **Imbalance:** always answering `pos` already gives ≈ 83% accuracy — compare models with macro-F1 and the recall of `neg`.
- Informal language, sarcasm, HTML entities (`&lt;`, `&amp;`), emoticons and labels that are sometimes debatable.

---

## Sentiment lexicons (`lexicons/`)

Word lists where each word has a sentiment score. Used in notebooks 01 and 02.2 to score text word by word.

| File | Lexicon | Format | Licence |
|---|---|---|---|
| `lexicons/vader_lexicon.txt` | **VADER** (Hutto & Gilbert, 2014), 7,520 entries incl. emoticons and slang. <https://github.com/cjhutto/vaderSentiment> | tab-separated: `token`, mean score (−4 to +4), standard deviation, the 10 raw human ratings | [MIT](https://github.com/cjhutto/vaderSentiment/blob/master/LICENSE.txt) |
| `lexicons/afinn_lexicon.txt` | **AFINN-en-165** (Finn Årup Nielsen, 2011–2015), 3,382 English words. <https://github.com/fnielsen/afinn> | tab-separated: `word`, integer score (−5 to +5) | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) — © Finn Årup Nielsen |
| `lexicons/Portuguese-NRC-EmoLex.txt` | **NRC Word-Emotion Association Lexicon (EmoLex)** (Mohammad & Turney, 2013), 14,154 English words with 0/1 flags for 8 emotions and positive/negative, plus a Portuguese translation column (only the English words are used). <https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm> | tab-separated: `English Word`, anger … trust, `Portuguese Word` | Free for research and education; commercial use needs a licence from the NRC |

**Known issues:** lexicons know nothing about context (sarcasm, "not bad"), and cover only some of the words in a tweet: about 1 in 4 cleaned test tweets contains *no* VADER word at all (so it scores 0 and falls back to the tie rule).

---

## Cats and dogs photos (`cats_and_dogs/`)

| | |
|---|---|
| **What** | 3,000 colour photos of cats and dogs: 2,000 for training (1,000 cats + 1,000 dogs) and 1,000 for validation (500 + 500) |
| **Source** | A subset of the Kaggle competition *Dogs vs. Cats* (2013) — <https://www.kaggle.com/c/dogs-vs-cats> — whose 25,000 photos come from **Microsoft Research's Asirra** CAPTCHA project (photos provided by Petfinder.com). Elson, J., Douceur, J., Howell, J. & Saul, J. (2007). *Asirra: a CAPTCHA that exploits interest-aligned manual image categorization*. ACM CCS. Same file selection as TensorFlow's `cats_and_dogs_filtered` tutorial dataset |
| **Licence** | Kaggle competition data, distributed for research and education (no explicit open licence). Fine for this lesson; check the original terms before any commercial use |
| **Files** | `cats_and_dogs/train/{cats,dogs}/` (`cat.0.jpg` … `cat.999.jpg`, same for `dog.`) and `cats_and_dogs/validation/{cats,dogs}/` (`cat.2000.jpg` … `cat.2499.jpg`). JPEG, RGB, **resized so that the longest side is 256 px** (aspect ratio kept) to keep the repository small (43 MB) |
| **One file =** | one photo; the **label is the folder name** (`cats` → 0, `dogs` → 1) |
| **Used in** | `02_computer_vision/02_project_cats_vs_dogs/04_Cats_vs_Dogs_And_Our_Pets.ipynb` and `03_multimodal/02_challenge_clip_pictionary/Challenge_3_CLIP_Pictionary.ipynb` (19 validation photos on the Pictionary table) |

**Known issues:** photos are very varied (several animals, people, cages, text banners, drawings) and a few look mislabelled or contain both a cat and a dog. Sizes vary: the shortest side goes from 41 px to 256 px (median 192 px). The notebook always builds **explicit, sorted file lists** with their labels and never shuffles the evaluation data (a shuffled evaluation loader misaligns predictions and labels).

---

## Test pets (`pets/`)

| | |
|---|---|
| **What** | Photos of the teacher's own animals: 9 cats and 7 dogs, plus one extra cat (`cat10000.jpg`) that only exists in `cropped/` |
| **Source** | Taken by the teacher at home with a phone, used **with permission** for this lesson. **`dog9016AI.jpg` is not a real photo**: it was generated by an image AI (a dog with a kitten on its head) and is kept on purpose as a discussion case |
| **Licence** | Shared with the owner's permission for this lesson only; not under an open licence — please do not reuse the photos outside the lesson |
| **Files** | `pets/original/{cats,dogs}/` (16 full photos, up to 1024 px) and `pets/cropped/{cats,dogs}/` (17 photos cut around the animal, roughly square). JPEG, RGB. Names: `cat9001.jpg`, `dog9006.jpg`, … — the same name in both folders is the same photo |
| **One file =** | one photo; the label is the folder name |
| **Used in** | `02_computer_vision/02_project_cats_vs_dogs/04_Cats_vs_Dogs_And_Our_Pets.ipynb` (predictions, original vs cropped, Grad-CAM) and `03_multimodal/01_project_everything_is_a_vector/05_Everything_Is_A_Vector.ipynb` (CLIP), plus the challenges: `03_multimodal/02_challenge_clip_pictionary/` (all 17 cropped photos on the Pictionary table) |

**Known issues:** only 17 photos — enough for a live demo, far too few to rank models (one photo = 6 percentage points). Phone photos may carry an EXIF rotation flag; `src.vision.load_image` applies it. The photos are a *different source* from the training data on purpose (the real test of a model).

---

## YOLO test media (`yolo_media/` + `03_challenge_yolo/media/`)

| | |
|---|---|
| **What** | Test media for Challenge 2 (YOLO Live): **8 photos** chosen for object detection (a busy street, a laid table, a desk, a market, elephants, a skatepark, an aerial parking lot, kangaroos) and **2 short videos** (traffic at a crossroads, people walking in a shopping street) |
| **Source** | Wikimedia Commons; author, link and licence of every file in the [challenge README](../02_computer_vision/03_challenge_yolo/README.md#credits) (and in `yolo_media/CREDITS.md`) |
| **Licence** | Open licences per file: CC0, CC BY 2.0 / 4.0, CC BY-SA 2.0 / 3.0 / 4.0 (both videos are CC BY-SA). Credit the authors when you reuse them; the BY-SA files keep their licence |
| **Files** | Photos (committed, ≈ 1.8 MB): `02_computer_vision/03_challenge_yolo/media/*.jpg`, resized to ≤ 1280 px. Videos (release zip `yolo_media.zip`, ≈ 4.2 MB → `data/yolo_media/`): `traffic_chiang_mai.mp4`, `pedestrians_madrid.mp4` (12 s each, 1280 × 720, 25 / 24 FPS, H.264, no sound) + `CREDITS.md` |
| **One file =** | one photo or one video; no labels (YOLO's pretrained COCO classes are the "labels") |
| **Used in** | `02_computer_vision/03_challenge_yolo/Challenge_2_YOLO_Live.ipynb` (`data_dir("yolo_media")`; the videos are downloaded only by this notebook) |

**Known issues:** some subjects are **not** COCO classes on purpose (kangaroos → YOLO says *sheep*; an iMac → *tv*), and the parking lot is an aerial photo with tiny cars (most are missed at 640 px). The street video is handheld, so the heatmap of the pedestrian video is blurrier than the traffic one (fixed camera). People in the photos and videos are in public places; the files are published on Commons with these licences.

---

## Fashion-MNIST (downloaded at runtime → `_cache/FashionMNIST/`)

| | |
|---|---|
| **What** | 70,000 grayscale 28 × 28 images of clothing from Zalando's online shop, 10 balanced classes (60,000 train / 10,000 test) |
| **Source** | Xiao, H., Rasul, K. & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747. <https://github.com/zalandoresearch/fashion-mnist> |
| **Licence** | [MIT](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE) — © 2017 Zalando SE |
| **Files** | not committed: `torchvision.datasets.FashionMNIST(root="data/_cache", download=True)` downloads ≈ 30 MB on the first run (git-ignored) |
| **One row =** | one image (28 × 28 pixels, 0 = black … 255 = white) + its label |
| **Used in** | `02_computer_vision/01_mini_project_fashion_mnist/03_Computer_Vision_Basics_Fashion_MNIST.ipynb` |

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

**Known issues:** *Shirt*, *T-shirt/top*, *Pullover* and *Coat* overlap a lot (even for people), so they cause most of the errors; some labels are debatable. Images are small, centred, grey and on a black background — much easier than real shop photos.

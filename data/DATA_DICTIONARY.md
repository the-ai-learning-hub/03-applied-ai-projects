# 📖 Data dictionary · Lesson 03

Every dataset used in Lesson 03: what it is, where it comes from, its licence, its columns and its known issues.
The datasets are not stored in git: each one is a zip file in the GitHub release [`data-v1`](https://github.com/the-ai-learning-hub/03-applied-ai-projects/releases/tag/data-v1), downloaded and unzipped into this folder the first time a notebook needs it (`src/paths.py`). Other downloads go to `data/_cache/`. Everything except this file and `pets/` (the test pets, committed) is git-ignored.

**Contents**

- [SMS Spam Collection (`sms_spam/`)](#sms-spam-collection-sms_spam)
- [English tweets with sentiment (`tweets/`)](#english-tweets-with-sentiment-tweets)
- [Sentiment lexicons (`lexicons/`)](#sentiment-lexicons-lexicons)
- [Cats and dogs photos (`cats_and_dogs/`)](#cats-and-dogs-photos-cats_and_dogs)
- [Test pets (`pets/`)](#test-pets-pets)
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
| **Used in** | `01_text_mining/01_mini_project_sms_spam/01_Text_Mining_Basics_SMS_Spam.ipynb` and `01_text_mining/03_challenge_fool_the_model/Challenge_1_Fool_The_Sentiment_Model.ipynb` (spam filter of Round 2) |

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
| **Used in** | `01_text_mining/02_project_tweets/02_Tweet_Sentiment_Analysis.ipynb` and `01_text_mining/03_challenge_fool_the_model/Challenge_1_Fool_The_Sentiment_Model.ipynb` (training data of judge A, `tweets_train.csv` only) |

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

Word lists where each word has a sentiment score. Used in notebook 02 to score tweets word by word.

| File | Lexicon | Format | Licence |
|---|---|---|---|
| `lexicons/vader_lexicon.txt` | **VADER** (Hutto & Gilbert, 2014), 7,520 entries incl. emoticons and slang. <https://github.com/cjhutto/vaderSentiment> | tab-separated: `token`, mean score (−4 to +4), standard deviation, the 10 raw human ratings | [MIT](https://github.com/cjhutto/vaderSentiment/blob/master/LICENSE.txt) |
| `lexicons/afinn_lexicon.txt` | **AFINN-en-165** (Finn Årup Nielsen, 2011–2015), 3,382 English words. <https://github.com/fnielsen/afinn> | tab-separated: `word`, integer score (−5 to +5) | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) — © Finn Årup Nielsen |

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
| **Used in** | `02_computer_vision/02_project_cats_vs_dogs/04_Cats_vs_Dogs_And_Our_Pets.ipynb` (predictions, original vs cropped, Grad-CAM) and `03_multimodal/01_project_everything_is_a_vector/05_Everything_Is_A_Vector.ipynb` (CLIP), plus the challenges: `03_multimodal/02_challenge_clip_pictionary/` (all 17 cropped photos on the Pictionary table) and `02_computer_vision/03_challenge_yolo/my_photos/` (example photos, see below) |

**Known issues:** only 17 photos — enough for a live demo, far too few to rank models (one photo = 6 percentage points). Phone photos may carry an EXIF rotation flag; `src.vision.load_image` applies it. The photos are a *different source* from the training data on purpose (the real test of a model).


**Example photos of Challenge 2 (`02_computer_vision/03_challenge_yolo/my_photos/`).** Three photos derived from this folder so the YOLO scavenger hunt runs out of the box (same permission and restrictions as above, resized to ≤ 1024 px):

| File | Made from | How |
|---|---|---|
| `example_cat_in_my_bag.jpg` | `original/cats/cat9009.jpg` | resized copy |
| `example_dog_and_plants.jpg` | `original/dogs/dog9008.jpg` | resized copy |
| `example_cats_on_a_screen.jpg` | `original/cats/cat9003.jpg` | composite made with Pillow for this lesson: the photo pasted on a hand-drawn laptop screen on a synthetic desk (the "Fool YOLO" example: the cats are only pixels on a screen). No downloaded image was used |

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

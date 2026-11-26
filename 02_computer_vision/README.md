# 02 · Computer Vision: from pixels to our own pets

An image is a grid of numbers. We go from one pixel to a model that recognises *our own* cats and dogs, see why convolutions work, race the CPU against a GPU, and ask the model where it looked. 🏠 [Lesson 03](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#computer-vision)

| Notebook | Question | Time | Open |
|---|---|---|---|
| [03 · Computer Vision basics: Fashion-MNIST](01_mini_project_fashion_mnist/03_Computer_Vision_Basics_Fashion_MNIST.ipynb) | An image is a grid of numbers: how does a model read it? | 60 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/02_computer_vision/01_mini_project_fashion_mnist/03_Computer_Vision_Basics_Fashion_MNIST.ipynb) |
| [04 · Cats vs dogs, and our pets](02_project_cats_vs_dogs/04_Cats_vs_Dogs_And_Our_Pets.ipynb) | From scratch or borrowed knowledge? And does it work on *our* animals? | 100 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/02_computer_vision/02_project_cats_vs_dogs/04_Cats_vs_Dogs_And_Our_Pets.ipynb) |
| [🎮 Challenge 2 · YOLO scavenger hunt](03_challenge_yolo/Challenge_2_YOLO_Scavenger_Hunt.ipynb) | How many objects can YOLO find on your desk? | 15 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/02_computer_vision/03_challenge_yolo/Challenge_2_YOLO_Scavenger_Hunt.ipynb) |

## 🎯 Objectives

- **Pixels are numbers:** read one Fashion-MNIST image as a 28 × 28 table, flatten it and train the MLP from Lesson 02; look at what its first layer learned.
- **Convolutions:** apply an edge detector *by hand* (on clothes and on our pets), see feature maps, and check that a small CNN beats the MLP with far fewer parameters.
- **Real photos:** train a CNN from scratch on 2,000 cat/dog photos, see overfitting and how **data augmentation** helps.
- **CPU vs GPU race:** the same model, the same batches, timed on both — seconds per epoch and images per second.
- **Transfer learning:** reuse a pretrained torchvision backbone (MobileNetV3), train only a new head in seconds, and compare in a leaderboard.
- **Our pets, the star:** predictions on the teacher's own animals, cropped vs original photos, an AI-generated dog, **Grad-CAM** heatmaps ("where did the model look?") and a cell to try **your own photo**.

## 📋 Prerequisites

Lesson 02 (train/validation split, confusion matrix, overfitting, MLP, PyTorch basics: tensors, `nn.Module`, AdamW) and the slides of the Computer Vision block.

## 📂 Data

| Folder | What | Size |
|---|---|---|
| `data/_cache/FashionMNIST/` | Fashion-MNIST, downloaded by `torchvision` on the first run (git-ignored) | ≈ 30 MB download, ≈ 80 MB on disk |
| `data/cats_and_dogs/` | 2,000 training + 1,000 validation photos (half cats, half dogs), resized to 256 px | 43 MB |
| `data/pets/original/`, `data/pets/cropped/` | our own pets: `original/` has 9 cats + 7 dogs (16 photos, one dog AI-generated); `cropped/` has the same plus one extra cat (`cat10000.jpg`), 17 photos | 4 MB |

Details, sources and licences: [`data/DATA_DICTIONARY.md`](../data/DATA_DICTIONARY.md).

## ⏱️ Runtime on a laptop CPU

Measured on the authors' laptop (8 CPU threads), downloads excluded. Colab's free CPU is roughly 2× slower.

| Notebook | First run | Later runs |
|---|---|---|
| 03 · Fashion-MNIST | ≈ 1–2 min | ≈ 1–2 min |
| 04 · Cats vs dogs | ≈ 3 min (images and pretrained features are computed once) | ≈ 1.5–2 min (cached in `data/_cache/`) |

Heavy steps (30-epoch trainings, the four-backbone leaderboard) are behind `RUN_HEAVY = False` and use results saved in [`02_project_cats_vs_dogs/results/`](02_project_cats_vs_dogs/results/). Set `RUN_HEAVY = True` on a GPU to recompute them.

## 📦 Pretrained models (downloaded on first run)

| Model | Used for | Size |
|---|---|---|
| [MobileNetV3-Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html) (`IMAGENET1K_V2`) | frozen feature extractor (960 numbers per photo) + Grad-CAM | ≈ 21 MB |
| MobileNetV3-Small, EfficientNet-B0, ResNet18 | only with `RUN_HEAVY = True` (leaderboard of backbones) | 10–45 MB |

Weights go to the PyTorch cache (`~/.cache/torch/hub/checkpoints`). Trained models are saved in `models/` (git-ignored).

## GPU (optional)

Both notebooks run on CPU. Notebook 04 includes a **CPU vs GPU race**: with an NVIDIA GPU (CUDA) or Apple Silicon (MPS) it times both live; without one it shows the numbers measured on an RTX 4060 Laptop GPU ([`02_project_cats_vs_dogs/results/cpu_vs_gpu.json`](02_project_cats_vs_dogs/results/cpu_vs_gpu.json)).

- **Colab:** *Runtime → Change runtime type → T4 GPU*, then re-run.
- **Locally (NVIDIA):** the project installs the CPU build of PyTorch. To try your GPU: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` (go back with `uv sync`).

## 📸 Your own photo

Put any JPG/PNG in `data/pets/` (or anywhere) and call `predict_photo("path/to/photo.jpg")` in the last section of notebook 04, or use the upload button there.

"""Computer Vision helpers for Lesson 03 (notebooks 03 and 04).

Everything here is small and readable on purpose: the notebooks explain the ideas, these
functions only remove repetition. Design rules:

* **Explicit, sorted file lists** for every dataset. Evaluation never uses a shuffled loader:
  shuffling the evaluation data is how predictions and labels get silently misaligned.
  :func:`predict_proba` refuses a shuffled ``DataLoader``.
* Images are cached once as ``uint8`` tensors (small, fast); they become ``float`` only per batch.
* CPU first. Every function takes a ``device`` and works the same on ``"cpu"``, ``"cuda"`` or ``"mps"``.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler

#: Class order used everywhere for the binary task (label 0 = cat, label 1 = dog).
CLASSES = ("cat", "dog")
#: Folder name for each class inside ``cats_and_dogs/<split>/`` and ``pets/<version>/``.
CLASS_FOLDERS = ("cats", "dogs")
#: Fashion-MNIST label names (index = label).
FASHION_CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
#: ImageNet statistics used by every torchvision pretrained backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

#: Hand-made 3×3 kernels for the "convolution by hand" demos.
KERNELS = {
    "vertical edges": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "horizontal edges": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "outline": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
    "blur": np.full((3, 3), 1 / 9, dtype=np.float32),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
}


# --------------------------------------------------------------------------------------------
# Reproducibility and devices
# --------------------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fix the Python, NumPy and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def available_devices() -> list[str]:
    """Devices PyTorch can use here, CPU first (e.g. ``["cpu", "cuda"]``)."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def best_device() -> str:
    """The fastest available device (GPU if there is one, else CPU)."""
    return available_devices()[-1]


def device_name(device: str) -> str:
    """Human-readable name of a device (GPU model, or the CPU description)."""
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple GPU (MPS)"
    return _cpu_name()


def _cpu_name() -> str:
    """Marketing name of the CPU (e.g. 'Intel(R) Core(TM) i7-...'), best effort on each OS."""
    import platform
    import subprocess
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001 - a name is nice to have, never required
        pass
    return platform.processor() or platform.machine() or "CPU"


def synchronize(device: str) -> None:
    """Wait until the device has finished its queued work (GPU calls are asynchronous)."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------------
# Files and images
# --------------------------------------------------------------------------------------------
def list_images(folder: str | Path) -> list[Path]:
    """All image files directly inside ``folder``, sorted by name (deterministic order)."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def labeled_files(root: str | Path, class_folders: Sequence[str] = CLASS_FOLDERS
                  ) -> tuple[list[Path], np.ndarray]:
    """Explicit ``(paths, labels)`` for ``root/<class_folder>/*.jpg``; label = folder index."""
    paths, labels = [], []
    for label, name in enumerate(class_folders):
        files = list_images(Path(root) / name)
        paths += files
        labels += [label] * len(files)
    return paths, np.array(labels, dtype=np.int64)


def cats_dogs_files(split: str = "train", root: str | Path | None = None
                    ) -> tuple[list[Path], np.ndarray]:
    """File list of ``data/cats_and_dogs/<split>`` (``"train"``: 2000, ``"validation"``: 1000)."""
    if root is None:
        from src.paths import data_dir
        root = data_dir() / "cats_and_dogs"
    return labeled_files(Path(root) / split)


def pets_files(version: str = "cropped", root: str | Path | None = None) -> list[dict]:
    """Our own pets: one dict per photo with ``path``, ``name``, ``label`` and ``ai_generated``.

    ``version`` is ``"cropped"`` (animal fills the frame) or ``"original"`` (full photo).
    """
    if root is None:
        from src.paths import data_dir
        root = data_dir() / "pets"
    paths, labels = labeled_files(Path(root) / version)
    return [{"path": p, "name": p.stem, "label": int(y), "ai_generated": "AI" in p.stem}
            for p, y in zip(paths, labels)]


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB, honouring the phone's rotation flag (EXIF)."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Resize so the *shorter* side is ``size``, then cut the central ``size × size`` square."""
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def pil_to_uint8(img: Image.Image) -> torch.Tensor:
    """PIL image → ``uint8`` tensor ``(3, H, W)``."""
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)


def load_images_tensor(paths: Iterable[str | Path], size: int = 128) -> torch.Tensor:
    """Load many images into one ``uint8`` tensor ``(N, 3, size, size)`` (order = ``paths``)."""
    return torch.stack([pil_to_uint8(resize_center_crop(load_image(p), size)) for p in paths])


def to_float(images: torch.Tensor) -> torch.Tensor:
    """``uint8`` 0–255 → ``float32`` 0–1 (works on one image or a batch)."""
    return images.float().div_(255.0) if images.dtype == torch.uint8 else images.float()


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Standardise a 0–1 float batch with the ImageNet mean/std (for pretrained backbones)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x - mean) / std


def to_display(img) -> np.ndarray:
    """Anything image-like (PIL, ``(C,H,W)`` tensor, ``(H,W[,C])`` array) → array for ``imshow``."""
    if isinstance(img, Image.Image):
        return np.asarray(img)
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = img.numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1) if img.max() <= 1.0 + 1e-6 else img
    return img


# --------------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------------
class TensorImageDataset(Dataset):
    """Images already in memory (``uint8`` ``(N,3,H,W)``) + labels, with an optional transform.

    The transform receives a float 0–1 image ``(3,H,W)`` (e.g. a ``torchvision.transforms.v2``
    pipeline). Labels are returned as ``float32`` for a single-logit (binary) model, else ``int64``.
    """

    def __init__(self, images: torch.Tensor, labels: Sequence[int], transform: Callable | None = None,
                 binary: bool = True):
        assert len(images) == len(labels), "images and labels must have the same length"
        self.images, self.transform = images, transform
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32 if binary else torch.int64)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        x = to_float(self.images[i])
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i]


class ImageFileDataset(Dataset):
    """Images read from an explicit list of files (``transform`` gets a PIL image)."""

    def __init__(self, paths: Sequence[str | Path], labels: Sequence[int] | None = None,
                 transform: Callable | None = None):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_image(self.paths[i])
        x = self.transform(img) if self.transform is not None else pil_to_uint8(img)
        y = -1 if self.labels is None else int(self.labels[i])
        return x, y


def load_fashion_mnist(root: str | Path | None = None, train: bool = True
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fashion-MNIST as ``uint8`` images ``(N, 28, 28)`` and ``int64`` labels (downloads once)."""
    from torchvision.datasets import FashionMNIST
    if root is None:
        from src.paths import cache_dir
        root = cache_dir()
    ds = FashionMNIST(root=str(root), train=train, download=True)
    return ds.data.clone(), ds.targets.clone()


# --------------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------------
def show_grid(images: Sequence, titles: Sequence[str] | None = None, ncols: int = 5,
              size: float = 2.2, cmap: str | None = "gray", suptitle: str | None = None,
              title_colors: Sequence[str] | None = None, axes=None):
    """Show images in a grid (no axes). Returns the figure."""
    import matplotlib.pyplot as plt
    n = len(images)
    if axes is None:
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.08),
                                 squeeze=False, layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    flat = np.ravel(axes)
    for k, ax in enumerate(flat):
        ax.axis("off")
        if k >= n:
            continue
        img = to_display(images[k])
        ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
        if titles is not None:
            colour = title_colors[k] if title_colors is not None else "#1a2433"
            ax.set_title(titles[k], fontsize=10, color=colour, loc="center", fontweight="normal")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontweight="bold", fontsize=13)
    return fig


def overlay_heatmap(img, heatmap: np.ndarray, alpha: float = 0.45, cmap: str = "inferno") -> np.ndarray:
    """Blend a 0–1 ``heatmap`` (any size) on top of an image; returns an RGB float array."""
    import matplotlib as mpl
    base = to_display(img).astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    h, w = base.shape[:2]
    hm = torch.as_tensor(heatmap, dtype=torch.float32)[None, None]
    hm = F.interpolate(hm, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    colour = mpl.colormaps[cmap](np.clip(hm, 0, 1))[..., :3]
    return (1 - alpha) * base + alpha * colour


def random_crop_flip(batch: torch.Tensor, size: int, flip: bool = True,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Batch augmentation: a random ``size × size`` crop and a random left-right flip per image.

    Fast (pure tensor slicing), so it can run inside the training loop on CPU or GPU.
    """
    n, _, h, w = batch.shape
    tops = torch.randint(0, h - size + 1, (n,), generator=generator).tolist()
    lefts = torch.randint(0, w - size + 1, (n,), generator=generator).tolist()
    out = torch.stack([batch[i, :, t:t + size, l:l + size] for i, (t, l) in enumerate(zip(tops, lefts))])
    if flip:
        mask = (torch.rand(n, generator=generator) < 0.5).to(batch.device)
        out[mask] = out[mask].flip(-1)
    return out


def center_crop_batch(batch: torch.Tensor, size: int) -> torch.Tensor:
    """The central ``size × size`` square of every image in a batch (the evaluation twin)."""
    h, w = batch.shape[-2:]
    t, l = (h - size) // 2, (w - size) // 2
    return batch[..., t:t + size, l:l + size]


def apply_kernel(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide a small ``kernel`` over a 2-D image (a convolution, exactly what a CNN layer does)."""
    x = torch.as_tensor(np.asarray(gray, dtype=np.float32))[None, None]
    k = torch.as_tensor(np.asarray(kernel, dtype=np.float32))[None, None]
    return F.conv2d(x, k, padding=k.shape[-1] // 2)[0, 0].numpy()


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """A small CNN: blocks of conv → batch-norm → ReLU → max-pool, then global average pooling.

    ``channels=(16, 32, 64, 128, 256)`` gives ≈ 390 k parameters (each block halves the size). With ``n_out=1`` the model returns
    one logit per image (binary: P(dog) = sigmoid(logit)).
    """

    def __init__(self, in_channels: int = 3, channels: Sequence[int] = (16, 32, 64, 128, 256),
                 n_out: int = 1, dropout: float = 0.2):
        super().__init__()
        layers, c_in = [], in_channels
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)          # global average pooling: one number per map
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.pool(self.features(x)))
        return out.squeeze(1) if out.shape[1] == 1 else out


def _loss_and_correct(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
    if logits.ndim == 1:                              # binary, one logit
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        correct = ((logits > 0).long() == y.long()).sum().item()
    else:
        loss = F.cross_entropy(logits, y.long())
        correct = (logits.argmax(1) == y.long()).sum().item()
    return loss, correct


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None = None,
                epochs: int = 5, lr: float = 1e-3, weight_decay: float = 1e-4, device: str = "cpu",
                verbose: bool = True, input_fn: Callable | None = None,
                augment_fn: Callable | None = None, cosine: bool = False) -> dict[str, list[float]]:
    """Train with AdamW and return the history per epoch.

    Works for binary (one logit, BCE) and multi-class (CE) models. History keys:
    ``loss_tr, loss_va, acc_tr, acc_va, seconds`` (``seconds`` = training time of the epoch).
    ``input_fn`` optionally transforms each batch on the device (e.g. flatten, normalise);
    ``augment_fn`` is applied to *training* batches only (e.g. :func:`random_crop_flip`).
    ``cosine=True`` lowers the learning rate smoothly to ~0 by the last epoch (steadier curves).
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
             if cosine else None)
    hist = {"loss_tr": [], "loss_va": [], "acc_tr": [], "acc_va": [], "seconds": []}
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n_ok, n = time.perf_counter(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if augment_fn is not None:
                x = augment_fn(x)
            if input_fn is not None:
                x = input_fn(x)
            opt.zero_grad()
            loss, correct = _loss_and_correct(model(x), y)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            total += loss.item() * len(y); n_ok += correct; n += len(y)
        synchronize(device)
        hist["seconds"].append(time.perf_counter() - t0)
        hist["loss_tr"].append(total / n); hist["acc_tr"].append(n_ok / n)
        if val_loader is not None:
            res = evaluate(model, val_loader, device=device, input_fn=input_fn)
            hist["loss_va"].append(res["loss"]); hist["acc_va"].append(res["accuracy"])
        if verbose:
            va = (f" | val loss {hist['loss_va'][-1]:.3f} acc {hist['acc_va'][-1]:.3f}"
                  if val_loader is not None else "")
            print(f"epoch {epoch:>2}/{epochs}: train loss {hist['loss_tr'][-1]:.3f} "
                  f"acc {hist['acc_tr'][-1]:.3f}{va} | {hist['seconds'][-1]:.1f} s")
    return hist


def _check_not_shuffled(loader: DataLoader) -> None:
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Evaluation loaders must not shuffle (shuffle=False): otherwise the "
                         "predictions no longer line up with the labels.")


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: str = "cpu",
                   input_fn: Callable | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Logits and labels in the loader's (fixed) order. Raises if the loader shuffles."""
    _check_not_shuffled(loader)
    model.to(device).eval()
    outs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        if input_fn is not None:
            x = input_fn(x)
        outs.append(model(x).float().cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def predict_proba(model: nn.Module, loader: DataLoader, device: str = "cpu",
                  input_fn: Callable | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities (P(class 1) for binary, softmax rows for multi-class) and labels, in order."""
    logits, y = predict_logits(model, loader, device, input_fn)
    if logits.ndim == 1:
        return 1 / (1 + np.exp(-logits)), y
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True), y


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu",
             input_fn: Callable | None = None) -> dict:
    """Loss, accuracy, predictions, probabilities and confusion matrix on a non-shuffled loader."""
    logits, y = predict_logits(model, loader, device, input_fn)
    t_logits, t_y = torch.from_numpy(logits), torch.from_numpy(y)
    loss, _ = _loss_and_correct(t_logits, t_y)
    if logits.ndim == 1:
        proba = 1 / (1 + np.exp(-logits)); pred = (logits > 0).astype(int); k = 2
    else:
        proba = torch.softmax(t_logits, 1).numpy(); pred = logits.argmax(1); k = logits.shape[1]
    y = y.astype(int)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return {"loss": float(loss), "accuracy": float((pred == y).mean()), "y_true": y,
            "y_pred": pred, "proba": proba, "confusion": cm}


# --------------------------------------------------------------------------------------------
# Transfer learning
# --------------------------------------------------------------------------------------------
def pretrained_backbone(name: str = "mobilenet_v3_large"):
    """A frozen torchvision backbone that outputs one feature vector per image.

    Returns ``(backbone, preprocess, n_features)``. ``preprocess`` is the official transform of
    the weights (resize, centre crop 224, ImageNet normalisation) and works on PIL images.
    """
    import torchvision.models as tvm
    weights = tvm.get_model_weights(name).DEFAULT
    model = tvm.get_model(name, weights=weights)
    if name.startswith("resnet"):
        n_features = model.fc.in_features
        model.fc = nn.Identity()
    elif name.startswith(("mobilenet", "efficientnet")):
        n_features = next(m for m in model.classifier if isinstance(m, nn.Linear)).in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"unsupported backbone: {name}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights.transforms(), n_features


@torch.no_grad()
def extract_features(backbone: nn.Module, loader: DataLoader, device: str = "cpu"
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Run every image once through a frozen backbone; returns ``(features, labels)`` in order.

    The backbone is moved to ``device`` for the extraction and then **back to where it was**, so
    later CPU code (one photo at a time, Grad-CAM) keeps working after a GPU extraction.
    """
    _check_not_shuffled(loader)
    home = next(backbone.parameters()).device
    backbone.to(device).eval()
    feats, ys = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device)).float().cpu())
        ys.append(torch.as_tensor(y))
    backbone.to(home)
    return torch.cat(feats).numpy(), torch.cat(ys).numpy()


def train_linear_head(f_train: np.ndarray, y_train: np.ndarray, f_val: np.ndarray | None = None,
                      y_val: np.ndarray | None = None, epochs: int = 60, lr: float = 1e-2,
                      weight_decay: float = 1e-2, batch_size: int = 128, seed: int = 42
                      ) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train a single ``nn.Linear`` (logistic regression) on precomputed features, in seconds.

    Mini-batches of ``batch_size``; history keys ``loss_tr, acc_tr, loss_va, acc_va, seconds``.
    """
    torch.manual_seed(seed)
    head = nn.Linear(f_train.shape[1], 1)
    xt, yt = torch.as_tensor(f_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32)
    train_dl = DataLoader(torch.utils.data.TensorDataset(xt, yt), batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    val_dl = None
    if f_val is not None:
        val_dl = DataLoader(torch.utils.data.TensorDataset(
            torch.as_tensor(f_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.float32)),
            batch_size=1024)
    model = _Squeeze(head)
    hist = train_model(model, train_dl, val_dl, epochs=epochs, lr=lr, weight_decay=weight_decay,
                       verbose=False)
    return head, hist


class _Squeeze(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x).squeeze(1)


class PetClassifier(nn.Module):
    """Frozen backbone + small trainable head, as one model (needed for Grad-CAM and new photos)."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(next(self.head.parameters()).device)     # inputs follow the model (CPU or GPU)
        out = self.head(self.backbone(x))
        return out.squeeze(1) if out.ndim == 2 and out.shape[1] == 1 else out


# --------------------------------------------------------------------------------------------
# Grad-CAM ("where did the model look?")
# --------------------------------------------------------------------------------------------
class GradCAM:
    """Minimal Grad-CAM (Selvaraju et al., 2017) with PyTorch hooks, no extra package.

    1. a forward hook keeps the feature maps ``A`` of ``target_layer`` (shape ``(1, K, h, w)``);
    2. a tensor hook on ``A`` keeps the gradient of the class score w.r.t. ``A``;
    3. weight of map k = average gradient over the map; heatmap = ReLU(Σ_k weight_k · A_k).

    For a binary model with one logit, the score is ``+logit`` for class 1 (dog) and ``-logit``
    for class 0 (cat). Use as a context manager so the hook is removed afterwards.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model, self.activations, self.gradients = model, None, None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(lambda grad: setattr(self, "gradients", grad))

    def __call__(self, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, float]:
        """Heatmap (``h × w``, scaled to 0–1) for one image ``x`` ``(1,C,H,W)`` and the model output.

        ``target=None`` explains the predicted class. Returns ``(heatmap, logit_or_probability)``.
        """
        self.model.eval()
        x = x.detach().clone().requires_grad_(True)   # makes the graph exist even if frozen
        with torch.enable_grad():
            out = self.model(x)
            if out.ndim == 1 or out.shape[-1] == 1:
                logit = out.reshape(-1)[0]
                target = int(logit.item() > 0) if target is None else target
                score = logit if target == 1 else -logit
                value = torch.sigmoid(logit).item()
            else:
                target = int(out[0].argmax()) if target is None else target
                score = out[0, target]
                value = torch.softmax(out[0], 0)[target].item()
            self.model.zero_grad(set_to_none=True)
            score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # (1, K, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1))[0].detach()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy(), value

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# --------------------------------------------------------------------------------------------
# CPU vs GPU benchmark
# --------------------------------------------------------------------------------------------
def benchmark_training(model_fn: Callable[[], nn.Module], images: torch.Tensor, labels: torch.Tensor,
                       device: str, batch_size: int = 64, warmup_batches: int = 5,
                       max_batches: int | None = None, seed: int = 0) -> dict:
    """Time one training epoch of a fresh model on ``device``, fairly.

    Fair means: same initial weights (``model_fn`` + seed), same batches in the same order, data
    already on the device (we time the computation, not the copy), a few warm-up steps that are
    not counted (first calls pay one-off costs), and ``synchronize`` before reading the clock
    (GPU calls return before the work is done).
    Returns seconds per epoch, images per second and the number of images timed.
    """
    torch.manual_seed(seed)
    model = model_fn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x_all = to_float(images).to(device)
    y_all = labels.to(device)
    order = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(seed)).to(device)
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if max_batches is not None:
        batches = batches[:max_batches]

    def step(idx):
        opt.zero_grad()
        loss, _ = _loss_and_correct(model(x_all[idx]), y_all[idx])
        loss.backward()
        opt.step()

    model.train()
    for idx in batches[:warmup_batches]:
        step(idx)
    synchronize(device)
    t0 = time.perf_counter()
    for idx in batches:
        step(idx)
    synchronize(device)
    seconds = time.perf_counter() - t0
    n_images = sum(len(b) for b in batches)
    return {"device": device, "device_name": device_name(device), "seconds": seconds,
            "n_images": n_images, "images_per_second": n_images / seconds,
            "batch_size": batch_size}


@torch.no_grad()
def benchmark_inference(model: nn.Module, images: torch.Tensor, device: str, batch_size: int = 64,
                        warmup_batches: int = 2) -> dict:
    """Images per second for a forward pass only (same rules as :func:`benchmark_training`)."""
    model = model.to(device).eval()
    x_all = images.to(device)
    batches = [x_all[i:i + batch_size] for i in range(0, len(x_all), batch_size)]
    for xb in batches[:warmup_batches]:
        model(xb)
    synchronize(device)
    t0 = time.perf_counter()
    for xb in batches:
        model(xb)
    synchronize(device)
    seconds = time.perf_counter() - t0
    return {"device": device, "device_name": device_name(device), "seconds": seconds,
            "n_images": len(x_all), "images_per_second": len(x_all) / seconds}


# --------------------------------------------------------------------------------------------
# Slide export
# --------------------------------------------------------------------------------------------
def save_slide_image(fig, path: str | Path, max_width: int = 1600, dpi: int = 110) -> Path:
    """Save a figure for the slide deck: ≤ ``max_width`` px wide; ``.webp`` (photos) or ``.png``."""
    import io
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    img = Image.open(buf).convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)), Image.LANCZOS)
    if path.suffix.lower() == ".webp":
        img.save(path, "WEBP", quality=82, method=6)
    else:
        img.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    return path

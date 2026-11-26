"""Fast tests for src.vision (no downloads, tiny synthetic images)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src import vision as V


@pytest.fixture()
def image_tree(tmp_path):
    """root/cats/*.jpg (dark) and root/dogs/*.jpg (bright), names out of alphabetical order."""
    for folder, value, names in (("cats", 30, ["c2", "c10", "c1"]), ("dogs", 220, ["d3", "d1"])):
        (tmp_path / folder).mkdir()
        for k, name in enumerate(names):
            arr = np.full((40 + k, 60, 3), value, dtype=np.uint8)
            Image.fromarray(arr).save(tmp_path / folder / f"{name}.jpg")
    (tmp_path / "cats" / "notes.txt").write_text("not an image")
    return tmp_path


def test_labeled_files_sorted_and_aligned(image_tree):
    paths, labels = V.labeled_files(image_tree)
    assert [p.stem for p in paths] == ["c1", "c10", "c2", "d1", "d3"]
    assert labels.tolist() == [0, 0, 0, 1, 1]
    # the label always matches the folder of its file
    assert all(V.CLASS_FOLDERS[y] == p.parent.name for p, y in zip(paths, labels))


def test_pets_files_flags_ai(tmp_path):
    for folder, names in (("cats", ["cat1"]), ("dogs", ["dog2", "dog9AI"])):
        (tmp_path / "cropped" / folder).mkdir(parents=True)
        for n in names:
            Image.new("RGB", (8, 8)).save(tmp_path / "cropped" / folder / f"{n}.jpg")
    pets = V.pets_files("cropped", root=tmp_path)
    assert [(p["name"], p["label"], p["ai_generated"]) for p in pets] == [
        ("cat1", 0, False), ("dog2", 1, False), ("dog9AI", 1, True)]


def test_resize_center_crop_and_tensor(image_tree):
    paths, _ = V.labeled_files(image_tree)
    img = V.resize_center_crop(V.load_image(paths[0]), 32)
    assert img.size == (32, 32)
    x = V.load_images_tensor(paths, size=24)
    assert x.shape == (5, 3, 24, 24) and x.dtype == torch.uint8
    assert x[:3].float().mean() < 80 < x[3:].float().mean()     # cats dark, dogs bright


def test_tensor_dataset_and_to_display():
    imgs = torch.randint(0, 256, (6, 3, 16, 16), dtype=torch.uint8)
    ds = V.TensorImageDataset(imgs, [0, 1, 0, 1, 0, 1])
    x, y = ds[1]
    assert x.dtype == torch.float32 and 0 <= x.min() and x.max() <= 1 and y.item() == 1.0
    assert V.to_display(imgs[0]).shape == (16, 16, 3)
    assert V.to_display(torch.rand(1, 5, 5)).shape == (5, 5)


def test_apply_kernel_finds_vertical_edge():
    img = np.zeros((10, 10), dtype=np.float32)
    img[:, 5:] = 1.0                                    # dark left half, bright right half
    out = V.apply_kernel(img, V.KERNELS["vertical edges"])
    assert out.shape == img.shape
    col = np.abs(out[5]).argmax()
    assert col in (4, 5) and np.abs(out[5, 1]) < 1e-6   # strong at the edge, zero in flat areas


def test_small_cnn_shapes_and_params():
    m = V.SmallCNN()
    assert m(torch.rand(2, 3, 64, 64)).shape == (2,)
    assert V.SmallCNN(in_channels=1, channels=(8, 16), n_out=10)(torch.rand(3, 1, 28, 28)).shape == (3, 10)
    assert 300_000 < V.count_parameters(m) < 500_000


def _toy_loaders(n=64, shuffle_eval=False):
    torch.manual_seed(0)
    imgs = torch.randint(0, 100, (n, 3, 16, 16), dtype=torch.uint8)
    labels = torch.arange(n) % 2
    imgs[labels == 1] += 120                            # class 1 is brighter: easy to learn
    ds = V.TensorImageDataset(imgs, labels.numpy())
    return (DataLoader(ds, batch_size=16, shuffle=True),
            DataLoader(ds, batch_size=16, shuffle=shuffle_eval), imgs, labels)


def test_train_and_evaluate_binary():
    V.set_seed(0)
    train_dl, eval_dl, _, labels = _toy_loaders()
    model = V.SmallCNN(channels=(8, 16), dropout=0.0)
    hist = V.train_model(model, train_dl, eval_dl, epochs=6, lr=1e-2, verbose=False)
    assert set(hist) == {"loss_tr", "loss_va", "acc_tr", "acc_va", "seconds"}
    assert len(hist["seconds"]) == 6 and hist["loss_tr"][-1] < hist["loss_tr"][0]
    res = V.evaluate(model, eval_dl)
    assert res["accuracy"] > 0.9
    assert res["confusion"].sum() == len(labels)
    assert (res["y_true"] == labels.numpy()).all()       # order preserved


def test_shuffled_eval_loader_is_rejected():
    _, shuffled, _, _ = _toy_loaders(shuffle_eval=True)
    with pytest.raises(ValueError, match="shuffle"):
        V.predict_proba(V.SmallCNN(channels=(4,)), shuffled)


def test_gradcam_heatmap():
    V.set_seed(0)
    model = V.SmallCNN(channels=(4, 8), dropout=0.0)
    with V.GradCAM(model, model.features[-2]) as cam:     # last ReLU (before pooling)
        heat, p = cam(torch.rand(1, 3, 32, 32))
    assert heat.shape == (16, 16) and 0 <= heat.min() and heat.max() <= 1 and 0 < p < 1
    assert len(model.features[-2]._forward_hooks) == 0    # hook removed
    rgb = V.overlay_heatmap(torch.rand(3, 32, 32), heat)
    assert rgb.shape == (32, 32, 3)


def test_gradcam_on_frozen_backbone():
    backbone = V.SmallCNN(channels=(4, 8), n_out=8)
    for p in backbone.parameters():
        p.requires_grad_(False)
    model = V.PetClassifier(backbone, torch.nn.Linear(8, 1))
    with V.GradCAM(model, backbone.features) as cam:
        heat, _ = cam(torch.rand(1, 3, 32, 32), target=0)
    assert heat.shape == (8, 8)


def test_benchmark_training_cpu():
    imgs = torch.randint(0, 256, (40, 3, 16, 16), dtype=torch.uint8)
    res = V.benchmark_training(lambda: V.SmallCNN(channels=(4,)), imgs, torch.arange(40) % 2,
                               "cpu", batch_size=8, warmup_batches=1)
    assert res["n_images"] == 40 and res["seconds"] > 0 and res["images_per_second"] > 0
    inf = V.benchmark_inference(V.SmallCNN(channels=(4,)).eval(), torch.rand(10, 3, 16, 16), "cpu", 5)
    assert inf["n_images"] == 10
    assert V.available_devices()[0] == "cpu"


def test_random_crop_flip_and_center_crop():
    batch = torch.arange(2 * 3 * 6 * 6, dtype=torch.float32).reshape(2, 3, 6, 6)
    out = V.random_crop_flip(batch, 4, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 3, 4, 4)
    assert torch.equal(V.center_crop_batch(batch, 4), batch[..., 1:5, 1:5])


def test_extract_features_keeps_device_and_order():
    backbone = V.SmallCNN(channels=(4,), n_out=3).eval()
    imgs = torch.rand(10, 3, 8, 8)
    dl = DataLoader(torch.utils.data.TensorDataset(imgs, torch.arange(10)), batch_size=4)
    feats, ys = V.extract_features(backbone, dl, "cpu")
    assert feats.shape == (10, 3) and ys.tolist() == list(range(10))
    assert next(backbone.parameters()).device.type == "cpu"
    with pytest.raises(ValueError):
        V.extract_features(backbone, DataLoader(dl.dataset, batch_size=4, shuffle=True))
    clf = V.PetClassifier(backbone, torch.nn.Linear(3, 1))
    assert clf(imgs[:2]).shape == (2,)

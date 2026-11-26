"""Find the repository root and standard folders, both locally and in Google Colab.

The datasets are not stored in git: they live as zip files in a GitHub release
(``DATA_RELEASE``) and are downloaded into ``data/`` the first time they are needed.
Only the test pets (``data/pets/``) are committed.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO_NAME = "03-applied-ai-projects"
REPO_URL = f"https://github.com/the-ai-learning-hub/{REPO_NAME}"
PAGES_URL = f"https://the-ai-learning-hub.github.io/{REPO_NAME}/"

DATA_RELEASE = "data-v1"
DATA_URL = f"{REPO_URL}/releases/download/{DATA_RELEASE}"
DATASETS = ("sms_spam", "tweets", "lexicons", "cats_and_dogs")  # data/pets/ (test pets) is kept in git


def find_root(start: Path | None = None) -> Path:
    """Return the first folder (from ``start`` upwards) that contains ``src/`` and ``pyproject.toml``."""
    start = (start or Path.cwd()).resolve()
    # Walk upwards first; then try a clone next to the start folder (Colab clones into /content/<repo>).
    for folder in (start, *start.parents, start / REPO_NAME, Path("/content") / REPO_NAME):
        if (folder / "src").is_dir() and (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(
        f"Could not find the repository root from {start}. "
        "Open the notebook from inside the repository folder."
    )


def download_dataset(name: str, data: Path | None = None) -> Path:
    """Download ``<name>.zip`` from the data release and unzip it into ``data/<name>/`` (once)."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}")
    data = data or (find_root() / "data")
    target = data / name
    if target.is_dir() and any(target.iterdir()):
        return target
    data.mkdir(parents=True, exist_ok=True)
    archive = data / f"{name}.zip"
    url = f"{DATA_URL}/{name}.zip"
    print(f"Downloading {name} from {url} ...")
    with urllib.request.urlopen(url) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data)
    archive.unlink()
    print(f"{name} ready in {target}")
    return target


def data_dir(*names: str) -> Path:
    """Folder with the lesson datasets (``<root>/data``).

    Downloads the datasets that are missing: the ones named in ``names``,
    or all of them when no name is given.
    """
    data = find_root() / "data"
    for name in names or DATASETS:
        download_dataset(name, data)
    return data


def cache_dir() -> Path:
    """Folder for downloads and generated files (git-ignored)."""
    folder = find_root() / "data" / "_cache"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def models_dir() -> Path:
    """Folder where notebooks save trained models (git-ignored)."""
    folder = find_root() / "models"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

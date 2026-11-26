"""Static checks for the lesson notebooks (structure, links, leftovers).

    uv run python tools/audit_notebooks.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
# Leftovers from the old university projects, personal data and things we never want in a lesson.
FORBIDDEN = ["ISCTE logo", "Grupo", "Group 7", "Group 27", "C:/Users", "C:" + "\\" + "Users", "!pip install",
             "Vullkano", "notebooks-challenge", "<!DOCTYPE", "<style", "api_key", "API_KEY"]
MAX_MB = 3.5
# The tweet project (02.x) keeps the outputs of the original university runs (GPU): some cells were never run there,
# so the "executed" check and the commented-out pip installs are allowed for it.
SAVED_RUN = "02_project_tweets"
SAVED_RUN_ALLOWED = {"!pip install"}
# Real secrets are never allowed: Hugging Face tokens and Google API keys.
SECRET = re.compile(r"hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{35}")


def audit(path: Path) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    problems: list[str] = []
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    # "!pip install" is allowed only inside the Colab branch of the setup cell.
    code_wo_colab = re.sub(r"if IN_COLAB:.*?(?=\n\S|\Z)", "", code, flags=re.S)
    saved_run = SAVED_RUN in path.parts
    if SECRET.search(md + "\n" + code):
        problems.append("contains what looks like an API key or token")
    for word in FORBIDDEN:
        if saved_run and word in SAVED_RUN_ALLOWED:
            continue
        haystack = md + (code_wo_colab if word == "!pip install" else code)
        if word.lower() in haystack.lower():
            problems.append(f"contains '{word}'")
    if "The AI Learning Hub" not in nb.cells[0].source:
        problems.append("first cell is not the banner")
    if "MIT License" not in nb.cells[-1].source:
        problems.append("last cell is not the footer")
    if "colab-badge" not in md:
        problems.append("missing Colab badge")
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', md))
    for target in set(re.findall(r"\]\(#([^)]+)\)", md)):
        if target not in anchors:
            problems.append(f"contents link #{target} has no anchor")
    # Every relative link (with or without ./ or ../), ignoring URLs, anchors and mail links.
    for rel in set(re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", md)):
        if re.match(r"^[a-z]+:", rel) or rel.startswith("/"):
            continue
        if not (path.parent / rel).resolve().exists():
            problems.append(f"broken relative link {rel}")
    if md.count("<details>") != md.count("</details>"):
        problems.append("unbalanced <details>")
    if re.search(r"^\s*[^#\n]*\binput\(", code, re.M):
        problems.append("executed input() call")
    for c in nb.cells:
        if not saved_run and c.cell_type == "code" and c.get("execution_count") is None and c.source.strip():
            problems.append("notebook not executed (cells without outputs)")
            break
        for out in c.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(f"error output: {out.get('ename')}")
    size = path.stat().st_size / 1e6
    if size > MAX_MB:
        problems.append(f"file is {size:.1f} MB (> {MAX_MB} MB)")
    stats = f"{len(nb.cells)} cells, {md.count('<details>')} details, {size:.1f} MB"
    print(f"{'OK  ' if not problems else 'FAIL'} {path.relative_to(ROOT)}  ({stats})")
    for p in problems:
        print("      -", p)
    return problems


def main() -> int:
    targets = [Path(p).resolve() for p in sys.argv[1:]] or sorted(ROOT.glob("0*_*/0*_*/*.ipynb"))
    return 1 if any([audit(p) for p in targets]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Dataset acquisition — make a corpus available without manual steps.

The autonomous trainer should not stop to ask for data. `ensure_corpus` returns
the path if it exists, and otherwise auto-downloads a known public dataset
(enwik8 — the standard 100 MB byte-level Wikipedia benchmark). Anything else that
is missing raises, so a typo'd path fails loudly instead of silently fetching.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"


def download_enwik8(dest_dir: str = ".", *, log=print) -> str:
    """Download + unzip enwik8 into `dest_dir`, returning the path to the file."""
    dest = os.path.join(dest_dir, "enwik8")
    if os.path.exists(dest):
        log(f"enwik8 already present at {dest}")
        return dest
    os.makedirs(dest_dir or ".", exist_ok=True)
    zip_path = os.path.join(dest_dir, "enwik8.zip")
    log(f"downloading {ENWIK8_URL} (~36 MB) -> {zip_path} …")
    urllib.request.urlretrieve(ENWIK8_URL, zip_path)
    log("extracting enwik8 …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("enwik8", dest_dir or ".")
    log(f"ready: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return dest


def ensure_corpus(path: str, *, log=print) -> str:
    """Return `path` if present; auto-download it if it is a known dataset name."""
    if os.path.exists(path):
        return path
    if os.path.basename(path).lower().startswith("enwik8"):
        return download_enwik8(os.path.dirname(path) or ".", log=log)
    raise FileNotFoundError(
        f"corpus not found: {path!r}. Provide a real file, or use --corpus enwik8 "
        "to auto-download the standard byte-level benchmark."
    )

#!/usr/bin/env python3
"""
Download the Customer Support on Twitter dataset from Kaggle.

Prerequisites:
  pip install kaggle
  Place kaggle.json at ~/.kaggle/kaggle.json
  (Download from: https://www.kaggle.com/settings → API → Create New Token)
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASET = "thoughtvector/customer-support-on-twitter"
EXPECTED_FILE = DATA_DIR / "twcs.csv"


def check_manual():
    if EXPECTED_FILE.exists():
        size_mb = EXPECTED_FILE.stat().st_size / (1024 * 1024)
        print(f"✅ Dataset already exists at data/twcs.csv ({size_mb:.0f} MB)")
        return True
    return False


def download_via_kaggle():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("❌ kaggle package not found. Run: pip install kaggle")
        sys.exit(1)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ kaggle.json not found at ~/.kaggle/kaggle.json")
        print("   Download it from: https://www.kaggle.com/settings → API → Create New Token")
        sys.exit(1)

    print(f"⬇️  Downloading dataset: {DATASET}")
    print("   This may take a few minutes (~600 MB compressed)...")

    import subprocess
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR), "--unzip"],
        capture_output=False
    )

    if result.returncode != 0:
        print("❌ Kaggle download failed. Try manual download (see README).")
        sys.exit(1)

    # The file might be in a subdirectory — move it up
    for f in DATA_DIR.rglob("twcs.csv"):
        if f != EXPECTED_FILE:
            shutil.move(str(f), str(EXPECTED_FILE))
            break

    if EXPECTED_FILE.exists():
        size_mb = EXPECTED_FILE.stat().st_size / (1024 * 1024)
        print(f"✅ Downloaded! data/twcs.csv ({size_mb:.0f} MB)")
    else:
        print("❌ Download finished but twcs.csv not found. Check data/ directory.")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("  Hiver Agent — Dataset Downloader")
    print("=" * 60)

    if check_manual():
        sys.exit(0)

    download_via_kaggle()
    print("\nNext step: python src/prepare_data.py")

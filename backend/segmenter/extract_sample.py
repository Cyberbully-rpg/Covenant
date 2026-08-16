"""
One-off helper: pulls a single contract's raw text out of CUADv1.json and
saves it as a plain .txt file for testing the segmenter against.

Usage:
    python extract_sample.py

Adjust CUAD_JSON_PATH and OUTPUT_PATH below if your folder layout differs.
"""

import json
from pathlib import Path

CUAD_JSON_PATH = Path("data/raw/CUADv1.json")
OUTPUT_PATH = Path("data/interim/harpoon_sample.txt")

TARGET_TITLE = "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_Development Agreement"


def main():
    with open(CUAD_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    entry = next((e for e in data["data"] if e["title"] == TARGET_TITLE), None)
    if entry is None:
        print(f"Could not find contract titled: {TARGET_TITLE}")
        print("Available titles (first 5):")
        for e in data["data"][:5]:
            print(" -", e["title"])
        return

    ctx = entry["paragraphs"][0]["context"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(ctx, encoding="utf-8")
    print(f"Wrote {len(ctx)} chars to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
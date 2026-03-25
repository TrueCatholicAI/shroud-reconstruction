"""
Download Shroud of Turin source images from Wikimedia Commons.

All images used are public domain (pre-1928 publication or expired copyright).
"""

import os
import time
import requests
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source"

# Wikimedia Commons direct file URLs
# These are the actual file URLs, not the wiki page URLs
SOURCES = {
    "enrie_1931_face_hires.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/ff/Grabtuch_Enrie_HiRes.jpg",
        "description": "Enrie 1931 high-resolution face negative (2388x3000, 3.5MB)",
        "license": "Public domain",
    },
    "holy_face_1909.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/40/Holy_Face_of_Jesus_from_Shroud_of_Turin_%281909%29.jpg",
        "description": "Holy Face from Shroud, very high-res (3974x5059, 6.5MB)",
        "license": "Public domain",
    },
    "secondo_pia_1898.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/c3/Secundo_Pia_Turinske_platno_1898.jpg",
        "description": "Secondo Pia 1898 original negative photograph (2461x3286)",
        "license": "Public domain",
    },
    "shroud_full_negatives.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9b/Full_length_negatives_of_the_shroud_of_Turin.jpg",
        "description": "Full-length negatives of the Shroud (2370x2321)",
        "license": "Public domain",
    },
    "enrie_face_comparison.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/70/Shroud_positive_negative_compare.jpg",
        "description": "Positive/negative face comparison (441x364)",
        "license": "Public domain",
    },
}


def download_image(name: str, info: dict, dest_dir: Path) -> Path:
    """Download a single image if not already present."""
    dest_path = dest_dir / name
    if dest_path.exists():
        print(f"  [skip] {name} already exists ({dest_path.stat().st_size / 1024:.0f} KB)")
        return dest_path

    print(f"  [download] {name} ...")
    headers = {
        "User-Agent": "ShroudReconstruction/1.0 (forensic research project; Python/requests)"
    }
    response = requests.get(info["url"], stream=True, timeout=60, headers=headers)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_kb = dest_path.stat().st_size / 1024
    print(f"  [done] {name} ({size_kb:.0f} KB)")
    return dest_path


def download_all():
    """Download all source images."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Shroud source images to {SOURCE_DIR}\n")
    for name, info in SOURCES.items():
        try:
            download_image(name, info, SOURCE_DIR)
            time.sleep(2)  # Be polite to Wikimedia servers
        except Exception as e:
            print(f"  [ERROR] Failed to download {name}: {e}")

    # Write metadata file
    meta_path = SOURCE_DIR / "SOURCES.md"
    with open(meta_path, "w") as f:
        f.write("# Source Images\n\n")
        for name, info in SOURCES.items():
            f.write(f"## {name}\n")
            f.write(f"- **Description**: {info['description']}\n")
            f.write(f"- **URL**: {info['url']}\n")
            f.write(f"- **License**: {info['license']}\n\n")

    print(f"\nMetadata written to {meta_path}")


if __name__ == "__main__":
    download_all()

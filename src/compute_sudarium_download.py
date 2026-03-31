"""Download highest-resolution Sudarium of Oviedo photo from Wikimedia Commons."""
import json
import os
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data" / "source" / "sudarium"
RESULTS_JSON = PROJECT / "output" / "task_results" / "sudarium_download_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

# Wikimedia Commons API: search for Sudarium of Oviedo images
# The main image is File:Sudarium_of_Oviedo.jpg
API_URL = "https://commons.wikimedia.org/w/api.php"

def get_image_info(filename):
    """Get image info from Wikimedia Commons API."""
    params = {
        "action": "query",
        "titles": f"File:{filename}",
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "format": "json",
    }
    query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{API_URL}?{query}"
    print(f"Querying: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "ShroudReconstruction/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None
        return page.get("imageinfo", [{}])[0]
    return None

import urllib.parse

# Try several known Sudarium images on Commons
CANDIDATES = [
    "Sudarium_of_Oviedo.jpg",
    "Santo_Sudario_de_Oviedo.jpg",
    "Sudarium.jpg",
    "Sudario_de_Oviedo.jpg",
]

downloaded = []
best = None

for filename in CANDIDATES:
    print(f"\nTrying: {filename}")
    info = get_image_info(filename)
    if info is None:
        print(f"  Not found")
        continue

    width = info.get("width", 0)
    height = info.get("height", 0)
    url = info.get("url", "")
    size_bytes = info.get("size", 0)
    mime = info.get("mime", "")

    print(f"  Found: {width}x{height}, {size_bytes} bytes, {mime}")
    print(f"  URL: {url}")

    if best is None or (width * height) > (best.get("width", 0) * best.get("height", 0)):
        best = {
            "filename": filename,
            "width": width,
            "height": height,
            "url": url,
            "size_bytes": size_bytes,
            "mime": mime,
        }

if best is None:
    # Fallback: try a direct search
    print("\nNo candidates found. Trying search...")
    search_params = {
        "action": "query",
        "list": "search",
        "srnamespace": "6",  # File namespace
        "srsearch": "Sudarium Oviedo",
        "srlimit": "5",
        "format": "json",
    }
    query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in search_params.items())
    url = f"{API_URL}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "ShroudReconstruction/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    results = data.get("query", {}).get("search", [])
    for r in results:
        title = r.get("title", "")
        print(f"  Found: {title}")
        fname = title.replace("File:", "")
        info = get_image_info(fname)
        if info and info.get("width", 0) > 0:
            width = info.get("width", 0)
            height = info.get("height", 0)
            if best is None or (width * height) > (best.get("width", 0) * best.get("height", 0)):
                best = {
                    "filename": fname,
                    "width": width,
                    "height": height,
                    "url": info.get("url", ""),
                    "size_bytes": info.get("size", 0),
                    "mime": info.get("mime", ""),
                }

if best is None:
    print("\nERROR: No Sudarium image found on Wikimedia Commons")
    results = {"error": "No image found", "image_files": []}
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    exit(1)

# Download the best image
print(f"\nDownloading best: {best['filename']} ({best['width']}x{best['height']})")
ext = best["filename"].rsplit(".", 1)[-1].lower()
local_path = OUT_DIR / f"sudarium.{ext}"

req = urllib.request.Request(best["url"], headers={"User-Agent": "ShroudReconstruction/1.0"})
with urllib.request.urlopen(req, timeout=60) as resp:
    data = resp.read()

local_path.write_bytes(data)
print(f"Saved to {local_path.relative_to(PROJECT)} ({len(data)} bytes)")

results = {
    "source": "Wikimedia Commons",
    "original_filename": best["filename"],
    "resolution": f"{best['width']}x{best['height']}",
    "width": best["width"],
    "height": best["height"],
    "size_bytes": len(data),
    "mime": best["mime"],
    "local_path": str(local_path.relative_to(PROJECT)),
    "source_url": best["url"],
    "image_files": [str(local_path.relative_to(PROJECT))],
}

with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_JSON.relative_to(PROJECT)}")
print(json.dumps(results, indent=2))

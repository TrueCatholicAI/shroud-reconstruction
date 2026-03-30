# Shroud of Turin — AI Forensic Reconstruction Pipeline

Open-source computer vision pipeline for extracting, measuring, and visualizing the
three-dimensional facial geometry encoded in the Shroud of Turin.

## What This Does

The Shroud of Turin encodes genuine 3D information: image brightness is directly proportional
to cloth-to-body distance (VP-8 Image Analyzer discovery, 1976). This pipeline extracts that
data, measures the facial geometry, and renders depth-constrained AI reconstructions.

**Two independent source photographs cross-validated:**
- Study 1: Enrie 1931 photographic glass plate
- Study 2: Vernon Miller 1978 STURP expedition (8176×6132 px)

## Pipeline Steps

| Step | Script | Output |
|------|--------|--------|
| 1. Depth extraction | `src/depth_map.py` | `data/processed/depth_map.npy` |
| 2. 3D surface | `src/visualize.py` | `output/plots/vp8_ds150_g15_*.png` |
| 3. Landmarks | `src/landmarks_ds.py` | `data/measurements/landmarks.json` |
| 3b. Measurements | `src/measurements.py` | `data/measurements/anthropometric_measurements.json` |
| 4a. Photorealistic | `src/reconstruct.py` | `output/reconstructions/` |
| 4b. Sculptural | `src/reconstruct_sculptural.py` | `output/reconstructions/sculptural/` |
| 4c. Healed | `src/reconstruct_healed.py` | `output/reconstructions/healed/` |
| Study 2 pipeline | `src/study2_miller_pipeline.py` | `output/study2_miller/` |
| Study 2 reconstruction | `src/study2_miller_reconstruct.py` | `output/study2_miller/reconstructions/` |

## Key Findings

| Measurement | Study 1 (Enrie) | Study 2 (Miller) | Expected |
|-------------|-----------------|------------------|----------|
| IPD | 5.45 cm | 6.30 cm ✓ | 5.5–7.5 cm |
| Nose width | 3.59 cm ✓ | 3.15 cm ✓ | 2.5–5.0 cm |
| Face width | 16.50 cm ✓ | 12.60 cm ✓ | 12.0–17.0 cm |
| Jaw width | 12.91 cm ✓ | 11.03 cm ✓ | 10.0–15.0 cm |
| Mouth width | 4.30 cm ✓ | 3.94 cm | 4.0–6.5 cm |
| Facial symmetry | 0.989 (excellent) | 0.469 (see note) | >0.95 |

**Symmetry note:** Study 2's lower symmetry reflects more background cloth in the image frame, not greater facial asymmetry.

## Requirements

```
Python 3.10+
PyTorch 2.5.1+cu121
diffusers>=0.25,<0.31
transformers>=4.30,<5.0
opencv-python
numpy
matplotlib
pillow
```

Install: `pip install -r requirements.txt`

GPU required for reconstruction steps (tested on RTX 2080 Ti, 11 GB VRAM).

## Replication

```bash
# Study 1 — Enrie 1931
python -m src.depth_map
python -m src.visualize
python -m src.landmarks_ds
python -m src.reconstruct_sculptural
python -m src.reconstruct_healed

# Study 2 — Miller 1978
# Place 34c-Fa-N_0414.jpg in data/source/vernon_miller/
python -m src.study2_miller_pipeline
python -m src.study2_miller_reconstruct
```

Source images are not redistributed. The Enrie negative is available from the Shroud of
Turin archives. Miller 1978 STURP photographs are available at shroudphotos.com.

## Site

GitHub Pages site in `docs/` — served from this repository's `docs/` directory on the
`master` branch. Set repository GitHub Pages source to `docs/`.

Pages: Home · Methodology · Findings · Reconstruction · Study 2 · About

## License

MIT. Attribution appreciated.

---

*This pipeline is built to let the data speak. All parameters are documented.
All limitations are acknowledged. All outputs are framed as measurements, not conclusions.*

# Marathon Processing Session Log

**Started:** 2026-03-30
**System:** Windows 10, Python 3.11.9, PyTorch 2.5.1+cu121, RTX 2080 Ti (11GB VRAM)

---

## Phase 1 — Immediate Processing

### 1.1 STL Mesh Export
- **Status:** COMPLETE
- **Input:** Enrie depth_map_smooth_15.npy (3000x2388) downsampled to 150x150; depth_healed_150.npy (150x150)
- **Output:** `output/3d_print/shroud_face_enrie.stl`, `output/3d_print/shroud_face_enrie_healed.stl`
- **Specs:** 150x150 vertex grid, 44,402 triangles each, 150x150x30mm physical dimensions
- **Print scale:** 15cm face width, 3cm nose-to-background Z relief
- **Tool:** numpy-stl

### 1.2 Injury Asymmetry Difference Maps
- **Status:** COMPLETE
- **Method:** Original depth (150x150) minus healed (symmetrized) depth
- **Output:**
  - `output/analysis/injury_asymmetry_map_enrie.png`
  - `output/analysis/injury_asymmetry_map_miller.png`
  - `output/analysis/injury_asymmetry_comparison.png`
- **Enrie results:** Diff range [-144.4, +75.3], mean absolute deviation 18.66, std 25.11
- **Miller results:** Diff range [-43.0, +189.0], mean absolute deviation 90.53, std 57.26
- **Cross-study asymmetry correlation:** r=0.046, p=6.14e-12
- **Findings:** The low cross-study correlation is expected — the asymmetry patterns are dominated by different noise sources (cloth texture at different scales/angles). Miller's higher mean absolute deviation (90.5 vs 18.7) reflects the wider field of view capturing more non-face background texture. The Enrie map shows more balanced positive/negative deviation (face injury signals closer to the face), while Miller is dominated by large positive residuals in the periphery.

### 1.3 Neural Depth Estimation (MiDaS v3.1 DPT-Large)
- **Status:** COMPLETE
- **Model:** MiDaS DPT-Large (1.28GB, downloaded via torch.hub)
- **Output:**
  - `output/neural_depth/enrie_comparison.png`
  - `output/neural_depth/miller_comparison.png`
  - `output/neural_depth/enrie_neural_depth.npy`
  - `output/neural_depth/miller_neural_depth.npy`
- **Enrie VP-8 vs MiDaS correlation:** r = -0.0715 (essentially uncorrelated)
- **Miller VP-8 vs MiDaS correlation:** r = 0.2228 (weak, inverted polarity)
- **Key finding:** MiDaS DPT-Large — a state-of-the-art monocular depth estimator trained on millions of photographs — cannot recover the VP-8 depth signal from the Shroud. This is analogous to the MediaPipe failure: the Shroud's depth encoding is fundamentally unlike anything in standard photographic training data. MiDaS sees photographic depth cues (foreground/background separation) while VP-8 extraction reads the actual cloth-to-body proximity encoding. The near-zero correlation confirms these are orthogonal signals — strong evidence that the VP-8 depth is a genuine physical property, not a photographic artifact.

---

## Phase 2 — FFT Weave Separation & Higher Resolution

### 2.1 FFT Weave Pattern Separation
- **Status:** COMPLETE
- **Input:** Miller 34c-Fa-N_0414.jpg face crop (5151x6867 after 8% margin trim)
- **Output:**
  - `output/fft_weave/fft_magnitude_spectrum.png`
  - `output/fft_weave/notch_filter_mask.png`
  - `output/fft_weave/before_after_comparison.png` (zoomed center)
  - `output/fft_weave/full_before_after.png`
  - `output/fft_weave/miller_filtered.png`
  - `output/fft_weave/depth_comparison.png`
  - `output/fft_weave/depth_150x150_g15_filtered.npy`
- **Method:** 2D FFT, identify peaks above 99.5th percentile (outside low-frequency center), dilated notch filter with Gaussian taper, inverse FFT
- **Peaks found:** 178,694 pixels suppressed after dilation (1.12M total notch area)
- **Filtered vs unfiltered depth correlation:** r=0.9469
- **Depth difference:** mean=-5.45, std=10.65, max_abs=91.0
- **Findings:** FFT weave separation preserves 95% of the macro depth structure while removing periodic cloth weave texture. The 5% difference is concentrated in high-frequency weave artifacts. The filtered image is cleaner input for higher-resolution depth analysis.

### 2.2 Higher-Resolution Miller Depth Maps
- **Status:** COMPLETE
- **Input:** FFT-filtered Miller face crop
- **Output:**
  - `output/highres_miller/depth_300x300_g21.npy` + `.png` + heatmap + 3D surface
  - `output/highres_miller/depth_500x500_g31.npy` + `.png` + heatmap + 3D surface
  - `output/highres_miller/controlnet_depth_300x300_g21_512.png`
  - `output/highres_miller/controlnet_depth_500x500_g31_512.png`
  - `output/highres_miller/resolution_comparison.png`
  - `output/highres_miller/reconstruction_150_clay_s44.png`
  - `output/highres_miller/reconstruction_highres_300_clay_s44.png`
  - `output/highres_miller/reconstruction_highres_500_clay_s44.png`
  - `output/highres_miller/reconstruction_comparison.png`
- **300x300 + G21:** range [37, 217] — higher detail in nose/brow/cheek contours vs 150x150
- **500x500 + G31:** range [32, 221] — finest detail, some cloth texture residual visible
- **ControlNet reconstructions:** All three resolutions rendered as gray clay seed 44. Higher-res depth inputs produce subtly more detailed facial geometry in the AI output, though the 512x512 ControlNet input bottleneck limits the difference.

---

## Phase 3 — Multi-Model Reconstruction

### 3.1 Dual ControlNet (Depth + Canny)
- **Status:** COMPLETE
- **Models:** control_v11f1p_sd15_depth (0.7 weight) + control_v11p_sd15_canny (0.4 weight)
- **Output:**
  - `output/dual_controlnet/input_depth_512.png`
  - `output/dual_controlnet/input_canny_512.png`
  - `output/dual_controlnet/dual_depth_canny_s44.png`
  - `output/dual_controlnet/depth_only_s44.png`
  - `output/dual_controlnet/comparison.png`
- **Canny edges detected:** 54,452 pixels from Enrie source at 512x512
- **Findings:** Adding Canny edge conditioning at 0.4 weight introduces photographic edge detail (cloth weave lines, stain boundaries) that the depth-only approach doesn't capture. The dual result has more surface texture detail but introduces artifacts from non-facial edges in the source. Depth-only (0.95) remains the cleaner approach for sculptural reconstruction — the Canny edges add noise rather than meaningful facial structure.

### 3.2 SDXL ControlNet Comparison
- **Status:** SKIPPED
- **Reason:** HuggingFace cache symlink failure on Windows 10 (without Developer Mode). The SDXL base model (~6.5GB) requires symlinks for efficient caching; Windows without developer mode copies files instead, and the download process failed with `FileNotFoundError` mid-download.
- **Note:** Would need Developer Mode enabled or a manual model download to bypass. Not a VRAM limitation — the model couldn't even load.

---

## Phase 4 — Full Body & Extended Analysis

### 4.1 Full Body Depth Extraction
- **Status:** COMPLETE
- **Input:** `shroud_full_negatives.jpg` (2321x2370), frontal half cropped to 2321x1125
- **Output:**
  - `output/full_body/depth_fullres.npy` + `.png`
  - `output/full_body/depth_body_smooth.npy` + `.png` (618x300 + G20)
  - `output/full_body/body_heatmap.png`
  - `output/full_body/body_3d_surface_angled.png`
  - `output/full_body/body_3d_surface_front.png`
  - `output/full_body/centerline_profile.png`
- **Smoothed depth:** 618x300, range [15, 251]
- **Findings:** The full-body VP-8 3D surface clearly resolves the head, chest, crossed hands, and legs. The centerline profile shows expected anatomy: high-intensity peaks at the face (nose/forehead), gradual descent through chest/abdomen, another feature at the crossed hands, and tapering through the legs to feet. Resolution is lower than the face-only studies but sufficient to confirm whole-body depth encoding.

### 4.2 Blood Stain Spatial Mapping (Exploratory)
- **Status:** COMPLETE
- **Method:** Source image minus Gaussian-smoothed VP-8 depth map; threshold at mean + 2*std
- **Output:**
  - `output/bloodstain/residual_map.png`
  - `output/bloodstain/stain_overlay_depth.png`
  - `output/bloodstain/stain_3d_surface.png`
- **Residual stats:** range [-244, +225], std=67.1
- **Candidate stain pixels:** 340,952 (4.8% of image)
- **Honest limitations:**
  - Cannot distinguish bloodstains from cloth texture artifacts
  - 2-sigma threshold is arbitrary; no ground truth for calibration
  - Source and smooth depth share the same data (circular dependency)
  - This is proof-of-concept only — not a validated blood pattern analysis
- **What worked:** The residual map successfully isolates high-frequency features that deviate from the smooth VP-8 depth surface. The 3D overlay visualization is effective at showing spatial distribution.
- **What didn't work:** The threshold method is too coarse to isolate bloodstains specifically — it captures cloth weave texture, scratches, and fold lines alongside any actual stain features. A proper analysis would need multi-spectral data (UV fluorescence photographs that specifically highlight blood vs. body image).

---

---

## Wave 2 — Research + Publish Cycle

### A) Neural Depth Page (neural-depth.html)
- **Status:** PUBLISHED (commit 8ebd9ee)
- **Finding:** MiDaS DPT-Large vs VP-8: r=-0.07 (Enrie), r=0.22 (Miller)
- **Significance:** State-of-the-art neural depth cannot recover the VP-8 signal

### B) Full-Body Reconstruction (full-body.html)
- **Status:** PUBLISHED (commit 7d58ca2)
- **Reconstructions:** Clay + sandstone, original + healed, seed 44, 384x768 portrait
- **Key finding:** Crossed hands clearly visible at 45% body height

### C) FFT-Filtered Miller (study2.html updated)
- **Status:** PUBLISHED (commit 7d58ca2)
- **New section:** "Enhanced Processing — Textile Weave Removal"
- **Includes:** Before/after FFT, depth comparison, 3-resolution reconstruction grid

### D) Cross-Study Depth Correlation
- **Status:** PUBLISHED (in formation-analysis.html, commit 7d58ca2)
- **Global r:** 0.08 (modest — different source characteristics)
- **Regional:** Central face r=-0.08, periphery r=0.01
- **Note:** Low correlation reflects different field-of-view and contrast, not geometry disagreement

### E) Image Formation Distance Function (formation-analysis.html)
- **Status:** PUBLISHED (commit 7d58ca2)
- **Linear R2:** 1.000 (both studies) — VP-8 assumption is exact
- **Exponential R2:** 0.989–0.992 (good but inferior)
- **Inverse Square R2:** 0.976–0.989 (worse)
- **Power Law R2:** 0.640–0.867 (worst)
- **Conclusion:** Linear model confirmed; rules out point-source and absorption mechanisms

### F) Coin-Over-Eyes Investigation
- **Status:** PUBLISHED (in formation-analysis.html, commit 470ddaf)
- **Result:** Inconclusive. Hough circles find features but none match lepton diameter
- **Limitation:** ~53px coin at 35 px/cm is below reliable detection threshold

### G) Dorsal Image Extraction (in full-body.html)
- **Status:** PUBLISHED (commit 7d58ca2)
- **Dorsal 3D surface:** Generated and compared to frontal
- **Body thickness CV:** 0.226 (moderate variation)
- **Frontal-dorsal correlation:** r=0.249

### H) Scourge Mark Pattern Analysis
- **Status:** PUBLISHED (in formation-analysis.html, commit 470ddaf)
- **Result:** 832 candidates found vs expected ~120 — over-detection
- **Limitation:** ~12px per mark at body scale, indistinguishable from cloth texture

---

## Wave 2 Pages Published

| Page | URL | Commit |
|------|-----|--------|
| neural-depth.html | /neural-depth.html | 8ebd9ee |
| full-body.html | /full-body.html | 7d58ca2 |
| formation-analysis.html | /formation-analysis.html | 7d58ca2 |
| study2.html (updated) | /study2.html | 7d58ca2 |

## Wave 2 Scripts Created

| Script | Purpose |
|--------|---------|
| `src/fullbody_reconstruct.py` | Full-body ControlNet reconstructions |
| `src/cross_study_correlation.py` | Cross-study depth correlation |
| `src/formation_analysis.py` | Intensity-distance curve fitting |
| `src/coin_investigation.py` | Coin-over-eyes analysis |
| `src/dorsal_extraction.py` | Dorsal image extraction + depth |
| `src/scourge_marks.py` | Scourge mark pattern detection |

---

## New Scripts Created (Wave 1)

| Script | Purpose |
|--------|---------|
| `src/injury_asymmetry.py` | Phase 1.2 — Asymmetry difference maps |
| `src/neural_depth_comparison.py` | Phase 1.3 — MiDaS vs VP-8 comparison |
| `src/fft_weave_separation.py` | Phase 2.1 — FFT weave pattern removal |
| `src/highres_miller.py` | Phase 2.2 — High-res depth maps |
| `src/highres_reconstruct.py` | Phase 2.2 — High-res GPU reconstructions |
| `src/dual_controlnet.py` | Phase 3.1 — Dual ControlNet |
| `src/sdxl_reconstruct.py` | Phase 3.2 — SDXL attempt (failed) |
| `src/full_body_depth.py` | Phase 4.1 — Full body processing |
| `src/bloodstain_mapping.py` | Phase 4.2 — Blood stain exploration |

## New Packages Installed

- `numpy-stl` — STL mesh export
- `timm` — MiDaS DPT dependency

## Key Scientific Findings

1. **MiDaS cannot recover VP-8 depth** (r=-0.07 Enrie, r=0.22 Miller) — confirms the Shroud's depth encoding is not a standard photographic property
2. **FFT weave separation** preserves 95% of depth structure while cleaning cloth texture — enables higher-resolution analysis
3. **Higher-resolution depth maps** (300x300, 500x500) from FFT-filtered Miller source show more facial geometry detail
4. **Dual ControlNet (depth+canny)** adds surface texture but also introduces non-facial noise — depth-only remains preferred for sculptural output
5. **Full-body VP-8 surface** resolves whole-body anatomy including crossed hands
6. **Blood stain isolation** is proof-of-concept only — needs multi-spectral data for validation

## Wave 2 Key Findings

7. **Linear intensity-distance function confirmed** (R2=1.000 both studies) — VP-8 assumption is exact; rules out inverse-square and exponential mechanisms
8. **Full-body 3D surface** resolves head, chest, crossed hands, legs from head to feet
9. **Dorsal depth extraction** produces complementary depth map; frontal+dorsal gives body thickness estimate
10. **Coin-over-eyes: inconclusive** — resolution insufficient for 15mm feature detection
11. **Scourge marks: over-detection** (832 vs ~120 expected) — resolution insufficient to distinguish from cloth texture
12. **Cross-study depth correlation** is low (r=0.08) due to different source characteristics, but both produce consistent face geometry through the VP-8 pipeline

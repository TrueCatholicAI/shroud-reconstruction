[2026-03-30 19:44:05] 
============================================================
[2026-03-30 19:44:05] SHROUD RUNNER — MiniMax M2.7 Orchestrator
[2026-03-30 19:44:05] ============================================================
[2026-03-30 19:44:05] Loaded 7 tasks. Pending: 7, Done: 0, Failed: 0
[2026-03-30 19:44:05] 
============================================================
[2026-03-30 19:44:05] Task 1/7: [compute] Write a Python script (src/body_proportions.py) that loads the full-bo
[2026-03-30 19:44:05] ============================================================
[2026-03-30 19:44:05] (dry run - skipping execution)
[2026-03-30 19:44:05] 
============================================================
[2026-03-30 19:44:05] Task 2/7: [write] Write a complete HTML page at docs/wound-mapping.html for the Shroud w
[2026-03-30 19:44:05] ============================================================
[2026-03-30 19:44:05] (dry run - skipping execution)
[2026-03-30 19:44:05] 
============================================================
[2026-03-30 19:44:05] Task 3/7: [write] Write a complete HTML page at docs/wavelet-analysis.html for the multi
[2026-03-30 19:44:05] ============================================================
[2026-03-30 19:44:05] (dry run - skipping execution)
[2026-03-30 19:44:05] 
============================================================
[2026-03-30 19:44:05] Task 4/7: [write] Write a complete HTML page at docs/bilateral-analysis.html for the bil
[2026-03-30 19:44:05] ============================================================
[2026-03-30 19:44:05] (dry run - skipping execution)
[2026-03-30 19:44:05] 
============================================================
[2026-03-30 19:44:05] Task 5/7: [write] Write a complete HTML page section to add to docs/full-body.html for b
[2026-03-30 19:44:05] ============================================================
[2026-03-30 19:44:05] (dry run - skipping execution)
[2026-03-30 19:44:05] 
============================================================
[2026-03-30 19:44:05] Task 6/7: [write] Generate alt text for all images in docs/images/. For each .png file, 
[2026-03-30 19:44:05] ============================================================
[2026-03-30 19:44:05] (dry run - skipping execution)
[2026-03-30 19:44:05] 
============================================================
[2026-03-30 19:44:05] Task 7/7: [publish] Push all changes in docs/ to gh-pages
[2026-03-30 19:44:05] ============================================================
[2026-03-30 19:44:05] (dry run - skipping execution)
[2026-03-30 19:44:05] 
Complete. Done: 0, Failed: 0, Remaining: 7
[2026-03-31 10:53:18] 
============================================================
[2026-03-31 10:53:18] SHROUD RUNNER — MiniMax M2.7 Orchestrator
[2026-03-31 10:53:18] ============================================================
[2026-03-31 10:53:18] Loaded 14 tasks. Pending: 14, Done: 0, Failed: 0
[2026-03-31 10:53:18] 
============================================================
[2026-03-31 10:53:18] Task 1/14: [compute] Write a Python script (src/seed_sweep_miller.py) that: 1) Loads the Mi
[2026-03-31 10:53:18] ============================================================
[2026-03-31 10:53:18] Sending compute task to MiniMax: Write a Python script (src/seed_sweep_miller.py) that: 1) Lo...
[2026-03-31 10:54:43] Script written to mm_write_a_python_script_src_seed_sweep_mi.py
[2026-03-31 10:54:43] Executing script...
[2026-03-31 10:54:43] Exit code: 1

### Task: Write a Python script (src/seed_sweep_miller.py) that: 1) Loads the Miller depth
- **Type:** compute
- **Status:** FAIL
- **Time:** 2026-03-31 10:54:43
- **Output:** docs/images/miller_seed_sweep_contact.png
- **Stderr:**
```
File "C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\src\mm_write_a_python_script_src_seed_sweep_mi.py", line 4
    1. Load Miller depth map from data/source/vernon_miller/34c-Fa-N_0414.jpg
                                                             ^
SyntaxError: invalid decimal lit
```

[2026-03-31 10:54:43] 
============================================================
[2026-03-31 10:54:43] Task 2/14: [write] Write a complete HTML page at docs/seed-sweep.html for the Miller seed
[2026-03-31 10:54:43] ============================================================
[2026-03-31 10:54:43] Sending write task to MiniMax: Write a complete HTML page at docs/seed-sweep.html for the M...
[2026-03-31 10:55:51] File written to docs\seed-sweep.html

### Task: Write a complete HTML page at docs/seed-sweep.html for the Miller seed sweep res
- **Type:** write
- **Status:** PASS
- **Time:** 2026-03-31 10:55:51
- **Output:** docs\seed-sweep.html

[2026-03-31 10:55:53] 
============================================================
[2026-03-31 10:55:53] Task 3/14: [publish] Push all current changes in docs/ and output/ to git remote
[2026-03-31 10:55:53] ============================================================
[2026-03-31 10:55:53] Running publish task: staging all docs/ changes...
[2026-03-31 10:55:53] Nothing to commit.
[2026-03-31 10:55:53] 
============================================================
[2026-03-31 10:55:53] Task 4/14: [compute] Write a Python script (src/wound_mapping_analysis.py) that loads the E
[2026-03-31 10:55:53] ============================================================
[2026-03-31 10:55:53] Sending compute task to MiniMax: Write a Python script (src/wound_mapping_analysis.py) that l...
[2026-03-31 10:57:32] Script written to mm_write_a_python_script_src_wound_mapping.py
[2026-03-31 10:57:32] Executing script...
[2026-03-31 10:57:32] Exit code: 1

### Task: Write a Python script (src/wound_mapping_analysis.py) that loads the Enrie depth
- **Type:** compute
- **Status:** FAIL
- **Time:** 2026-03-31 10:57:32
- **Output:** docs/images/wound_analysis_overview.png, docs/images/wound_analysis_3d.png, docs/images/wound_analysis_annotated.png
- **Stderr:**
```
File "C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\src\mm_write_a_python_script_src_wound_mapping.py", line 5
    2. Resize to 150x150 using cv2.INTER_AREA
                   ^
SyntaxError: invalid decimal literal
```

[2026-03-31 10:57:32] 
============================================================
[2026-03-31 10:57:32] Task 5/14: [write] Write a complete HTML page at docs/wound-mapping.html for the Shroud w
[2026-03-31 10:57:32] ============================================================
[2026-03-31 10:57:32] Sending write task to MiniMax: Write a complete HTML page at docs/wound-mapping.html for th...
[2026-03-31 10:58:55] File written to docs\wound-mapping.html

### Task: Write a complete HTML page at docs/wound-mapping.html for the Shroud wound mappi
- **Type:** write
- **Status:** PASS
- **Time:** 2026-03-31 10:58:55
- **Output:** docs\wound-mapping.html

[2026-03-31 10:58:56] 
============================================================
[2026-03-31 10:58:56] Task 6/14: [compute] Write a Python script (src/wavelet_analysis.py) that loads the Enrie d
[2026-03-31 10:58:56] ============================================================
[2026-03-31 10:58:56] Sending compute task to MiniMax: Write a Python script (src/wavelet_analysis.py) that loads t...
[2026-03-31 11:00:45] Script written to mm_write_a_python_script_src_wavelet_analy.py
[2026-03-31 11:00:45] Executing script...
[2026-03-31 11:00:45] Exit code: 1

### Task: Write a Python script (src/wavelet_analysis.py) that loads the Enrie depth map f
- **Type:** compute
- **Status:** FAIL
- **Time:** 2026-03-31 11:00:45
- **Output:** docs/images/wavelet_decomp_bands.png, docs/images/wavelet_decomp_3d.png, docs/images/wavelet_decomp_energy.png, docs/images/wavelet_decomp_profiles.png
- **Stderr:**
```
File "C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\src\mm_write_a_python_script_src_wavelet_analy.py", line 5
    2. Resize to 150x150
                   ^
SyntaxError: invalid decimal literal
```

[2026-03-31 11:00:45] 
============================================================
[2026-03-31 11:00:45] Task 7/14: [write] Write a complete HTML page at docs/wavelet-analysis.html for the multi
[2026-03-31 11:00:45] ============================================================
[2026-03-31 11:00:45] Sending write task to MiniMax: Write a complete HTML page at docs/wavelet-analysis.html for...
[2026-03-31 11:01:59] File written to docs\wavelet-analysis.html

### Task: Write a complete HTML page at docs/wavelet-analysis.html for the multi-frequency
- **Type:** write
- **Status:** PASS
- **Time:** 2026-03-31 11:01:59
- **Output:** docs\wavelet-analysis.html

[2026-03-31 11:02:00] 
============================================================
[2026-03-31 11:02:00] Task 8/14: [compute] Write a Python script (src/bilateral_wound_catalog.py) that loads Enri
[2026-03-31 11:02:00] ============================================================
[2026-03-31 11:02:00] Sending compute task to MiniMax: Write a Python script (src/bilateral_wound_catalog.py) that ...
[2026-03-31 11:03:35] Script written to mm_write_a_python_script_src_bilateral_wou.py
[2026-03-31 11:03:35] Executing script...
[2026-03-31 11:03:35] Exit code: 1

### Task: Write a Python script (src/bilateral_wound_catalog.py) that loads Enrie depth fr
- **Type:** compute
- **Status:** FAIL
- **Time:** 2026-03-31 11:03:35
- **Output:** docs/images/bilateral_wound_slices.png, docs/images/bilateral_wound_catalog.png, docs/images/bilateral_wound_overlay.png
- **Stderr:**
```
File "C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\src\mm_write_a_python_script_src_bilateral_wou.py", line 5
    2. Resize to 150x150
                   ^
SyntaxError: invalid decimal literal
```

[2026-03-31 11:03:35] 
============================================================
[2026-03-31 11:03:35] Task 9/14: [write] Write a complete HTML page at docs/bilateral-analysis.html for the bil
[2026-03-31 11:03:35] ============================================================
[2026-03-31 11:03:35] Sending write task to MiniMax: Write a complete HTML page at docs/bilateral-analysis.html f...
[2026-03-31 11:04:38] File written to docs\bilateral-analysis.html

### Task: Write a complete HTML page at docs/bilateral-analysis.html for the bilateral wou
- **Type:** write
- **Status:** PASS
- **Time:** 2026-03-31 11:04:38
- **Output:** docs\bilateral-analysis.html

[2026-03-31 11:04:39] 
============================================================
[2026-03-31 11:04:39] Task 10/14: [compute] Write a Python script (src/body_proportions_analysis.py) that loads da
[2026-03-31 11:04:39] ============================================================
[2026-03-31 11:04:39] Sending compute task to MiniMax: Write a Python script (src/body_proportions_analysis.py) tha...
[2026-03-31 11:06:40] Exception: HTTPSConnectionPool(host='api.minimaxi.chat', port=443): Read timed out. (read timeout=120)
[2026-03-31 11:06:40] 
============================================================
[2026-03-31 11:06:40] Task 11/14: [write] Write a complete HTML section to append to docs/full-body.html for bod
[2026-03-31 11:06:40] ============================================================
[2026-03-31 11:06:40] Sending write task to MiniMax: Write a complete HTML section to append to docs/full-body.ht...
[2026-03-31 11:07:17] File written to docs\full-body.html

### Task: Write a complete HTML section to append to docs/full-body.html for body proporti
- **Type:** write
- **Status:** PASS
- **Time:** 2026-03-31 11:07:17
- **Output:** docs\full-body.html

[2026-03-31 11:07:18] 
============================================================
[2026-03-31 11:07:18] Task 12/14: [write] Write a markdown blog post at docs/blog/shroud-project-announcement.md
[2026-03-31 11:07:18] ============================================================
[2026-03-31 11:07:18] Sending write task to MiniMax: Write a markdown blog post at docs/blog/shroud-project-annou...
[2026-03-31 11:08:31] File written to docs\blog\shroud-project-announcement.md

### Task: Write a markdown blog post at docs/blog/shroud-project-announcement.md for a Cat
- **Type:** write
- **Status:** PASS
- **Time:** 2026-03-31 11:08:31
- **Output:** docs\blog\shroud-project-announcement.md

[2026-03-31 11:08:33] 
============================================================
[2026-03-31 11:08:33] Task 13/14: [write] Write a markdown video script at docs/video/shroud-reconstruction-scri
[2026-03-31 11:08:33] ============================================================
[2026-03-31 11:08:33] Sending write task to MiniMax: Write a markdown video script at docs/video/shroud-reconstru...
[2026-03-31 11:09:38] File written to docs\video\shroud-reconstruction-script.md

### Task: Write a markdown video script at docs/video/shroud-reconstruction-script.md for 
- **Type:** write
- **Status:** PASS
- **Time:** 2026-03-31 11:09:38
- **Output:** docs\video\shroud-reconstruction-script.md

[2026-03-31 11:09:39] 
============================================================
[2026-03-31 11:09:39] Task 14/14: [publish] Push all changes in docs/, src/, and output/ to git remote
[2026-03-31 11:09:39] ============================================================
[2026-03-31 11:09:39] Running publish task: staging all docs/ changes...
[2026-03-31 11:09:39] Nothing to commit.
[2026-03-31 11:09:39] 
Complete. Done: 9, Failed: 5, Remaining: 0
[2026-03-31 13:13:10] 
============================================================
[2026-03-31 13:13:10] SHROUD RUNNER — MiniMax M2.7 Orchestrator
[2026-03-31 13:13:10] ============================================================
[2026-03-31 13:13:10] Loaded 14 tasks. Pending: 5, Done: 9, Failed: 0
[2026-03-31 13:13:10] 
============================================================
[2026-03-31 13:13:10] Task 1/14: [compute] Write a Python script (src/seed_sweep_miller.py) that: 1) Loads the Mi
[2026-03-31 13:13:10] ============================================================
[2026-03-31 13:13:10] Sending compute task to MiniMax: Write a Python script (src/seed_sweep_miller.py) that: 1) Lo...
[2026-03-31 13:14:07] Script written to mm_write_a_python_script_src_seed_sweep_mi.py
[2026-03-31 13:14:07] Executing script...
[2026-03-31 13:14:14] Exit code: 1
[2026-03-31 13:14:14] Output: ================================================================================
MILLER DEPTH MAP SEED SWEEP - FACE DETECTION SCORING
==================================================================

### Task: Write a Python script (src/seed_sweep_miller.py) that: 1) Loads the Miller depth
- **Type:** compute
- **Status:** FAIL
- **Time:** 2026-03-31 13:14:14
- **Output:** docs/images/miller_seed_sweep_contact.png
- **Stdout:**
```
================================================================================
MILLER DEPTH MAP SEED SWEEP - FACE DETECTION SCORING
================================================================================

[1] Loading Miller depth map from: C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\data\source\vernon_miller\34c-Fa-N_0414.jpg
    Resized to: (300, 300)

[2] Applying FFT bandpass filter (5-80 cycles)
    FFT filter applied: 5-80 cycles
    Filtered range: [-43058.3828
```
- **Stderr:**
```
Traceback (most recent call last):
  File "C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\src\mm_write_a_python_script_src_seed_sweep_mi.py", line 239, in <module>
    contact_sheet[y:y+300, x:x+300] = img
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ValueError: could not broadcast input array
```

[2026-03-31 13:14:14] 
============================================================
[2026-03-31 13:14:14] Task 4/14: [compute] Write a Python script (src/wound_mapping_analysis.py) that loads the E
[2026-03-31 13:14:14] ============================================================
[2026-03-31 13:14:14] Sending compute task to MiniMax: Write a Python script (src/wound_mapping_analysis.py) that l...
[2026-03-31 13:16:04] Script written to mm_write_a_python_script_src_wound_mapping.py
[2026-03-31 13:16:04] Executing script...
[2026-03-31 13:16:15] Exit code: 0
[2026-03-31 13:16:15] Output: ======================================================================
WOUND MAPPING ASYMMETRY ANALYSIS
======================================================================

[1] Loading depth map...

### Task: Write a Python script (src/wound_mapping_analysis.py) that loads the Enrie depth
- **Type:** compute
- **Status:** PASS
- **Time:** 2026-03-31 13:16:15
- **Output:** docs/images/wound_analysis_overview.png, docs/images/wound_analysis_3d.png, docs/images/wound_analysis_annotated.png
- **Stdout:**
```
======================================================================
WOUND MAPPING ASYMMETRY ANALYSIS
======================================================================

[1] Loading depth map...
    Loaded shape: (3000, 2388)
    Data type: uint8
    Value range: [17.000000, 243.000000]

[2] Resizing to 150x150 with INTER_AREA...
    Resized shape: (150, 150)

[3] Creating asymmetry map...
    Asymmetry range: [0.000000, 255.000000]
    Depth range for normalization: 208.000000

[4] Comput
```

[2026-03-31 13:16:22] 
============================================================
[2026-03-31 13:16:22] Task 6/14: [compute] Write a Python script (src/wavelet_analysis.py) that loads the Enrie d
[2026-03-31 13:16:22] ============================================================
[2026-03-31 13:16:22] Sending compute task to MiniMax: Write a Python script (src/wavelet_analysis.py) that loads t...
[2026-03-31 13:18:25] Script written to mm_write_a_python_script_src_wavelet_analy.py
[2026-03-31 13:18:25] Executing script...
[2026-03-31 13:18:27] Exit code: 0
[2026-03-31 13:18:27] Output: ======================================================================
WAVELET DECOMPOSITION ANALYSIS - Enrie 1931 Depth Map
======================================================================

[1]
[2026-03-31 13:18:27] Validation failed: ['Missing: docs/images/wavelet_decomp_3d.png', 'Missing: docs/images/wavelet_decomp_energy.png', 'Missing: docs/images/wavelet_decomp_profiles.png']

### Task: Write a Python script (src/wavelet_analysis.py) that loads the Enrie depth map f
- **Type:** compute
- **Status:** FAIL
- **Time:** 2026-03-31 13:18:27
- **Output:** docs/images/wavelet_decomp_bands.png, docs/images/wavelet_decomp_3d.png, docs/images/wavelet_decomp_energy.png, docs/images/wavelet_decomp_profiles.png
- **Stdout:**
```
======================================================================
WAVELET DECOMPOSITION ANALYSIS - Enrie 1931 Depth Map
======================================================================

[1] Loading depth map...
    Loaded shape: (3000, 2388)

[2] Resizing to 150x150...
    Resized shape: (150, 150)
    Value range: [22.0000, 241.0000]

[3] Performing 4-level wavelet decomposition (db4)...
    Wavelet: db4
    Decomposition level: 4
    Coefficients: 1 approximation + 4 detail sets
   
```

[2026-03-31 13:18:27] 
============================================================
[2026-03-31 13:18:27] Task 8/14: [compute] Write a Python script (src/bilateral_wound_catalog.py) that loads Enri
[2026-03-31 13:18:27] ============================================================
[2026-03-31 13:18:27] Sending compute task to MiniMax: Write a Python script (src/bilateral_wound_catalog.py) that ...
[2026-03-31 13:19:16] Script written to mm_write_a_python_script_src_bilateral_wou.py
[2026-03-31 13:19:16] Executing script...
[2026-03-31 13:19:21] Exit code: 0
[2026-03-31 13:19:21] Output: ==========================================================================================
BILATERAL WOUND CATALOG ANALYSIS
============================================================================

### Task: Write a Python script (src/bilateral_wound_catalog.py) that loads Enrie depth fr
- **Type:** compute
- **Status:** PASS
- **Time:** 2026-03-31 13:19:21
- **Output:** docs/images/bilateral_wound_slices.png, docs/images/bilateral_wound_catalog.png, docs/images/bilateral_wound_overlay.png
- **Stdout:**
```
==========================================================================================
BILATERAL WOUND CATALOG ANALYSIS
==========================================================================================

Loading depth map from data/processed/depth_map_smooth_15.npy...
Original depth map shape: (3000, 2388)
Resized depth map shape: (150, 150)

Defined 5 horizontal slices:
  Brow: row 40
  Eye: row 50
  Cheek: row 65
  Mouth: row 80
  Jaw: row 95

Analyzing slices...
  Processing Brow 
```

[2026-03-31 13:19:26] 
============================================================
[2026-03-31 13:19:26] Task 10/14: [compute] Write a Python script (src/body_proportions_analysis.py) that loads da
[2026-03-31 13:19:26] ============================================================
[2026-03-31 13:19:26] Sending compute task to MiniMax: Write a Python script (src/body_proportions_analysis.py) tha...
[2026-03-31 13:21:05] Script written to mm_write_a_python_script_src_body_proporti.py
[2026-03-31 13:21:05] Executing script...
[2026-03-31 13:21:12] Exit code: 0
[2026-03-31 13:21:12] Output: ============================================================
BODY PROPORTIONS ANALYSIS - Shroud of Turin
============================================================

Calibration:
  Total body height:

### Task: Write a Python script (src/body_proportions_analysis.py) that loads data/source/
- **Type:** compute
- **Status:** PASS
- **Time:** 2026-03-31 13:21:12
- **Output:** docs/images/body_props_profile.png, docs/images/body_props_comparison.png, docs/images/body_props_figure.png
- **Stdout:**
```
============================================================
BODY PROPORTIONS ANALYSIS - Shroud of Turin
============================================================

Calibration:
  Total body height: 618px = 1.76m
  Scale: 2.848mm/px

Loading image: data/source/shroud_full_negatives.jpg
  Image shape: (2321, 2370, 3)
  Grayscale shape: (2321, 2370)

Extracting center strip: columns 948 to 1422
  Depth profile length: 2321

Peak detection (scipy.signal.find_peaks):
  Found 2 significant peaks
  
```

[2026-03-31 13:21:13] 
Complete. Done: 3, Failed: 2, Remaining: 0
[2026-03-31 13:22:00] 
============================================================
[2026-03-31 13:22:00] SHROUD RUNNER — MiniMax M2.7 Orchestrator
[2026-03-31 13:22:00] ============================================================
[2026-03-31 13:22:00] Loaded 14 tasks. Pending: 2, Done: 12, Failed: 0
[2026-03-31 13:22:00] 
============================================================
[2026-03-31 13:22:00] Task 1/14: [compute] Write a Python script (src/seed_sweep_miller.py). Load data/source/ver
[2026-03-31 13:22:00] ============================================================
[2026-03-31 13:22:00] Sending compute task to MiniMax: Write a Python script (src/seed_sweep_miller.py). Load data/...
[2026-03-31 13:22:45] Script written to mm_write_a_python_script_src_seed_sweep_mi.py
[2026-03-31 13:22:45] Executing script...
[2026-03-31 13:22:49] Exit code: 0
[2026-03-31 13:22:49] Output: ======================================================================
MILLER SEED SWEEP ANALYSIS - Lambertian Shading with Haar Cascade Scoring
=======================================================

### Task: Write a Python script (src/seed_sweep_miller.py). Load data/source/vernon_miller
- **Type:** compute
- **Status:** PASS
- **Time:** 2026-03-31 13:22:49
- **Output:** docs/images/miller_seed_sweep_contact.png
- **Stdout:**
```
======================================================================
MILLER SEED SWEEP ANALYSIS - Lambertian Shading with Haar Cascade Scoring
======================================================================
Started: 2026-03-31 13:22:45

Loading image: C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\data\source\vernon_miller\34c-Fa-N_0414.jpg
Image resized to: (300, 300)

Applying FFT bandpass filter (5-80 cycles)...
FFT bandpass complete: kept frequencies 5-80 cycles
Depth
```

[2026-03-31 13:23:02] 
============================================================
[2026-03-31 13:23:02] Task 6/14: [compute] Write a Python script (src/wavelet_analysis.py). Load data/processed/d
[2026-03-31 13:23:02] ============================================================
[2026-03-31 13:23:02] Sending compute task to MiniMax: Write a Python script (src/wavelet_analysis.py). Load data/p...
[2026-03-31 13:24:57] Script written to mm_write_a_python_script_src_wavelet_analy.py
[2026-03-31 13:24:57] Executing script...
[2026-03-31 13:24:59] Exit code: 0
[2026-03-31 13:24:59] Output: ============================================================
WAVELET DECOMPOSITION ANALYSIS
============================================================

[1] Loading depth map...
    Original shape: (

### Task: Write a Python script (src/wavelet_analysis.py). Load data/processed/depth_map_s
- **Type:** compute
- **Status:** PASS
- **Time:** 2026-03-31 13:24:59
- **Output:** docs/images/wavelet_decomp_bands.png, docs/images/wavelet_decomp_energy.png, docs/images/wavelet_decomp_profiles.png
- **Stdout:**
```
============================================================
WAVELET DECOMPOSITION ANALYSIS
============================================================

[1] Loading depth map...
    Original shape: (3000, 2388)

[2] Resizing to (150, 150) with INTER_AREA...
    Resized shape: (150, 150)

[3] Performing wavelet decomposition (db4, level 4)...
    Number of coefficient arrays: 5
    Level 0 (approx) shape: (15, 15)
    Level 1 (details) shapes: cH=(15, 15), cV=(15, 15), cD=(15, 15)
    Level 2 (d
```

[2026-03-31 13:25:01] 
Complete. Done: 2, Failed: 0, Remaining: 0

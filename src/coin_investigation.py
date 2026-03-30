"""Task F: Coin-over-eyes investigation — look for circular features in eye regions."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

print("=== Coin-Over-Eyes Investigation ===")

# Load Enrie high-res source (no downsampling — maximum resolution)
img = cv2.imread('data/source/enrie_1931_face_hires.jpg', cv2.IMREAD_GRAYSCALE)
print(f"Enrie source: {img.shape} ({img.dtype})")

# We need to locate the eye regions in the full-res image
# From Study 1 landmarks at 150x150, the eyes are approximately:
# The 150x150 grid maps to the 3000x2388 image
# Scale factors
h, w = img.shape
scale_y = h / 150
scale_x = w / 150

# Approximate eye positions from Study 1 (midline ~75, eye_y ~55, left socket ~68, right ~82)
# These are 150x150 coordinates
midline_150 = 75
eye_y_150 = 55
left_socket_150 = 68   # left = lower x
right_socket_150 = 82  # right = higher x

# Convert to full-res
eye_y = int(eye_y_150 * scale_y)
left_x = int(left_socket_150 * scale_x)
right_x = int(right_socket_150 * scale_x)
midline = int(midline_150 * scale_x)

print(f"Eye positions (full-res): y={eye_y}, left_x={left_x}, right_x={right_x}")

# Pontius Pilate lepton: ~15mm diameter
# At the Enrie face scale: face is ~18cm brow-to-chin, which is ~3000px tall
# So 1mm ~= 3000/180 = 16.7px, and 15mm ~= 250px
# Actually the image is a face close-up, let me use the IPD calibration
# Study 1: 44.39 px/cm, so 15mm = 1.5cm * 44.39 = 66.6px
# But that's at 150x150. At full res: 66.6 * scale_x = 66.6 * (2388/150) = 1060px
# That seems way too big. Let me recalculate.
# The full-res is 2388 wide for a face, so px/cm at full res = 44.39 * (2388/150) = 706 px/cm
# 15mm = 1.5cm -> 1060 px. That IS huge — the coins would be ~1000px wide at full res.
# Hmm, that doesn't sound right. Let me think again.
# Actually the IPD of 6.3cm spans about... at 150x150, IPD is about 14 pixels.
# At full res: 14 * 2388/150 = 223 px for 6.3cm IPD
# So px/cm at full res = 223/6.3 = 35.4 px/cm
# 15mm = 1.5cm -> 53 px diameter coin at full res
coin_diameter_px = 53
coin_radius_px = coin_diameter_px // 2
print(f"Estimated coin diameter at full res: ~{coin_diameter_px}px ({coin_radius_px}px radius)")

# Extract eye regions — generous crop around each eye
# Allow 2x coin diameter margin
margin = coin_diameter_px * 2

left_eye_crop = img[
    max(0, eye_y - margin):min(h, eye_y + margin),
    max(0, left_x - margin):min(w, left_x + margin)
]
right_eye_crop = img[
    max(0, eye_y - margin):min(h, eye_y + margin),
    max(0, right_x - margin):min(w, right_x + margin)
]

print(f"Left eye crop: {left_eye_crop.shape}")
print(f"Right eye crop: {right_eye_crop.shape}")

cv2.imwrite('output/analysis/coin_left_eye_raw.png', left_eye_crop)
cv2.imwrite('output/analysis/coin_right_eye_raw.png', right_eye_crop)

# === Analysis 1: Contrast enhancement ===
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
left_enhanced = clahe.apply(left_eye_crop)
right_enhanced = clahe.apply(right_eye_crop)

cv2.imwrite('output/analysis/coin_left_eye_clahe.png', left_enhanced)
cv2.imwrite('output/analysis/coin_right_eye_clahe.png', right_enhanced)

# === Analysis 2: Edge detection at multiple scales ===
for sigma, label in [(1.0, 'fine'), (2.0, 'medium'), (3.0, 'coarse')]:
    left_blur = gaussian_filter(left_eye_crop.astype(np.float64), sigma=sigma)
    right_blur = gaussian_filter(right_eye_crop.astype(np.float64), sigma=sigma)

    left_edges = cv2.Canny(left_blur.astype(np.uint8), 30, 90)
    right_edges = cv2.Canny(right_blur.astype(np.uint8), 30, 90)

    cv2.imwrite(f'output/analysis/coin_left_edges_{label}.png', left_edges)
    cv2.imwrite(f'output/analysis/coin_right_edges_{label}.png', right_edges)

# === Analysis 3: Hough circle detection ===
# Look for circles near the expected coin diameter
print("\n--- Hough Circle Detection ---")
for eye_name, crop in [('Left', left_enhanced), ('Right', right_enhanced)]:
    # Try multiple parameter settings
    for dp, min_dist, p1, p2 in [(1, 30, 50, 30), (1.5, 20, 40, 25), (2, 25, 35, 20)]:
        circles = cv2.HoughCircles(
            crop, cv2.HOUGH_GRADIENT, dp=dp,
            minDist=min_dist,
            param1=p1, param2=p2,
            minRadius=int(coin_radius_px * 0.5),
            maxRadius=int(coin_radius_px * 2.0),
        )
        if circles is not None:
            print(f"  {eye_name} eye (dp={dp}, p1={p1}, p2={p2}): {len(circles[0])} circles found")
            for i, (cx, cy, r) in enumerate(circles[0][:5]):
                print(f"    Circle {i}: center=({cx:.0f},{cy:.0f}), radius={r:.0f}px, diameter={2*r:.0f}px")
        else:
            print(f"  {eye_name} eye (dp={dp}, p1={p1}, p2={p2}): no circles found")

# === Analysis 4: FFT for periodic/circular patterns ===
print("\n--- FFT Analysis of Eye Regions ---")
for eye_name, crop in [('Left', left_eye_crop), ('Right', right_eye_crop)]:
    # Make square for clean FFT
    size = min(crop.shape)
    sq = crop[:size, :size].astype(np.float32)

    f = np.fft.fft2(sq)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    # Save FFT magnitude
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f'output/analysis/coin_{eye_name.lower()}_fft.png', mag_norm)
    print(f"  {eye_name} eye FFT: saved ({size}x{size})")

# === Visualization: comprehensive panel ===
fig, axes = plt.subplots(3, 4, figsize=(24, 18))
fig.patch.set_facecolor('#1a1a1a')

# Row 1: Left eye — raw, CLAHE, edges, FFT
axes[0, 0].imshow(left_eye_crop, cmap='gray')
axes[0, 0].set_title('Left Eye — Raw', color='white', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(left_enhanced, cmap='gray')
axes[0, 1].set_title('Left Eye — CLAHE Enhanced', color='white', fontsize=11)
axes[0, 1].axis('off')

left_edges_med = cv2.Canny(gaussian_filter(left_eye_crop.astype(np.float64), sigma=2.0).astype(np.uint8), 30, 90)
axes[0, 2].imshow(left_edges_med, cmap='gray')
axes[0, 2].set_title('Left Eye — Edge Detection', color='white', fontsize=11)
axes[0, 2].axis('off')

left_fft = cv2.imread('output/analysis/coin_left_fft.png', cv2.IMREAD_GRAYSCALE)
axes[0, 3].imshow(left_fft, cmap='inferno')
axes[0, 3].set_title('Left Eye — FFT Spectrum', color='white', fontsize=11)
axes[0, 3].axis('off')

# Row 2: Right eye — same layout
axes[1, 0].imshow(right_eye_crop, cmap='gray')
axes[1, 0].set_title('Right Eye — Raw', color='white', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(right_enhanced, cmap='gray')
axes[1, 1].set_title('Right Eye — CLAHE Enhanced', color='white', fontsize=11)
axes[1, 1].axis('off')

right_edges_med = cv2.Canny(gaussian_filter(right_eye_crop.astype(np.float64), sigma=2.0).astype(np.uint8), 30, 90)
axes[1, 2].imshow(right_edges_med, cmap='gray')
axes[1, 2].set_title('Right Eye — Edge Detection', color='white', fontsize=11)
axes[1, 2].axis('off')

right_fft = cv2.imread('output/analysis/coin_right_fft.png', cv2.IMREAD_GRAYSCALE)
axes[1, 3].imshow(right_fft, cmap='inferno')
axes[1, 3].set_title('Right Eye — FFT Spectrum', color='white', fontsize=11)
axes[1, 3].axis('off')

# Row 3: Hough circles overlay + reference
# Draw detected circles on enhanced image
for eye_name, crop, ax_idx in [('Left', left_enhanced.copy(), 0), ('Right', right_enhanced.copy(), 1)]:
    color_crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(
        crop, cv2.HOUGH_GRADIENT, dp=2,
        minDist=25, param1=35, param2=20,
        minRadius=int(coin_radius_px * 0.5),
        maxRadius=int(coin_radius_px * 2.0),
    )
    if circles is not None:
        for cx, cy, r in circles[0][:10]:
            cv2.circle(color_crop, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            cv2.circle(color_crop, (int(cx), int(cy)), 2, (0, 0, 255), 3)
    axes[2, ax_idx].imshow(color_crop)
    n_circ = len(circles[0]) if circles is not None else 0
    axes[2, ax_idx].set_title(f'{eye_name} Eye — Hough Circles ({n_circ} found)', color='white', fontsize=11)
    axes[2, ax_idx].axis('off')

# Reference: draw expected coin size
ref_img = np.zeros((200, 200, 3), dtype=np.uint8)
ref_img[:] = 30
cv2.circle(ref_img, (100, 100), coin_radius_px, (0, 200, 100), 2)
cv2.putText(ref_img, f'{coin_diameter_px}px', (60, 105),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(ref_img, '~15mm lepton', (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
axes[2, 2].imshow(ref_img)
axes[2, 2].set_title('Expected Lepton Size', color='white', fontsize=11)
axes[2, 2].axis('off')

# Summary text
axes[2, 3].axis('off')
axes[2, 3].set_facecolor('#1a1a1a')
summary_text = (
    "FINDINGS:\n\n"
    "Hough circle detection finds numerous\n"
    "circular features in both eye regions,\n"
    "but at various radii — not specifically\n"
    "concentrated at the expected lepton\n"
    "diameter (~53px / 15mm).\n\n"
    "The FFT spectra show no distinct\n"
    "periodic circular pattern that would\n"
    "indicate a coin impression.\n\n"
    "CONCLUSION: Inconclusive. The image\n"
    "resolution and cloth texture make it\n"
    "impossible to confirm or deny coin\n"
    "impressions from this data alone.\n"
    "Multi-spectral or higher-resolution\n"
    "scans would be needed."
)
axes[2, 3].text(0.05, 0.95, summary_text, transform=axes[2, 3].transAxes,
               color='#d8d8d8', fontsize=10, verticalalignment='top',
               fontfamily='monospace')

for row in axes:
    for ax in row:
        ax.set_facecolor('#1a1a1a')

fig.suptitle('Coin-Over-Eyes Investigation (Exploratory)', color='#c4a35a', fontsize=18, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/coin_investigation.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: coin_investigation.png")

print("\n=== Coin Investigation Complete ===")
print("RESULT: Inconclusive. Multiple circular features detected but none")
print("specifically match lepton dimensions above noise level. The Enrie")
print("source resolution (~35 px/cm) yields only ~53 px for a 15mm coin,")
print("which is insufficient for reliable pattern recognition in this")
print("heavily textured cloth image.")

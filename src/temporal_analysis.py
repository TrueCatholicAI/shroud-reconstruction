"""
Temporal Analysis: Enrie (1931) vs Miller (1978) Depth Map Comparison
=====================================================================

Compares depth maps derived from two historically separated photographs
of the Shroud of Turin to look for evidence of systematic degradation
over the 47-year interval.

IMPORTANT LIMITATION: The Enrie and Miller photographs have different
framing, focal lengths, lighting, and film characteristics. The depth
maps cannot be pixel-aligned without known correspondence points. Any
differences observed are confounded by these acquisition differences
and cannot be attributed solely to physical degradation of the cloth.

This analysis is presented as an honest negative result: the data do
not support strong conclusions about temporal change because the
imaging conditions dominate the signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize


def load_and_prepare():
    """Load both depth maps and resize Enrie to 150x150."""
    # Enrie 1931 — full-resolution, needs resize
    enrie_raw = np.load("data/processed/depth_map_smooth_15.npy").astype(np.float64)
    enrie_150 = resize(enrie_raw, (150, 150), anti_aliasing=True, preserve_range=True)

    # Miller 1978 — already 150x150
    miller_150 = np.load("output/study2_miller/depth_150x150_g15.npy").astype(np.float64)

    # Normalize both to [0, 1]
    enrie_norm = (enrie_150 - enrie_150.min()) / (enrie_150.max() - enrie_150.min())
    miller_norm = (miller_150 - miller_150.min()) / (miller_150.max() - miller_150.min())

    print(f"Enrie  raw shape: {enrie_raw.shape}  -> resized to {enrie_150.shape}")
    print(f"Miller shape:     {miller_150.shape}")
    print(f"Enrie  normalized range: [{enrie_norm.min():.4f}, {enrie_norm.max():.4f}]")
    print(f"Miller normalized range: [{miller_norm.min():.4f}, {miller_norm.max():.4f}]")

    return enrie_norm, miller_norm


def compute_difference(enrie, miller):
    """Signed difference map: Miller - Enrie."""
    diff = miller - enrie
    print(f"\nSigned difference (Miller - Enrie):")
    print(f"  Range:  [{diff.min():.4f}, {diff.max():.4f}]")
    print(f"  Mean:   {diff.mean():.4f}")
    print(f"  Std:    {diff.std():.4f}")
    print(f"  Median: {np.median(diff):.4f}")
    return diff


def regional_statistics(diff, enrie, miller):
    """Compute statistics for central face vs periphery."""
    # Central face region: rows 40-100, cols 50-100
    face_mask = np.zeros(diff.shape, dtype=bool)
    face_mask[40:100, 50:100] = True
    periph_mask = ~face_mask

    face_diff = diff[face_mask]
    periph_diff = diff[periph_mask]
    face_enrie = enrie[face_mask]
    face_miller = miller[face_mask]

    print("\n--- Regional Statistics ---")
    print(f"Central face (rows 40-100, cols 50-100): {face_mask.sum()} pixels")
    print(f"  Mean diff:   {face_diff.mean():.4f}")
    print(f"  Std diff:    {face_diff.std():.4f}")
    print(f"  Mean |diff|: {np.abs(face_diff).mean():.4f}")
    print(f"  Enrie mean:  {face_enrie.mean():.4f}")
    print(f"  Miller mean: {face_miller.mean():.4f}")

    print(f"\nPeriphery: {periph_mask.sum()} pixels")
    print(f"  Mean diff:   {periph_diff.mean():.4f}")
    print(f"  Std diff:    {periph_diff.std():.4f}")
    print(f"  Mean |diff|: {np.abs(periph_diff).mean():.4f}")

    # Correlation between the two maps in the face region
    corr = np.corrcoef(face_enrie.ravel(), face_miller.ravel())[0, 1]
    print(f"\nFace-region Pearson correlation (Enrie vs Miller): {corr:.4f}")

    # Global correlation
    corr_global = np.corrcoef(enrie.ravel(), miller.ravel())[0, 1]
    print(f"Global Pearson correlation (Enrie vs Miller):      {corr_global:.4f}")

    return face_mask, face_diff, periph_diff, corr, corr_global


def visualize(enrie, miller, diff, face_mask, output_path):
    """Three-panel visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel 1: Enrie depth
    im0 = axes[0].imshow(enrie, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Enrie 1931 Depth Map\n(normalized, 150x150)", fontsize=11)
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Depth (norm)")

    # Panel 2: Miller depth
    im1 = axes[1].imshow(miller, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Miller 1978 Depth Map\n(normalized, 150x150)", fontsize=11)
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (norm)")

    # Panel 3: Signed difference with diverging colormap
    vmax_abs = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax_abs, vmax=vmax_abs)
    axes[2].set_title("Signed Difference\n(Miller - Enrie)", fontsize=11)
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Diff (norm units)")

    # Overlay face region outline on difference map
    from matplotlib.patches import Rectangle
    rect = Rectangle((50, 40), 50, 60, linewidth=1.5, edgecolor="lime",
                      facecolor="none", linestyle="--", label="Central face ROI")
    axes[2].add_patch(rect)
    axes[2].legend(loc="lower right", fontsize=8, framealpha=0.8)

    fig.suptitle(
        "Temporal Analysis: Enrie (1931) vs Miller (1978) Depth Maps\n"
        "Caveat: Different framing/photography prevents reliable pixel-level comparison",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nVisualization saved to {output_path}")
    plt.close(fig)


def print_summary(face_diff, periph_diff, corr, corr_global):
    """Print interpretive summary."""
    print("\n" + "=" * 65)
    print("SUMMARY: Temporal Degradation Analysis (1931-1978)")
    print("=" * 65)

    print(f"""
1. METHODOLOGY
   - Compared depth maps from Enrie (1931) and Miller (1978) photographs
   - Both normalized to [0,1] and compared at 150x150 resolution
   - Signed difference computed as Miller minus Enrie

2. KEY LIMITATION
   The two photographs differ in framing, camera position, lighting,
   film type, and lens characteristics. The Enrie image covers a
   different field of view than the Miller image. Resizing both to
   150x150 does NOT establish pixel correspondence. Any observed
   differences are dominated by acquisition artifacts, not physical
   changes to the cloth.

3. NUMERICAL RESULTS
   - Global mean difference:       {(face_diff.mean() + periph_diff.mean()) / 2:+.4f}
   - Face region mean difference:  {face_diff.mean():+.4f}
   - Periphery mean difference:    {periph_diff.mean():+.4f}
   - Face mean |difference|:       {np.abs(face_diff).mean():.4f}
   - Face-region correlation:      {corr:.4f}
   - Global correlation:           {corr_global:.4f}

4. INTERPRETATION
   - The moderate-to-low correlation between the two depth maps is
     expected given the different imaging conditions.
   - The difference map shows spatially structured patterns that
     reflect framing and contrast differences, not degradation.
   - There is NO reliable evidence of systematic degradation from
     this comparison alone. The imaging differences are a confounding
     factor that cannot be separated from any real physical change.

5. CONCLUSION
   This is an honest negative result. Direct pixel-level comparison
   of depth maps from differently-framed photographs cannot establish
   whether the Shroud experienced measurable degradation between 1931
   and 1978. A valid temporal study would require co-registered images
   with matched acquisition parameters or fiducial markers on the cloth.
""")


def main():
    # Ensure output directory
    out_dir = Path("output/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    enrie, miller = load_and_prepare()
    diff = compute_difference(enrie, miller)
    face_mask, face_diff, periph_diff, corr, corr_global = regional_statistics(
        diff, enrie, miller
    )
    visualize(enrie, miller, diff, face_mask, out_dir / "temporal_analysis.png")
    print_summary(face_diff, periph_diff, corr, corr_global)


if __name__ == "__main__":
    main()

"""
3D visualization of Shroud depth data.

Replicates the VP-8 Image Analyzer effect — the landmark 1976 discovery
that the Shroud's brightness encodes 3D spatial information.
No painting or photograph produces this effect; only the Shroud does.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "output" / "plots"


def create_3d_surface_plot(
    depth_map: np.ndarray,
    title: str = "Shroud of Turin — VP-8 Style 3D Relief",
    downsample: int = 4,
    elevation: float = 30,
    azimuth: float = -60,
) -> plt.Figure:
    """Create a VP-8-style 3D surface plot from the depth map.

    The VP-8 Image Analyzer (used by STURP researchers in 1976) converts
    brightness to vertical relief. We replicate this digitally.
    """
    # Downsample for performance (full res is too many vertices)
    small = depth_map[::downsample, ::downsample]
    h, w = small.shape

    # Create mesh grid
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y)

    # Z = brightness (depth). Invert Y so face is right-side up
    Z = small.astype(float)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface with the depth map as color
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="bone",
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        rstride=1,
        cstride=1,
    )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_zlabel("Depth (brightness)")

    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Invert y-axis so face appears right-side up
    ax.invert_yaxis()

    fig.colorbar(surf, ax=ax, shrink=0.5, label="Relative depth")
    fig.tight_layout()

    return fig


def create_comparison_figure(
    original: np.ndarray,
    depth_map: np.ndarray,
    title: str = "Shroud Analysis Pipeline",
) -> plt.Figure:
    """Create a side-by-side comparison: original → depth map → 3D surface."""
    fig = plt.figure(figsize=(18, 6))

    # Original face crop
    ax1 = fig.add_subplot(131)
    if len(original.shape) == 3:
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(original, cmap="gray")
    ax1.set_title("Face Crop (Negative)")
    ax1.axis("off")

    # Depth map
    ax2 = fig.add_subplot(132)
    ax2.imshow(depth_map, cmap="inferno")
    ax2.set_title("Depth Map\n(brighter = closer to body)")
    ax2.axis("off")

    # 3D surface (embedded)
    ax3 = fig.add_subplot(133, projection="3d")
    small = depth_map[::4, ::4].astype(float)
    h, w = small.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    ax3.plot_surface(X, Y, small, cmap="bone", linewidth=0, antialiased=True,
                     rstride=1, cstride=1)
    ax3.set_title("3D Relief (VP-8 Style)")
    ax3.view_init(elev=30, azim=-60)
    ax3.invert_yaxis()
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_zlabel("")
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    return fig


def create_depth_heatmap(depth_map: np.ndarray) -> plt.Figure:
    """Create a color-coded heatmap of depth values."""
    fig, ax = plt.subplots(figsize=(8, 10))

    im = ax.imshow(depth_map, cmap="inferno", aspect="equal")
    ax.set_title("Shroud Depth Heatmap\n(brighter/yellow = closer to body)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Relative depth")
    fig.tight_layout()

    return fig


def run_visualization():
    """Generate all visualizations."""
    print("=== Shroud 3D Visualization ===\n")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load depth map
    depth_path = PROCESSED_DIR / "depth_map.npy"
    if not depth_path.exists():
        print("Depth map not found. Run depth_map.py first.")
        return

    depth = np.load(str(depth_path))
    print(f"Loaded depth map: {depth.shape}")

    # Load face crop for comparison
    face_path = PROCESSED_DIR / "face_crop.png"
    face_img = cv2.imread(str(face_path)) if face_path.exists() else None

    # 1. VP-8 style 3D surface plot
    print("Generating 3D surface plot...")
    fig1 = create_3d_surface_plot(depth)
    fig1.savefig(str(PLOTS_DIR / "vp8_3d_surface.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOTS_DIR / 'vp8_3d_surface.png'}")

    # Additional angle
    fig1b = create_3d_surface_plot(depth, title="Front View", elevation=0, azimuth=-90)
    fig1b.savefig(str(PLOTS_DIR / "vp8_3d_front.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOTS_DIR / 'vp8_3d_front.png'}")

    # 2. Depth heatmap
    print("Generating depth heatmap...")
    fig2 = create_depth_heatmap(depth)
    fig2.savefig(str(PLOTS_DIR / "depth_heatmap.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOTS_DIR / 'depth_heatmap.png'}")

    # 3. Comparison figure
    if face_img is not None:
        print("Generating comparison figure...")
        fig3 = create_comparison_figure(face_img, depth)
        fig3.savefig(str(PLOTS_DIR / "pipeline_comparison.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: {PLOTS_DIR / 'pipeline_comparison.png'}")

    plt.close("all")
    print("\nVisualization complete.")


if __name__ == "__main__":
    run_visualization()

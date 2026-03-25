"""
Shroud of Turin — Forensic Reconstruction Pipeline
Main entry point: runs all extraction steps in sequence.

Usage:
    python run_pipeline.py           # Run full pipeline
    python run_pipeline.py download  # Just download images
    python run_pipeline.py depth     # Just depth map extraction
    python run_pipeline.py landmarks # Just landmark detection
    python run_pipeline.py measure   # Just measurements
    python run_pipeline.py visualize # Just visualization
"""

import sys
from src.download import download_all
from src.depth_map import run_depth_extraction
from src.landmarks import run_landmark_detection
from src.measurements import run_measurements
from src.visualize import run_visualization


def run_full_pipeline():
    """Run the complete extraction pipeline."""
    print("=" * 60)
    print("  SHROUD OF TURIN — FORENSIC DATA EXTRACTION PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Download source images
    download_all()
    print("\n" + "-" * 60 + "\n")

    # Step 2: Extract depth map
    depth = run_depth_extraction()
    print("\n" + "-" * 60 + "\n")

    # Step 3: Detect facial landmarks
    landmarks = run_landmark_detection()
    print("\n" + "-" * 60 + "\n")

    # Step 4: Calculate measurements
    if landmarks is not None:
        measurements = run_measurements()
        print("\n" + "-" * 60 + "\n")
    else:
        print("Skipping measurements (no landmarks detected).\n")

    # Step 5: Generate visualizations
    run_visualization()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print("\nOutput locations:")
    print("  data/processed/    — Depth maps and processed images")
    print("  data/measurements/ — Landmarks and measurements JSON")
    print("  output/plots/      — Visualizations")


STEPS = {
    "download": download_all,
    "depth": run_depth_extraction,
    "landmarks": run_landmark_detection,
    "measure": run_measurements,
    "visualize": run_visualization,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        step = sys.argv[1].lower()
        if step in STEPS:
            STEPS[step]()
        else:
            print(f"Unknown step: {step}")
            print(f"Available: {', '.join(STEPS.keys())}")
    else:
        run_full_pipeline()

"""
Star Tracker Attitude Determination Pipeline — Main Orchestrator

This is the entry point for the complete star tracker pipeline.
It chains all modules together: Image → Preprocess → Detect → Vectors →
Triangle Match → QUEST → Output.

Usage:
    # Process a single image
    python main.py --image path/to/image.fits

    # Run validation on synthetic test images
    python main.py --validate

    # Run validation on SkyView images
    python main.py --validate-skyview

    # Generate synthetic images
    python main.py --generate N

    # Run occlusion experiment
    python main.py --occlusion

    # Full pipeline demo
    python main.py --demo
"""

import os
import sys
import time
import argparse
import logging
import numpy as np

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config


def setup_logging(verbose=False):
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def load_resources():
    """Load catalogue and triangle database."""
    from catalogue.hipparcos import load_catalogue
    from modules.m5_triangle_match import load_triangle_database

    print("Loading star catalogue...")
    catalogue = load_catalogue(config.CATALOGUE_FILE)
    print(f"  ✓ {len(catalogue['hip_ids'])} stars loaded")

    print("Loading triangle database...")
    tri_db = load_triangle_database(config.TRIANGLE_DB_FILE)
    print(f"  ✓ {len(tri_db.pair_distances)} pair distances loaded")

    return catalogue, tri_db


def process_single_image(filepath, catalogue, tri_db, verbose=False):
    """
    Run the full pipeline on a single FITS image.

    Parameters
    ----------
    filepath : str
        Path to the FITS image
    catalogue : dict
        Loaded star catalogue
    tri_db : TriangleDatabase
        Triangle matching database
    verbose : bool
        Print detailed output

    Returns
    -------
    AttitudeResult or None
    """
    from modules.m1_image_input import load_fits_image
    from modules.m2_preprocessing import preprocess_image
    from modules.m3_star_detection import detect_stars
    from modules.m4_pixel_to_vector import convert_pixels_to_vectors
    from modules.m5_triangle_match import match_stars
    from modules.m7_quest import quest_from_matches
    from modules.m9_output import format_output
    from camera.camera_model import StarTrackerCamera
    from validation.ground_truth import extract_ground_truth

    camera = StarTrackerCamera()
    start = time.time()

    # Module 1: Load image
    print(f"\n{'─'*60}")
    print(f"  Processing: {os.path.basename(filepath)}")
    print(f"{'─'*60}")

    fits_image = load_fits_image(filepath)
    print(f"  [M1] Image loaded: {fits_image.width}×{fits_image.height}, "
          f"range=[{fits_image.data.min():.0f}, {fits_image.data.max():.0f}]")

    # Extract ground truth if available
    gt = extract_ground_truth(fits_image)
    if gt:
        print(f"  [GT] Ground truth: RA={gt['ra_deg']:.4f}°, Dec={gt['dec_deg']:.4f}°")

    # Module 2: Preprocess
    cleaned = preprocess_image(fits_image.data)
    print(f"  [M2] Preprocessed: range=[{cleaned.min():.1f}, {cleaned.max():.1f}]")

    # Module 3: Detect stars
    detected_stars = detect_stars(cleaned)
    print(f"  [M3] Detected: {len(detected_stars)} stars")

    if len(detected_stars) < config.MIN_STARS_FOR_MATCH:
        print(f"  ✗ Too few stars for identification "
              f"(need ≥{config.MIN_STARS_FOR_MATCH})")
        return None

    if verbose:
        for i, s in enumerate(detected_stars[:5]):
            print(f"       Star {i}: ({s.u:.1f}, {s.v:.1f}), "
                  f"mag={s.instrumental_mag:.2f}, SNR={s.snr:.1f}")

    # Module 4: Pixel → unit vectors
    star_vectors = convert_pixels_to_vectors(detected_stars, camera)
    print(f"  [M4] Computed {len(star_vectors)} body-frame vectors")

    # Module 5: Triangle matching
    matches, success = match_stars(star_vectors, tri_db, catalogue)
    print(f"  [M5] Triangle matching: {'SUCCESS' if success else 'FAILED'} "
          f"({len(matches)} matches)")

    # Module 6: GNN fallback
    if not success:
        print(f"  [M6] Triangle failed → trying GNN fallback…")
        try:
            from modules.m6_gnn import gnn_identify_stars, TORCH_AVAILABLE
            if TORCH_AVAILABLE:
                matches, success = gnn_identify_stars(
                    detected_stars, camera, catalogue, tri_db)
                print(f"  [M6] GNN: {'SUCCESS' if success else 'FAILED'} "
                      f"({len(matches)} matches)")
            else:
                print(f"  [M6] GNN unavailable (PyTorch not installed)")
        except Exception as e:
            print(f"  [M6] GNN error: {e}")

    if not success:
        print(f"  ✗ Star identification failed (both M5 and M6)")
        return None

    # Module 7: QUEST
    quest_result = quest_from_matches(star_vectors, matches, catalogue)
    print(f"  [M7] QUEST: residual = {quest_result['residual_arcsec']:.2f} arcsec")

    # Module 9: Format output
    gt_q = gt['quaternion'] if gt else None
    attitude = format_output(quest_result, method='triangle', ground_truth_q=gt_q)

    elapsed = time.time() - start

    # Print result
    print(f"\n{attitude}")
    print(f"  Processing time: {elapsed:.3f} seconds")

    return attitude


def cmd_demo(args):
    """Run a demo with a synthetic image."""
    from catalogue.hipparcos import load_catalogue
    from synthetic.image_generator import generate_synthetic_image, save_synthetic_image
    from camera.camera_model import StarTrackerCamera

    catalogue, tri_db = load_resources()

    # Generate a demo image pointing at a well-known region
    print("\nGenerating demo image...")
    print("  Pointing at: Orion's Belt region (RA=84°, Dec=-1°)")

    result = generate_synthetic_image(
        catalogue,
        ra_deg=84.0,     # Near Orion's Belt
        dec_deg=-1.0,
        roll_deg=0.0,
        occlusion_fraction=0.0,
    )

    demo_path = os.path.join(config.DATA_DIR, "demo_image.fits")
    save_synthetic_image(result, demo_path)
    print(f"  ✓ Demo image saved: {demo_path}")
    print(f"    {result['n_stars_rendered']} stars rendered")

    # Process it
    attitude = process_single_image(demo_path, catalogue, tri_db, verbose=True)

    if attitude and result['true_quaternion'] is not None:
        from catalogue.catalogue_utils import quaternion_angle
        error = quaternion_angle(attitude.quaternion, result['true_quaternion'])
        print(f"\n  ★ Demo result: {error*3600:.2f} arcsec error")


def cmd_process(args):
    """Process a single image."""
    if not os.path.exists(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    catalogue, tri_db = load_resources()
    process_single_image(args.image, catalogue, tri_db, verbose=args.verbose)


def cmd_validate(args):
    """Run validation suite."""
    from validation.run_validation import run_validation

    catalogue, tri_db = load_resources()
    image_dir = args.image_dir or config.SYNTHETIC_TEST_DIR

    print(f"\nRunning validation on: {image_dir}")
    run_validation(catalogue, tri_db, image_dir=image_dir)


def cmd_generate(args):
    """Generate synthetic images."""
    from catalogue.hipparcos import load_catalogue
    from synthetic.image_generator import generate_dataset

    catalogue = load_catalogue(config.CATALOGUE_FILE)

    n = args.count
    output_dir = args.output_dir or config.SYNTHETIC_TEST_DIR
    occ = args.occlusion or 0.0

    print(f"\nGenerating {n} synthetic images...")
    print(f"  Output: {output_dir}")
    print(f"  Occlusion: {occ:.0%}")

    generate_dataset(catalogue, output_dir, n, occlusion_fraction=occ)


def cmd_occlusion(args):
    """Run occlusion experiment."""
    from validation.occlusion_experiment import run_occlusion_experiment

    catalogue, tri_db = load_resources()
    image_dir = args.image_dir or config.SYNTHETIC_TEST_DIR

    print(f"\nRunning occlusion experiment on: {image_dir}")
    run_occlusion_experiment(catalogue, tri_db, image_dir=image_dir)


def cmd_skyview(args):
    """Download and validate with SkyView images."""
    from validation.skyview_fetcher import fetch_skyview_images
    from validation.run_validation import run_validation

    n = args.count or config.SKYVIEW_NUM_IMAGES

    print(f"\nDownloading {n} SkyView images...")
    metadata = fetch_skyview_images(n_images=n)
    print(f"  Downloaded {len(metadata)} images")

    if args.validate:
        catalogue, tri_db = load_resources()
        print("\nRunning validation on SkyView images...")
        run_validation(catalogue, tri_db, image_dir=config.SKYVIEW_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Star Tracker Attitude Determination Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    Run a demo with synthetic image
  python main.py --image test.fits         Process a single FITS image
  python main.py --validate                Validate on synthetic test images
  python main.py --generate 50             Generate 50 synthetic images
  python main.py --occlusion               Run occlusion experiment
  python main.py --skyview --count 10      Download 10 SkyView images
        """
    )

    parser.add_argument('--image', type=str, help='Path to FITS image to process')
    parser.add_argument('--demo', action='store_true', help='Run demo pipeline')
    parser.add_argument('--validate', action='store_true', help='Run validation suite')
    parser.add_argument('--generate', type=int, metavar='N', help='Generate N synthetic images')
    parser.add_argument('--occlusion', action='store_true', help='Run occlusion experiment')
    parser.add_argument('--skyview', action='store_true', help='Download SkyView images')
    parser.add_argument('--image-dir', type=str, help='Image directory for validation')
    parser.add_argument('--output-dir', type=str, help='Output directory for generation')
    parser.add_argument('--count', type=int, help='Number of images (for skyview/generate)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Dispatch to appropriate command
    if args.demo:
        cmd_demo(args)
    elif args.image:
        cmd_process(args)
    elif args.validate:
        cmd_validate(args)
    elif args.generate:
        args.count = args.generate
        cmd_generate(args)
    elif args.occlusion:
        cmd_occlusion(args)
    elif args.skyview:
        args.validate = True
        cmd_skyview(args)
    else:
        parser.print_help()
        print("\n  Tip: Run 'python setup_data.py' first, then 'python main.py --demo'")


if __name__ == "__main__":
    main()

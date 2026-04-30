"""
Validation Suite — Run Pipeline on Test Images

Runs the full star tracker pipeline on synthetic and/or SkyView
images and records accuracy metrics.
"""

import os
import csv
import logging
import numpy as np
import glob

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def run_validation(catalogue, tri_db, image_dir=None, output_csv=None):
    """
    Run the full pipeline on all FITS images in a directory
    and record results.

    Parameters
    ----------
    catalogue : dict
        Loaded Hipparcos catalogue
    tri_db : TriangleDatabase
        Precomputed triangle matching database
    image_dir : str
        Directory containing FITS test images
    output_csv : str
        Path to save results CSV

    Returns
    -------
    list of dict
        Result records for each image
    """
    from modules.m1_image_input import load_fits_image
    from modules.m2_preprocessing import preprocess_image
    from modules.m3_star_detection import detect_stars
    from modules.m4_pixel_to_vector import convert_pixels_to_vectors
    from modules.m5_triangle_match import match_stars
    from modules.m7_quest import quest_from_matches
    from modules.m9_output import format_output
    from camera.camera_model import StarTrackerCamera
    from validation.ground_truth import extract_ground_truth, compute_pointing_error

    if image_dir is None:
        image_dir = config.SYNTHETIC_TEST_DIR
    if output_csv is None:
        output_csv = os.path.join(config.RESULTS_DIR, "validation_results.csv")

    # Find all FITS files
    fits_files = sorted(glob.glob(os.path.join(image_dir, "*.fits")))
    if not fits_files:
        logger.warning(f"No FITS files found in {image_dir}")
        return []

    logger.info(f"Running validation on {len(fits_files)} images from {image_dir}...")

    camera = StarTrackerCamera()
    results = []

    for i, filepath in enumerate(fits_files):
        filename = os.path.basename(filepath)
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(fits_files)}] Processing: {filename}")
        logger.info(f"{'='*60}")

        record = {
            'filename': filename,
            'n_detected': 0,
            'n_matched': 0,
            'method': 'none',
            'success': False,
            'angular_error_arcsec': np.nan,
            'ra_error_arcsec': np.nan,
            'dec_error_arcsec': np.nan,
            'residual_arcsec': np.nan,
        }

        try:
            # Module 1: Load image
            fits_image = load_fits_image(filepath)

            # Extract ground truth
            gt = extract_ground_truth(fits_image)

            # Module 2: Preprocess
            cleaned = preprocess_image(fits_image.data)

            # Module 3: Detect stars
            detected_stars = detect_stars(cleaned)
            record['n_detected'] = len(detected_stars)

            if len(detected_stars) < config.MIN_STARS_FOR_MATCH:
                logger.warning(f"Only {len(detected_stars)} stars detected — skipping")
                results.append(record)
                continue

            # Module 4: Pixel → vectors
            star_vectors = convert_pixels_to_vectors(detected_stars, camera)

            # Module 5: Triangle matching
            matches, success = match_stars(star_vectors, tri_db, catalogue)
            record['n_matched'] = len(matches)

            if not success:
                logger.warning(f"Triangle matching failed")
                record['method'] = 'triangle_failed'
                results.append(record)
                continue

            record['method'] = 'triangle'
            record['success'] = True

            # Module 7: QUEST
            quest_result = quest_from_matches(star_vectors, matches, catalogue)

            # Module 9: Format output
            gt_q = gt['quaternion'] if gt else None
            attitude = format_output(quest_result, method='triangle',
                                     ground_truth_q=gt_q)

            record['residual_arcsec'] = attitude.residual_arcsec

            # Compute errors vs ground truth
            if gt:
                errors = compute_pointing_error(attitude.quaternion, gt)
                record['angular_error_arcsec'] = errors['angular_error_arcsec']
                record['ra_error_arcsec'] = errors['ra_error_arcsec']
                record['dec_error_arcsec'] = errors['dec_error_arcsec']

            logger.info(f"Result: {attitude.method}, "
                        f"{attitude.n_stars_used} stars, "
                        f"error={record['angular_error_arcsec']:.1f} arcsec")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)

        results.append(record)

    # Save results to CSV
    _save_results_csv(results, output_csv)

    # Print summary statistics
    _print_summary(results)

    return results


def _save_results_csv(results, output_csv):
    """Save validation results to CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = ['filename', 'n_detected', 'n_matched', 'method', 'success',
                  'angular_error_arcsec', 'ra_error_arcsec', 'dec_error_arcsec',
                  'residual_arcsec']

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {output_csv}")


def _print_summary(results):
    """Print summary statistics of validation results."""
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    errors = [r['angular_error_arcsec'] for r in successful
              if not np.isnan(r['angular_error_arcsec'])]

    print("\n" + "=" * 60)
    print("           VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total images:      {len(results)}")
    print(f"  Successful:        {len(successful)} ({100*len(successful)/max(1,len(results)):.1f}%)")
    print(f"  Failed:            {len(failed)}")

    if errors:
        errors = np.array(errors)
        print(f"\n  Angular Error (arcsec):")
        print(f"    Mean:    {errors.mean():.2f}")
        print(f"    Median:  {np.median(errors):.2f}")
        print(f"    Std:     {errors.std():.2f}")
        print(f"    Min:     {errors.min():.2f}")
        print(f"    Max:     {errors.max():.2f}")
        print(f"    < 60\":   {np.sum(errors < 60)} ({100*np.sum(errors<60)/len(errors):.1f}%)")
        print(f"    < 10\":   {np.sum(errors < 10)} ({100*np.sum(errors<10)/len(errors):.1f}%)")

    print("=" * 60)

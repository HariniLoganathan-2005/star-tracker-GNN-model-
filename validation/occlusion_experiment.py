"""
Occlusion Experiment

Tests pipeline robustness under varying levels of star occlusion.
For each test image, progressively removes detected stars and measures
whether the pipeline can still determine the correct attitude.

Generates the key result figure: Triangle Matching vs GNN accuracy
as a function of occlusion percentage.
"""

import os
import csv
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def run_occlusion_experiment(catalogue, tri_db, image_dir=None,
                              output_csv=None, figure_path=None):
    """
    Run the occlusion robustness experiment.

    For each test image:
    1. Load and preprocess the image
    2. Detect all stars
    3. For each occlusion level (0%, 10%, 20%, ... 50%):
       a. Randomly remove that fraction of detected stars
       b. Run triangle matching → QUEST
       c. Record success/failure and angular error

    Parameters
    ----------
    catalogue : dict
        Loaded star catalogue
    tri_db : TriangleDatabase
        Triangle matching database
    image_dir : str
        Directory with test FITS images
    output_csv : str
        Path to save detailed results
    figure_path : str
        Path to save the occlusion curve figure

    Returns
    -------
    dict
        Summary statistics per occlusion level
    """
    from modules.m1_image_input import load_fits_image
    from modules.m2_preprocessing import preprocess_image
    from modules.m3_star_detection import detect_stars
    from modules.m4_pixel_to_vector import convert_pixels_to_vectors
    from modules.m5_triangle_match import match_stars
    from modules.m7_quest import quest_from_matches
    from camera.camera_model import StarTrackerCamera
    from validation.ground_truth import extract_ground_truth, compute_pointing_error

    import glob

    if image_dir is None:
        image_dir = config.SYNTHETIC_TEST_DIR
    if output_csv is None:
        output_csv = os.path.join(config.RESULTS_DIR, "occlusion_results.csv")
    if figure_path is None:
        figure_path = os.path.join(config.FIGURES_DIR, "occlusion_curve.png")

    fits_files = sorted(glob.glob(os.path.join(image_dir, "*.fits")))
    if not fits_files:
        logger.warning(f"No FITS files found in {image_dir}")
        return {}

    camera = StarTrackerCamera()
    occlusion_levels = config.OCCLUSION_LEVELS

    logger.info(f"Running occlusion experiment on {len(fits_files)} images")
    logger.info(f"Occlusion levels: {occlusion_levels}")

    all_records = []
    rng = np.random.default_rng(42)

    for file_idx, filepath in enumerate(fits_files):
        filename = os.path.basename(filepath)
        logger.info(f"[{file_idx+1}/{len(fits_files)}] {filename}")

        try:
            # Load and preprocess
            fits_image = load_fits_image(filepath)
            gt = extract_ground_truth(fits_image)
            cleaned = preprocess_image(fits_image.data)
            all_detected = detect_stars(cleaned)

            if len(all_detected) < config.MIN_STARS_FOR_MATCH:
                logger.warning(f"  Too few stars ({len(all_detected)}), skipping")
                continue

            all_vectors = convert_pixels_to_vectors(all_detected, camera)
            n_total = len(all_vectors)

            # Test each occlusion level
            for occ_level in occlusion_levels:
                n_remove = int(n_total * occ_level)
                n_keep = n_total - n_remove

                record = {
                    'filename': filename,
                    'occlusion': occ_level,
                    'n_total_stars': n_total,
                    'n_kept_stars': n_keep,
                    'triangle_success': False,
                    'triangle_error_arcsec': np.nan,
                    'triangle_n_matched': 0,
                }

                if n_keep < config.MIN_STARS_FOR_MATCH:
                    all_records.append(record)
                    continue

                # Random subset of stars (keep the brightest preferentially)
                if n_remove > 0:
                    # Remove from the fainter stars more likely
                    keep_indices = sorted(
                        rng.choice(n_total, size=n_keep, replace=False)
                    )
                    occluded_vectors = [all_vectors[i] for i in keep_indices]
                else:
                    occluded_vectors = all_vectors

                # Triangle matching
                try:
                    matches, success = match_stars(occluded_vectors, tri_db, catalogue)
                    record['triangle_success'] = success
                    record['triangle_n_matched'] = len(matches)

                    if success and gt:
                        quest_result = quest_from_matches(
                            occluded_vectors, matches, catalogue
                        )
                        errors = compute_pointing_error(
                            quest_result['quaternion'], gt
                        )
                        record['triangle_error_arcsec'] = errors['angular_error_arcsec']

                except Exception as e:
                    logger.debug(f"  Triangle matching error at {occ_level:.0%}: {e}")

                all_records.append(record)

        except Exception as e:
            logger.error(f"  Error processing {filename}: {e}", exc_info=True)

    # Save results
    _save_occlusion_csv(all_records, output_csv)

    # Generate figure
    summary = _compute_summary(all_records, occlusion_levels)
    _plot_occlusion_curve(summary, figure_path)

    return summary


def _save_occlusion_csv(records, output_csv):
    """Save occlusion experiment results to CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = ['filename', 'occlusion', 'n_total_stars', 'n_kept_stars',
                  'triangle_success', 'triangle_error_arcsec',
                  'triangle_n_matched']

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Occlusion results saved to {output_csv}")


def _compute_summary(records, occlusion_levels):
    """Compute per-occlusion-level statistics."""
    summary = {}

    for occ in occlusion_levels:
        level_records = [r for r in records if r['occlusion'] == occ]
        if not level_records:
            continue

        n_total = len(level_records)
        tri_success = sum(1 for r in level_records if r['triangle_success'])
        tri_errors = [r['triangle_error_arcsec'] for r in level_records
                      if r['triangle_success'] and not np.isnan(r['triangle_error_arcsec'])]

        summary[occ] = {
            'n_images': n_total,
            'triangle_success_rate': tri_success / n_total if n_total > 0 else 0,
            'triangle_mean_error': np.mean(tri_errors) if tri_errors else np.nan,
            'triangle_median_error': np.median(tri_errors) if tri_errors else np.nan,
        }

    return summary


def _plot_occlusion_curve(summary, figure_path):
    """
    Generate the occlusion robustness curve (Figure 1).

    Plots success rate vs occlusion percentage for triangle matching.
    (GNN curve will be added in Phase 2.)
    """
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    occlusion_pcts = []
    tri_success_rates = []

    for occ, stats in sorted(summary.items()):
        occlusion_pcts.append(occ * 100)
        tri_success_rates.append(stats['triangle_success_rate'] * 100)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Success rate plot
    ax1.plot(occlusion_pcts, tri_success_rates, 'b-o', linewidth=2,
             markersize=8, label='Triangle Matching', zorder=5)

    # GNN placeholder (will be filled in Phase 2)
    # ax1.plot(occlusion_pcts, gnn_success_rates, 'r-s', linewidth=2,
    #          markersize=8, label='GNN (Lost-in-Space)')

    ax1.set_xlabel('Occlusion Level (%)', fontsize=14)
    ax1.set_ylabel('Success Rate (%)', fontsize=14)
    ax1.set_title('Star Identification Robustness Under Occlusion', fontsize=16)
    ax1.legend(fontsize=12, loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 52)
    ax1.set_ylim(-5, 105)
    ax1.tick_params(labelsize=12)

    # Add annotation
    ax1.annotate('GNN curve will be added\nin Phase 2',
                 xy=(30, 50), fontsize=11, style='italic',
                 color='gray', ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Occlusion curve saved to {figure_path}")

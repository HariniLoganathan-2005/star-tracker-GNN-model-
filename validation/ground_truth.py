"""
Ground Truth Extraction

Extracts ground truth pointing from FITS headers:
- WCS headers (for SkyView images)
- Embedded quaternion metadata (for synthetic images)

Provides ground truth quaternions for validation comparison.
"""

import logging
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catalogue.catalogue_utils import pointing_to_quaternion

logger = logging.getLogger(__name__)


def extract_ground_truth(fits_image):
    """
    Extract ground truth attitude from a FITS image.

    For synthetic images: reads the embedded quaternion directly.
    For SkyView images: extracts WCS pointing and derives quaternion.

    Parameters
    ----------
    fits_image : FITSImage
        Loaded FITS image from Module 1

    Returns
    -------
    dict or None with keys:
        'quaternion': numpy.ndarray — ground truth quaternion
        'ra_deg': float — boresight RA
        'dec_deg': float — boresight Dec
        'roll_deg': float — boresight roll (0 if unknown)
        'source': str — 'synthetic' or 'wcs'
    """
    header = fits_image.header

    # Check if this is a synthetic image with embedded truth
    if header.get('SYNTHIMG', False):
        return _extract_synthetic_truth(header)

    # Try WCS-based ground truth
    if fits_image.has_wcs:
        return _extract_wcs_truth(fits_image)

    logger.warning("No ground truth available for this image")
    return None


def _extract_synthetic_truth(header):
    """Extract ground truth from synthetic image header."""
    try:
        q = np.array([
            header['TRUE_Q0'],
            header['TRUE_Q1'],
            header['TRUE_Q2'],
            header['TRUE_Q3'],
        ])

        ra = header.get('TRUE_RA', 0)
        dec = header.get('TRUE_DEC', 0)
        roll = header.get('TRUE_ROL', 0)

        logger.info(f"Synthetic ground truth: RA={ra:.4f}°, Dec={dec:.4f}°, "
                    f"Roll={roll:.4f}°")

        return {
            'quaternion': q,
            'ra_deg': float(ra),
            'dec_deg': float(dec),
            'roll_deg': float(roll),
            'source': 'synthetic',
        }
    except KeyError as e:
        logger.warning(f"Missing synthetic ground truth key: {e}")
        return None


def _extract_wcs_truth(fits_image):
    """Extract ground truth from WCS headers."""
    try:
        pointing = fits_image.get_ground_truth_pointing()
        if pointing is None:
            return None

        ra_deg, dec_deg = pointing

        # WCS doesn't directly give us roll, estimate from WCS rotation
        roll_deg = _estimate_roll_from_wcs(fits_image.wcs)

        # Construct quaternion from pointing
        q = pointing_to_quaternion(ra_deg, dec_deg, roll_deg)

        logger.info(f"WCS ground truth: RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°, "
                    f"Roll={roll_deg:.4f}°")

        return {
            'quaternion': q,
            'ra_deg': float(ra_deg),
            'dec_deg': float(dec_deg),
            'roll_deg': float(roll_deg),
            'source': 'wcs',
        }
    except Exception as e:
        logger.warning(f"Error extracting WCS ground truth: {e}")
        return None


def _estimate_roll_from_wcs(wcs):
    """
    Estimate the roll angle from WCS CD matrix or CROTA keywords.

    The CD matrix defines the transformation from pixel to world coordinates.
    The rotation angle can be extracted from it.
    """
    try:
        # Try CD matrix first
        if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
            cd = wcs.wcs.cd
            roll = np.degrees(np.arctan2(cd[0, 1], cd[0, 0]))
            return float(roll)

        # Try PC matrix + CDELT
        if hasattr(wcs.wcs, 'pc') and wcs.wcs.pc is not None:
            pc = wcs.wcs.pc
            roll = np.degrees(np.arctan2(pc[0, 1], pc[0, 0]))
            return float(roll)

        # Try CROTA2
        if hasattr(wcs.wcs, 'crota') and wcs.wcs.crota is not None:
            if len(wcs.wcs.crota) >= 2:
                return float(wcs.wcs.crota[1])

    except Exception:
        pass

    return 0.0  # Default to no roll


def compute_pointing_error(result_quaternion, ground_truth):
    """
    Compute the angular error between estimated and ground truth attitude.

    Parameters
    ----------
    result_quaternion : numpy.ndarray
        Estimated quaternion from pipeline
    ground_truth : dict
        Ground truth dict from extract_ground_truth()

    Returns
    -------
    dict with keys:
        'angular_error_deg': float — total angular error in degrees
        'angular_error_arcsec': float — total angular error in arcseconds
        'ra_error_arcsec': float — RA error in arcseconds
        'dec_error_arcsec': float — Dec error in arcseconds
    """
    from catalogue.catalogue_utils import quaternion_angle, quaternion_to_pointing

    gt_q = ground_truth['quaternion']

    # Total angular error (quaternion geodesic distance)
    ang_error_deg = quaternion_angle(result_quaternion, gt_q)
    ang_error_arcsec = ang_error_deg * 3600

    # Pointing error (RA/Dec of boresight)
    est_ra, est_dec = quaternion_to_pointing(result_quaternion)
    gt_ra = ground_truth['ra_deg']
    gt_dec = ground_truth['dec_deg']

    # RA error (corrected for cos(dec))
    ra_err = (est_ra - gt_ra)
    if ra_err > 180:
        ra_err -= 360
    elif ra_err < -180:
        ra_err += 360
    ra_err_arcsec = ra_err * 3600 * np.cos(np.radians(gt_dec))
    dec_err_arcsec = (est_dec - gt_dec) * 3600

    return {
        'angular_error_deg': float(ang_error_deg),
        'angular_error_arcsec': float(ang_error_arcsec),
        'ra_error_arcsec': float(ra_err_arcsec),
        'dec_error_arcsec': float(dec_err_arcsec),
    }

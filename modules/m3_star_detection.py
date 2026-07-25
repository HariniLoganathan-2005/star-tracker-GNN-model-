"""
Module 3 — Star Detection

Detects stars in a preprocessed image using threshold-based detection
with connected component labeling. For each detected star, computes
sub-pixel position via Gaussian centroiding and estimates instrumental
magnitude from total flux.

Rejects artifacts based on size, shape, and brightness constraints.
"""

import logging
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.optimize import curve_fit
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


@dataclass
class DetectedStar:
    """
    A detected star in the image.

    Attributes
    ----------
    u : float
        Sub-pixel column position (centroid)
    v : float
        Sub-pixel row position (centroid)
    flux : float
        Total integrated flux in ADU
    instrumental_mag : float
        Instrumental magnitude (-2.5 * log10(flux))
    peak_value : float
        Maximum pixel value in the star blob
    area : int
        Number of pixels in the blob
    quality : str
        Quality flag: 'good', 'saturated', 'faint', 'extended'
    snr : float
        Signal-to-noise ratio
    """
    u: float
    v: float
    flux: float
    instrumental_mag: float
    peak_value: float
    area: int
    quality: str = 'good'
    snr: float = 0.0


def _gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian function for centroid fitting."""
    x, y = coords
    g = offset + amplitude * np.exp(
        -((x - x0) ** 2 / (2 * sigma_x ** 2) +
          (y - y0) ** 2 / (2 * sigma_y ** 2))
    )
    return g.ravel()


def _gaussian_centroid(image, blob_slice, center_guess, half_size=None):
    """
    Fit a 2D Gaussian to a star blob for sub-pixel centroiding.

    Parameters
    ----------
    image : numpy.ndarray
        Full preprocessed image
    blob_slice : tuple of slices
        Bounding box of the star blob
    center_guess : tuple
        Initial guess for center (row, col) from center_of_mass
    half_size : int
        Half-size of the fitting box

    Returns
    -------
    tuple
        (u_centroid, v_centroid, sigma_x, sigma_y) or None if fit fails
    """
    if half_size is None:
        half_size = config.CENTROID_BOX_HALF_SIZE

    row_c, col_c = int(round(center_guess[0])), int(round(center_guess[1]))
    h, w = image.shape

    # Define fitting region (clipped to image bounds)
    r_min = max(0, row_c - half_size)
    r_max = min(h, row_c + half_size + 1)
    c_min = max(0, col_c - half_size)
    c_max = min(w, col_c + half_size + 1)

    cutout = image[r_min:r_max, c_min:c_max]

    if cutout.size < 9:  # Need at least 3×3 for a meaningful fit
        return None

    # Create coordinate grids
    y_grid, x_grid = np.mgrid[r_min:r_max, c_min:c_max]
    coords = (x_grid, y_grid)

    # Initial parameters: amplitude, x0, y0, sigma_x, sigma_y, offset
    amp_guess = cutout.max()
    offset_guess = cutout.min()
    sigma_guess = 1.5  # Typical PSF sigma for star trackers

    p0 = [amp_guess - offset_guess, col_c, row_c,
          sigma_guess, sigma_guess, offset_guess]

    # Bounds for the fit
    bounds_lower = [0, c_min, r_min, 0.3, 0.3, -np.inf]
    bounds_upper = [np.inf, c_max, r_max, half_size, half_size, np.inf]

    try:
        popt, _ = curve_fit(
            _gaussian_2d, coords, cutout.ravel(),
            p0=p0, bounds=(bounds_lower, bounds_upper),
            maxfev=1000
        )
        amplitude, x0, y0, sigma_x, sigma_y, offset = popt
        return x0, y0, sigma_x, sigma_y
    except (RuntimeError, ValueError):
        # Fall back to center of mass if Gaussian fit fails
        return None


def detect_stars(image, noise_sigma=None, threshold_sigma=None):
    if threshold_sigma is None:
        threshold_sigma = config.DETECTION_SIGMA_THRESHOLD

    h, w = image.shape
    logger.info(f"Detecting stars in {w}×{h} image "
                f"(threshold = {threshold_sigma}σ)...")

    # Step 1: Estimate noise
    if noise_sigma is None:
        from modules.m2_preprocessing import estimate_noise
        noise_sigma = estimate_noise(image)

    # Enforce physical noise floor: never go below the camera read noise.
    # After background subtraction, synthetic images are mostly zero so MAD
    # returns ~0, causing the threshold to collapse and 60k+ blobs to appear.
    noise_sigma = max(noise_sigma, config.READ_NOISE_SIGMA)

    threshold = threshold_sigma * noise_sigma
    logger.info(f"Detection threshold: {threshold:.4f} ADU "
                f"(noise σ={noise_sigma:.4f})")

    # Step 2: Create binary mask
    binary_mask = image > threshold

    # Step 3: Label connected components
    labeled_array, num_features = label(binary_mask)
    logger.info(f"Found {num_features} candidate blobs")

    # Step 4: Process each blob
    stars = []
    rejected = {'too_small': 0, 'too_large': 0, 'elongated': 0, 'faint': 0}

    for blob_id in range(1, num_features + 1):
        blob_mask = labeled_array == blob_id
        blob_pixels = image[blob_mask]

        # Blob properties
        area = blob_pixels.size
        flux = np.sum(blob_pixels)
        peak_value = np.max(blob_pixels)

        # --- Quality Filter: Size ---
        if area < config.MIN_STAR_PIXELS:
            rejected['too_small'] += 1
            continue
        if area > config.MAX_STAR_PIXELS:
            rejected['too_large'] += 1
            continue

        # --- Quality Filter: Shape (eccentricity) ---
        blob_coords = np.argwhere(blob_mask)  # (row, col) pairs
        if len(blob_coords) >= 4:
            # Compute second central moments to check elongation
            centroid_rc = blob_coords.mean(axis=0)
            centered = blob_coords - centroid_rc
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]

            if eigenvalues[1] > 0:
                eccentricity = np.sqrt(1.0 - eigenvalues[1] / eigenvalues[0])
            else:
                eccentricity = 1.0

            if eccentricity > config.MAX_ECCENTRICITY:
                rejected['elongated'] += 1
                continue

        # --- Centroiding ---
        # Initial estimate: center of mass (weighted by intensity)
        com = center_of_mass(image, labeled_array, blob_id)

        # Attempt Gaussian centroid for sub-pixel accuracy
        gauss_result = _gaussian_centroid(image, None, com)

        if gauss_result is not None:
            u_centroid, v_centroid, sigma_x, sigma_y = gauss_result
        else:
            # Fall back to flux-weighted center of mass
            u_centroid = com[1]  # column
            v_centroid = com[0]  # row

        # --- Compute instrumental magnitude ---
        # m_instr = -2.5 * log10(flux)
        if flux > 0:
            instrumental_mag = -2.5 * np.log10(flux)
        else:
            rejected['faint'] += 1
            continue

        # --- Signal-to-noise ratio ---
        snr = peak_value / noise_sigma

        # --- Determine quality flag ---
        if peak_value >= config.SATURATION_LEVEL * 0.9:
            quality = 'saturated'
        elif snr < threshold_sigma * 2:
            quality = 'marginal'
        else:
            quality = 'good'

        star = DetectedStar(
            u=float(u_centroid),
            v=float(v_centroid),
            flux=float(flux),
            instrumental_mag=float(instrumental_mag),
            peak_value=float(peak_value),
            area=int(area),
            quality=quality,
            snr=float(snr),
        )
        stars.append(star)

    # Sort by brightness (brightest first = most negative instrumental mag)
    stars.sort(key=lambda s: s.instrumental_mag)

    logger.info(f"Detected {len(stars)} stars after filtering")
    logger.info(f"Rejected: {rejected}")

    if stars:
        logger.info(f"Brightest star: ({stars[0].u:.2f}, {stars[0].v:.2f}), "
                     f"mag={stars[0].instrumental_mag:.2f}, SNR={stars[0].snr:.1f}")
        logger.info(f"Faintest star: ({stars[-1].u:.2f}, {stars[-1].v:.2f}), "
                     f"mag={stars[-1].instrumental_mag:.2f}, SNR={stars[-1].snr:.1f}")

    return stars

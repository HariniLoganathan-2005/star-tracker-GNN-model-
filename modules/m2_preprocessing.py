"""
Module 2 — Image Preprocessing

Cleans raw star tracker images through three operations:
1. Dark frame subtraction — removes thermal noise pattern
2. Flat field correction — normalizes pixel sensitivity variations
3. Background removal — subtracts residual sky/sensor glow

After preprocessing, the background is near zero and only genuine
star signals remain as sharp peaks above the noise floor.
"""

import logging
import numpy as np
from scipy.ndimage import median_filter, uniform_filter

logger = logging.getLogger(__name__)


def subtract_dark_frame(image, dark_frame):
    """
    Subtract a dark frame (thermal noise reference) from the science image.

    A dark frame is captured with the lens covered — it records only sensor
    noise patterns (hot pixels, thermal gradients). Subtracting it removes
    these systematic patterns from the science image.

    Parameters
    ----------
    image : numpy.ndarray
        2D science image array
    dark_frame : numpy.ndarray
        2D dark frame array (same dimensions as image)

    Returns
    -------
    numpy.ndarray
        Dark-corrected image (clipped to ≥ 0)
    """
    if image.shape != dark_frame.shape:
        raise ValueError(f"Dark frame shape {dark_frame.shape} doesn't match "
                         f"image shape {image.shape}")

    corrected = image - dark_frame
    corrected = np.clip(corrected, 0, None)

    logger.info(f"Dark frame subtracted. Removed mean={dark_frame.mean():.2f} ADU")
    return corrected


def apply_flat_field(image, flat_field):
    """
    Apply flat field correction to normalize pixel-to-pixel sensitivity.

    A flat field image is captured of a uniformly illuminated surface.
    Dividing the science image by the normalized flat field corrects for
    non-uniform pixel responses (vignetting, dust, chip defects).

    Parameters
    ----------
    image : numpy.ndarray
        2D image array (already dark-subtracted)
    flat_field : numpy.ndarray
        2D flat field image (same dimensions)

    Returns
    -------
    numpy.ndarray
        Flat-corrected image
    """
    if image.shape != flat_field.shape:
        raise ValueError(f"Flat field shape {flat_field.shape} doesn't match "
                         f"image shape {image.shape}")

    # Normalize flat field to have mean = 1
    flat_norm = flat_field / np.mean(flat_field)

    # Avoid division by zero or very small flat values
    flat_norm = np.clip(flat_norm, 0.1, None)

    corrected = image / flat_norm

    logger.info(f"Flat field applied. Flat range: [{flat_norm.min():.3f}, "
                f"{flat_norm.max():.3f}]")
    return corrected


def estimate_background(image, box_size=64, filter_size=3):
    """
    Estimate the image background using a block-median approach.

    The image is divided into boxes of size `box_size × box_size`.
    The median of each box is computed, creating a low-resolution background map.
    This map is then interpolated back to the full image size.

    This is much faster than a full median_filter on large images,
    and effectively separates the slowly varying background from stars.

    Parameters
    ----------
    image : numpy.ndarray
        2D image array
    box_size : int
        Size of the boxes for background estimation
    filter_size : int
        Not used in block approach, kept for API compatibility

    Returns
    -------
    numpy.ndarray
        Estimated background map (same shape as input image)
    """
    from scipy.ndimage import zoom

    h, w = image.shape

    # Compute number of blocks
    n_rows = max(1, h // box_size)
    n_cols = max(1, w // box_size)

    # Compute median in each block
    bg_low = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            r_start = i * box_size
            r_end = min((i + 1) * box_size, h)
            c_start = j * box_size
            c_end = min((j + 1) * box_size, w)

            block = image[r_start:r_end, c_start:c_end]
            bg_low[i, j] = np.median(block)

    # Interpolate back to full resolution using zoom
    zoom_r = h / n_rows
    zoom_c = w / n_cols
    background = zoom(bg_low, (zoom_r, zoom_c), order=1)

    # Ensure same shape (zoom can be off by 1 pixel)
    background = background[:h, :w]

    return background


def remove_background(image, box_size=64, filter_size=3):
    """
    Remove the background from the image.

    Parameters
    ----------
    image : numpy.ndarray
        2D image array
    box_size : int
        Box size for background estimation
    filter_size : int
        Smoothing parameter for background map

    Returns
    -------
    numpy.ndarray
        Background-subtracted image
    """
    background = estimate_background(image, box_size, filter_size)
    subtracted = image - background
    subtracted = np.clip(subtracted, 0, None)

    logger.info(f"Background removed. Background mean={background.mean():.2f}, "
                f"std={background.std():.2f}")
    return subtracted


def estimate_noise(image):
    """
    Estimate the noise level (σ) of the image.

    Uses a robust approach that works on both raw and background-subtracted
    images:
    1. First tries Median Absolute Deviation (MAD) on all pixels
    2. If MAD is zero (common after background subtraction where most
       pixels are exactly 0), estimates from the standard deviation of
       non-star pixels (those below the 90th percentile)
    """
    # Method 1: MAD-based estimate
    median_val = np.median(image)
    mad = np.median(np.abs(image - median_val))
    sigma = 1.4826 * mad

    if sigma > 1e-10:
        logger.info(f"Estimated noise (MAD): σ={sigma:.4f} ADU")
        return sigma

    # Method 2: If MAD is zero, use std of low-value pixels
    # This happens when background subtraction leaves most pixels at 0
    # Use pixels below the 90th percentile (avoids star pixels)
    p90 = np.percentile(image, 90)
    if p90 > 0:
        low_pixels = image[image <= p90]
        sigma = np.std(low_pixels)
        if sigma > 1e-10:
            logger.info(f"Estimated noise (std of low pixels): σ={sigma:.4f} ADU")
            return sigma

    # Method 3: Use the standard deviation of all non-zero pixels
    nonzero = image[image > 0]
    if len(nonzero) > 100:
        # Use the lower half of non-zero values to avoid star pixels
        threshold = np.median(nonzero)
        low_nonzero = nonzero[nonzero < threshold]
        if len(low_nonzero) > 10:
            sigma = np.std(low_nonzero)
            if sigma > 1e-10:
                logger.info(f"Estimated noise (low non-zero): σ={sigma:.4f} ADU")
                return sigma

    # Method 4: Fallback — use image std / 10 as conservative estimate
    sigma = max(np.std(image) / 10.0, 1.0)
    logger.info(f"Estimated noise (fallback): σ={sigma:.4f} ADU")
    return sigma


def preprocess_image(image, dark_frame=None, flat_field=None, box_size=64):
    logger.info(f"Preprocessing image of shape {image.shape}...")
    result = image.astype(np.float64).copy()
    # Step 1: Dark frame subtraction
    if dark_frame is not None:
        result = subtract_dark_frame(result, dark_frame)
    else:
        logger.info("No dark frame provided — skipping dark subtraction")
    # Step 2: Flat field correction
    if flat_field is not None:
        result = apply_flat_field(result, flat_field)
    else:
        logger.info("No flat field provided — skipping flat correction")
    # Step 3: Background removal (always performed)
    result = remove_background(result, box_size=box_size)

    logger.info(f"Preprocessing complete. Output range: [{result.min():.2f}, "
                f"{result.max():.2f}], mean={result.mean():.4f}")

    return result

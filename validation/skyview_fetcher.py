"""
SkyView FITS Image Fetcher

Downloads real sky images from NASA SkyView service for validation.
Each image comes with WCS headers providing ground truth pointing.

SkyView docs: https://skyview.gsfc.nasa.gov/
"""

import os
import logging
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def fetch_skyview_images(output_dir=None, n_images=None, survey=None,
                         fov_deg=None, pixels=None, seed=42):
    """
    Download real sky images from NASA SkyView.

    Generates random sky positions and downloads FITS images with
    WCS headers for ground truth validation.

    Parameters
    ----------
    output_dir : str
        Directory to save downloaded FITS files
    n_images : int
        Number of images to download
    survey : str
        SkyView survey to use (e.g., 'DSS2 Red', 'DSS')
    fov_deg : float
        Field of view in degrees
    pixels : int
        Image size in pixels
    seed : int
        Random seed for reproducible sky positions

    Returns
    -------
    list of dict
        Metadata for each downloaded image
    """
    from astroquery.skyview import SkyView
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if output_dir is None:
        output_dir = config.SKYVIEW_DIR
    if n_images is None:
        n_images = config.SKYVIEW_NUM_IMAGES
    if survey is None:
        survey = config.SKYVIEW_SURVEY
    if fov_deg is None:
        fov_deg = config.SKYVIEW_FOV_DEG
    if pixels is None:
        pixels = config.SKYVIEW_IMAGE_PIXELS

    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Generate random sky positions
    # Uniform distribution on the sphere
    ra_values = rng.uniform(0, 360, n_images)
    dec_values = np.degrees(np.arcsin(rng.uniform(-1, 1, n_images)))

    # Avoid galactic plane (|b| < 10°) — too crowded for clean star matching
    # Also avoid very low declinations where DSS coverage may be patchy
    valid_positions = []
    attempts = 0
    while len(valid_positions) < n_images and attempts < n_images * 5:
        ra = rng.uniform(0, 360)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))

        # Check galactic latitude
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        gal_lat = coord.galactic.b.deg

        if abs(gal_lat) > 15:  # Avoid galactic plane
            valid_positions.append((ra, dec))

        attempts += 1

    if len(valid_positions) < n_images:
        logger.warning(f"Could only find {len(valid_positions)} valid positions "
                       f"(requested {n_images})")
        n_images = len(valid_positions)

    logger.info(f"Downloading {n_images} SkyView images ({survey})...")
    logger.info(f"FOV: {fov_deg}°, Pixels: {pixels}×{pixels}")

    metadata = []
    failed = 0

    for i, (ra, dec) in enumerate(valid_positions[:n_images]):
        filename = f"skyview_{i+1:03d}.fits"
        filepath = os.path.join(output_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            logger.debug(f"  Skipping {filename} (already exists)")
            metadata.append({
                'filename': filename,
                'filepath': filepath,
                'ra': ra,
                'dec': dec,
                'survey': survey,
            })
            continue

        try:
            position = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

            images = SkyView.get_images(
                position=position,
                survey=[survey],
                pixels=str(pixels),
                width=fov_deg * u.deg,
                height=fov_deg * u.deg,
            )

            if images and len(images) > 0:
                hdu_list = images[0]
                hdu_list.writeto(filepath, overwrite=True)

                meta = {
                    'filename': filename,
                    'filepath': filepath,
                    'ra': float(ra),
                    'dec': float(dec),
                    'survey': survey,
                }
                metadata.append(meta)

                logger.info(f"  [{i+1}/{n_images}] Downloaded: {filename} "
                            f"(RA={ra:.2f}°, Dec={dec:.2f}°)")
            else:
                logger.warning(f"  [{i+1}/{n_images}] No data for "
                               f"RA={ra:.2f}°, Dec={dec:.2f}°")
                failed += 1

        except Exception as e:
            logger.warning(f"  [{i+1}/{n_images}] Failed: {e}")
            failed += 1

    logger.info(f"Download complete: {len(metadata)} images saved, {failed} failed")
    return metadata

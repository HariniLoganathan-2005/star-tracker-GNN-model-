"""
Module 1 — Image Input

Loads FITS (Flexible Image Transport System) images into memory.
Handles both NASA SkyView FITS files and synthetic star tracker images.
Extracts metadata including WCS (World Coordinate System) headers when available.
"""

import os
import logging
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


class FITSImage:
    """
    Container for a loaded FITS image and its metadata.

    Attributes
    ----------
    data : numpy.ndarray
        2D array of pixel intensity values
    header : dict-like
        FITS header with metadata
    filepath : str
        Path to the source file
    wcs : astropy.wcs.WCS or None
        World Coordinate System object if WCS headers are present
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    has_wcs : bool
        Whether WCS information is available
    """

    def __init__(self, data, header, filepath):
        self.data = data
        self.header = header
        self.filepath = filepath
        self.width = data.shape[1]
        self.height = data.shape[0]
        self.wcs = None
        self.has_wcs = False

        # Try to extract WCS
        self._extract_wcs()

    def _extract_wcs(self):
        """Extract WCS information from FITS header if available."""
        try:
            from astropy.wcs import WCS
            wcs = WCS(self.header)
            # Verify WCS is valid by checking for required keys
            if wcs.has_celestial:
                self.wcs = wcs
                self.has_wcs = True
                logger.info("WCS information extracted from FITS header")
            else:
                logger.info("FITS header present but no celestial WCS found")
        except Exception as e:
            logger.debug(f"Could not extract WCS: {e}")

    def get_ground_truth_pointing(self):
        """
        Get the ground truth boresight pointing from WCS.

        Returns
        -------
        tuple or None
            (ra_deg, dec_deg) of the image center, or None if no WCS
        """
        if not self.has_wcs:
            return None

        # Get RA/Dec of the image center pixel
        center_x = self.width / 2.0
        center_y = self.height / 2.0

        try:
            sky = self.wcs.pixel_to_world(center_x, center_y)
            ra_deg = sky.ra.deg
            dec_deg = sky.dec.deg
            logger.info(f"Ground truth pointing: RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°")
            return ra_deg, dec_deg
        except Exception as e:
            logger.warning(f"Error computing ground truth pointing: {e}")
            return None

    def __repr__(self):
        wcs_str = "with WCS" if self.has_wcs else "no WCS"
        return (f"FITSImage({self.width}×{self.height}, "
                f"dtype={self.data.dtype}, {wcs_str}, "
                f"file={os.path.basename(self.filepath)})")


def load_fits_image(filepath):
    """
    Load a FITS image from disk.

    Handles multi-extension FITS files by selecting the first image extension.
    Converts data to float64 for processing.

    Parameters
    ----------
    filepath : str
        Path to the FITS file

    Returns
    -------
    FITSImage
        Loaded image container with data, header, and metadata

    Raises
    ------
    FileNotFoundError
        If the FITS file does not exist
    ValueError
        If no image data is found in the FITS file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    logger.info(f"Loading FITS image: {filepath}")

    with fits.open(filepath) as hdul:
        # Find the first extension with image data
        data = None
        header = None

        for i, hdu in enumerate(hdul):
            if hdu.data is not None and hdu.data.ndim >= 2:
                data = hdu.data
                header = hdu.header
                logger.info(f"Using HDU {i}: {hdu.name}, shape={data.shape}")
                break

        if data is None:
            raise ValueError(f"No image data found in FITS file: {filepath}")

        # Handle 3D data cubes — take the first slice
        if data.ndim == 3:
            logger.info(f"3D data cube detected, using first slice (shape was {data.shape})")
            data = data[0]

        # Convert to float64 for processing
        data = data.astype(np.float64)

        # Handle NaN values (common in SkyView images)
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            logger.info(f"Replacing {nan_count} NaN pixels with 0")
            data = np.nan_to_num(data, nan=0.0)

        # Handle negative values (can occur from calibration)
        neg_count = np.sum(data < 0)
        if neg_count > 0:
            logger.info(f"Clipping {neg_count} negative pixels to 0")
            data = np.clip(data, 0, None)

    image = FITSImage(data, header, filepath)
    logger.info(f"Loaded: {image}")
    logger.info(f"Data range: [{data.min():.2f}, {data.max():.2f}], "
                f"mean={data.mean():.2f}, std={data.std():.2f}")

    return image

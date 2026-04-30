"""
Module 4 — Pixel to Unit Vector Conversion

Converts each detected star's 2D pixel position into a 3D unit vector
representing its direction in the camera body frame.

Uses the camera intrinsic model (focal length, principal point, distortion)
to back-project pixel coordinates into rays in 3D space.
"""

import logging
import numpy as np
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class StarVector:
    """
    A detected star with both pixel and body-frame vector information.

    Attributes
    ----------
    u : float
        Sub-pixel column position
    v : float
        Sub-pixel row position
    bx, by, bz : float
        Unit vector components in body frame
    magnitude : float
        Instrumental magnitude
    flux : float
        Integrated flux
    snr : float
        Signal-to-noise ratio
    quality : str
        Quality flag
    """
    u: float
    v: float
    bx: float
    by: float
    bz: float
    magnitude: float
    flux: float
    snr: float
    quality: str = 'good'

    @property
    def body_vector(self):
        """Return the body-frame unit vector as a numpy array."""
        return np.array([self.bx, self.by, self.bz])


def convert_pixels_to_vectors(detected_stars, camera):
    """
    Convert a list of detected stars from pixel coordinates to
    unit vectors in the camera body frame.

    For each star at pixel position (u, v), the camera model computes:
        x = u - cx
        y = v - cy
        z = f  (focal length)
        After undistortion:
        unit_vector = normalize([x, y, z])

    Parameters
    ----------
    detected_stars : list of DetectedStar
        Stars from Module 3 with pixel positions
    camera : StarTrackerCamera
        Camera model with intrinsics and distortion parameters

    Returns
    -------
    list of StarVector
        Stars with both pixel and body-frame vector information
    """
    logger.info(f"Converting {len(detected_stars)} star positions to body-frame vectors...")

    star_vectors = []

    for star in detected_stars:
        # Use camera model to convert pixel → unit vector
        unit_vec = camera.pixel_to_unit_vector(star.u, star.v, undistort=True)

        sv = StarVector(
            u=star.u,
            v=star.v,
            bx=float(unit_vec[0]),
            by=float(unit_vec[1]),
            bz=float(unit_vec[2]),
            magnitude=star.instrumental_mag,
            flux=star.flux,
            snr=star.snr,
            quality=star.quality,
        )
        star_vectors.append(sv)

    logger.info(f"Converted {len(star_vectors)} stars to body-frame vectors")

    return star_vectors

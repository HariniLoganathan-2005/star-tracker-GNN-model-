"""
Star Tracker Camera Model

Defines the optical and sensor characteristics of a realistic star tracker camera.
Handles projection (3D → 2D pixel), back-projection (2D pixel → 3D ray),
and lens distortion correction.

Based on a pinhole camera model with radial distortion, similar to
commercial star trackers like the Sinclair Interplanetary ST-16.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class StarTrackerCamera:
    """
    Star tracker camera model with intrinsic parameters and distortion.

    The camera body frame convention:
        X-axis → right in the image (increasing column)
        Y-axis → down in the image (increasing row)
        Z-axis → boresight (pointing at the star field)

    Parameters
    ----------
    focal_length : float
        Focal length in pixels
    cx, cy : float
        Principal point (optical center) in pixels
    width, height : int
        Sensor dimensions in pixels
    k1, k2 : float
        Radial distortion coefficients
    """

    def __init__(self,
                 focal_length=None,
                 cx=None, cy=None,
                 width=None, height=None,
                 k1=None, k2=None):

        self.focal_length = focal_length or config.FOCAL_LENGTH_PX
        self.cx = cx if cx is not None else config.PRINCIPAL_POINT[0]
        self.cy = cy if cy is not None else config.PRINCIPAL_POINT[1]
        self.width = width or config.SENSOR_WIDTH_PX
        self.height = height or config.SENSOR_HEIGHT_PX
        self.k1 = k1 if k1 is not None else config.DISTORTION_K1
        self.k2 = k2 if k2 is not None else config.DISTORTION_K2

        # Derived parameters
        self.fov_deg = 2.0 * np.degrees(np.arctan(self.width / (2.0 * self.focal_length)))

    def pixel_to_unit_vector(self, u, v, undistort=True):
        """
        Convert pixel coordinates to a unit vector in the camera body frame.

        Parameters
        ----------
        u, v : float or array
            Pixel coordinates (u=column, v=row)
        undistort : bool
            If True, apply distortion correction before projection

        Returns
        -------
        numpy.ndarray
            Unit vector(s) in body frame, shape (3,) or (N, 3)
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        # Center the pixel coordinates
        x = u - self.cx
        y = v - self.cy

        if undistort:
            x, y = self._undistort(x, y)

        z = np.full_like(x, self.focal_length)

        if x.ndim == 0:
            vec = np.array([float(x), float(y), float(z)])
            return vec / np.linalg.norm(vec)
        else:
            vectors = np.column_stack([x, y, z])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms

    def unit_vector_to_pixel(self, body_vec, apply_distortion=True):
        """
        Project a unit vector in the camera body frame onto the sensor.

        Parameters
        ----------
        body_vec : numpy.ndarray
            Unit vector(s) in body frame, shape (3,) or (N, 3)
        apply_distortion : bool
            If True, apply lens distortion model

        Returns
        -------
        tuple
            (u, v) pixel coordinates — scalars or arrays
        """
        body_vec = np.asarray(body_vec, dtype=np.float64)

        if body_vec.ndim == 1:
            x, y, z = body_vec
        else:
            x, y, z = body_vec[:, 0], body_vec[:, 1], body_vec[:, 2]

        # Perspective projection
        # Only project if star is in front of camera (z > 0)
        valid = z > 0

        x_proj = np.where(valid, x / z * self.focal_length, np.nan)
        y_proj = np.where(valid, y / z * self.focal_length, np.nan)

        if apply_distortion:
            x_proj, y_proj = self._apply_distortion(x_proj, y_proj)

        u = x_proj + self.cx
        v = y_proj + self.cy

        return u, v

    def project_star_to_pixel(self, star_inertial_vec, attitude_quaternion):
        """
        Project a star (given by its inertial unit vector) onto the sensor,
        given the spacecraft attitude.

        Parameters
        ----------
        star_inertial_vec : numpy.ndarray
            Star unit vector in J2000 inertial frame, shape (3,) or (N, 3)
        attitude_quaternion : numpy.ndarray
            Quaternion [q0, q1, q2, q3] from inertial to body frame

        Returns
        -------
        tuple
            (u, v, visible) where visible is a boolean mask
        """
        from catalogue.catalogue_utils import quaternion_to_rotation_matrix

        R = quaternion_to_rotation_matrix(attitude_quaternion)

        star_inertial_vec = np.asarray(star_inertial_vec, dtype=np.float64)
        if star_inertial_vec.ndim == 1:
            star_body = R @ star_inertial_vec
        else:
            star_body = (R @ star_inertial_vec.T).T

        u, v = self.unit_vector_to_pixel(star_body)

        # Check if star is within sensor bounds
        if np.ndim(u) == 0:
            visible = (not np.isnan(u) and
                       0 <= u < self.width and
                       0 <= v < self.height)
        else:
            visible = (~np.isnan(u) &
                       (u >= 0) & (u < self.width) &
                       (v >= 0) & (v < self.height))

        return u, v, visible

    def get_visible_catalogue_stars(self, catalogue, attitude_quaternion):
        """
        Find all catalogue stars visible in the current FOV.

        Parameters
        ----------
        catalogue : dict
            Loaded catalogue from hipparcos.load_catalogue()
        attitude_quaternion : numpy.ndarray
            Current attitude quaternion

        Returns
        -------
        dict with keys:
            'u', 'v': pixel coordinates of visible stars
            'hip_ids': HIP IDs
            'vmag': visual magnitudes
            'body_vectors': unit vectors in body frame
            'inertial_vectors': unit vectors in inertial frame
        """
        from catalogue.catalogue_utils import quaternion_to_rotation_matrix

        R = quaternion_to_rotation_matrix(attitude_quaternion)

        # Transform all catalogue stars to body frame
        star_body = (R @ catalogue['unit_vectors'].T).T

        # Quick FOV check: star is in front of camera (z > 0)
        # and within angular radius of boresight
        z_check = star_body[:, 2] > 0

        # Project to pixels
        u, v = self.unit_vector_to_pixel(star_body)

        # Full visibility check
        visible = (z_check &
                   ~np.isnan(u) &
                   (u >= 0) & (u < self.width) &
                   (v >= 0) & (v < self.height))

        idx = np.where(visible)[0]

        return {
            'u': u[idx],
            'v': v[idx],
            'hip_ids': catalogue['hip_ids'][idx],
            'vmag': catalogue['vmag'][idx],
            'body_vectors': star_body[idx],
            'inertial_vectors': catalogue['unit_vectors'][idx],
        }

    def _apply_distortion(self, x, y):
        """
        Apply radial lens distortion to centered pixel coordinates.

        Uses the Brown-Conrady model:
            r² = x² + y²
            x_dist = x * (1 + k1*r² + k2*r⁴)
            y_dist = y * (1 + k1*r² + k2*r⁴)
        """
        r2 = x**2 + y**2
        radial = 1.0 + self.k1 * r2 + self.k2 * r2**2
        return x * radial, y * radial

    def _undistort(self, x, y, iterations=10):
        """
        Remove radial lens distortion using iterative method.

        Takes distorted centered coordinates and returns undistorted ones.
        """
        # Iterative approach: start with distorted coords as estimate,
        # refine until convergence
        x_u, y_u = x.copy(), y.copy()

        for _ in range(iterations):
            r2 = x_u**2 + y_u**2
            radial = 1.0 + self.k1 * r2 + self.k2 * r2**2
            x_u = x / radial
            y_u = y / radial

        return x_u, y_u

    def __repr__(self):
        return (f"StarTrackerCamera(f={self.focal_length:.1f}px, "
                f"FOV={self.fov_deg:.1f}°, "
                f"sensor={self.width}×{self.height}px, "
                f"center=({self.cx:.1f}, {self.cy:.1f}))")

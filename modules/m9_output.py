"""
Module 9 — Output Formatting

Formats the attitude determination result into a comprehensive output
including quaternion, Euler angles, pointing direction, and covariance.
"""

import logging
import numpy as np
from datetime import datetime, timezone

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catalogue.catalogue_utils import (
    quaternion_to_euler, quaternion_to_pointing, quaternion_angle
)

logger = logging.getLogger(__name__)


class AttitudeResult:
    """
    Complete attitude determination result.

    Contains the quaternion, Euler angles, pointing direction,
    error metrics, and metadata about the determination process.
    """

    def __init__(self, quest_result, method='triangle', ground_truth_q=None):
        """
        Parameters
        ----------
        quest_result : dict
            Output from Module 7 (QUEST algorithm)
        method : str
            Identification method used: 'triangle' or 'gnn'
        ground_truth_q : numpy.ndarray or None
            Ground truth quaternion for error computation
        """
        self.quaternion = quest_result['quaternion']
        self.rotation_matrix = quest_result['rotation_matrix']
        self.residual = quest_result['residual']
        self.residual_arcsec = quest_result['residual_arcsec']
        self.eigenvalue = quest_result['eigenvalue']
        self.n_stars_used = quest_result['n_observations']
        self.method = method
        self.timestamp = datetime.now(timezone.utc).isoformat()

        # Derived quantities
        self.roll, self.pitch, self.yaw = quaternion_to_euler(self.quaternion)
        self.ra_boresight, self.dec_boresight = quaternion_to_pointing(self.quaternion)

        # Covariance estimation (simplified — from residual and number of stars)
        self._estimate_covariance()

        # Angular velocity (not available from single image — set to zero)
        self.angular_velocity = np.zeros(3)

        # Ground truth comparison
        self.angular_error_deg = None
        self.angular_error_arcsec = None
        if ground_truth_q is not None:
            self.angular_error_deg = quaternion_angle(self.quaternion, ground_truth_q)
            self.angular_error_arcsec = self.angular_error_deg * 3600

    def _estimate_covariance(self):
        """
        Estimate attitude covariance from QUEST residual.

        A simplified covariance model based on the QUEST residual and the
        number of matched stars. The actual covariance depends on the
        star geometry and individual centroiding errors, but this gives
        a reasonable order-of-magnitude estimate.
        """
        if self.n_stars_used > 0 and self.residual > 0:
            # Per-axis variance (radians²) ≈ residual / (2 * n_stars)
            var_per_axis = self.residual / (2.0 * self.n_stars_used)
            self.covariance = np.diag([var_per_axis] * 3)
            self.attitude_uncertainty_arcsec = np.degrees(np.sqrt(var_per_axis)) * 3600
        else:
            self.covariance = np.diag([1e-6] * 3)
            self.attitude_uncertainty_arcsec = np.degrees(np.sqrt(1e-6)) * 3600

    def to_dict(self):
        """Convert result to dictionary for serialization."""
        return {
            'quaternion': self.quaternion.tolist(),
            'euler_deg': {
                'roll': float(self.roll),
                'pitch': float(self.pitch),
                'yaw': float(self.yaw),
            },
            'boresight': {
                'ra_deg': float(self.ra_boresight),
                'dec_deg': float(self.dec_boresight),
            },
            'angular_velocity': self.angular_velocity.tolist(),
            'covariance': self.covariance.tolist(),
            'attitude_uncertainty_arcsec': float(self.attitude_uncertainty_arcsec),
            'residual_arcsec': float(self.residual_arcsec),
            'n_stars_used': self.n_stars_used,
            'method': self.method,
            'timestamp': self.timestamp,
            'angular_error_arcsec': float(self.angular_error_arcsec) if self.angular_error_arcsec is not None else None,
        }

    def __str__(self):
        lines = [
            "=" * 65,
            "         STAR TRACKER ATTITUDE DETERMINATION RESULT",
            "=" * 65,
            "",
            "  Quaternion (scalar-first, inertial → body):",
            f"    q = [{self.quaternion[0]:+.8f}, {self.quaternion[1]:+.8f}, "
            f"{self.quaternion[2]:+.8f}, {self.quaternion[3]:+.8f}]",
            "",
            "  Euler Angles (ZYX convention):",
            f"    Roll  = {self.roll:+.4f}°",
            f"    Pitch = {self.pitch:+.4f}°",
            f"    Yaw   = {self.yaw:+.4f}°",
            "",
            "  Boresight Pointing:",
            f"    RA  = {self.ra_boresight:.4f}°  ({self.ra_boresight/15:.4f} h)",
            f"    Dec = {self.dec_boresight:+.4f}°",
            "",
            "  Angular Velocity:",
            f"    ω = [{self.angular_velocity[0]:.6f}, "
            f"{self.angular_velocity[1]:.6f}, "
            f"{self.angular_velocity[2]:.6f}] rad/s",
            "",
            "  Error Metrics:",
            f"    QUEST residual     = {self.residual_arcsec:.2f} arcsec RMS",
            f"    Attitude uncertainty = {self.attitude_uncertainty_arcsec:.2f} arcsec (1σ)",
        ]

        if self.angular_error_arcsec is not None:
            lines.append(f"    Ground truth error = {self.angular_error_arcsec:.2f} arcsec "
                         f"({self.angular_error_deg:.6f}°)")

        lines.extend([
            "",
            "  Identification:",
            f"    Method     = {self.method}",
            f"    Stars used = {self.n_stars_used}",
            f"    Timestamp  = {self.timestamp}",
            "",
            "  Covariance Matrix (3×3, radians²):",
        ])

        for i in range(3):
            line = "    ["
            for j in range(3):
                line += f" {self.covariance[i, j]:+.4e}"
            line += " ]"
            lines.append(line)

        lines.append("")
        lines.append("=" * 65)

        return "\n".join(lines)


def format_output(quest_result, method='triangle', ground_truth_q=None):
    """
    Create a formatted attitude result from QUEST output.

    Parameters
    ----------
    quest_result : dict
        Output from Module 7
    method : str
        Identification method used
    ground_truth_q : numpy.ndarray or None
        Ground truth quaternion for validation

    Returns
    -------
    AttitudeResult
        Complete attitude determination result
    """
    result = AttitudeResult(quest_result, method, ground_truth_q)

    logger.info(f"Attitude determined: RA={result.ra_boresight:.4f}°, "
                f"Dec={result.dec_boresight:+.4f}°, "
                f"residual={result.residual_arcsec:.2f} arcsec")

    if result.angular_error_arcsec is not None:
        logger.info(f"Ground truth error: {result.angular_error_arcsec:.2f} arcsec")

    return result

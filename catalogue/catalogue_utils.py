"""
Catalogue Utilities — Coordinate Conversions and Quaternion Math

Provides core mathematical functions for converting between coordinate systems:
- Right Ascension / Declination (celestial) ↔ Unit Vectors (Cartesian)
- Quaternion ↔ Rotation Matrix
- Angular distance computation
"""

import numpy as np


def radec_to_unit_vector(ra_deg, dec_deg):
    """
    Convert Right Ascension and Declination (J2000) to a 3D unit vector.

    The J2000 inertial frame:
        X → points to vernal equinox (RA=0, Dec=0)
        Y → RA=90°, Dec=0°
        Z → celestial north pole (Dec=90°)

    Parameters
    ----------
    ra_deg : float or array
        Right ascension in degrees [0, 360)
    dec_deg : float or array
        Declination in degrees [-90, 90]

    Returns
    -------
    numpy.ndarray
        Unit vector(s) of shape (3,) or (N, 3)
    """
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    if np.ndim(ra_deg) == 0:
        return np.array([x, y, z])
    else:
        return np.column_stack([x, y, z])


def unit_vector_to_radec(vec):
    """
    Convert a 3D unit vector back to Right Ascension and Declination.

    Parameters
    ----------
    vec : numpy.ndarray
        Unit vector of shape (3,) or (N, 3)

    Returns
    -------
    tuple
        (ra_deg, dec_deg) — scalars or arrays
    """
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim == 1:
        x, y, z = vec
    else:
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]

    dec_rad = np.arcsin(np.clip(z, -1.0, 1.0))
    ra_rad = np.arctan2(y, x)

    ra_deg = np.degrees(ra_rad) % 360.0
    dec_deg = np.degrees(dec_rad)

    return ra_deg, dec_deg


def angular_distance(v1, v2):
    """
    Compute the angular distance between two unit vectors.

    Uses the numerically stable formula:
        angle = atan2(|v1 × v2|, v1 · v2)

    Parameters
    ----------
    v1 : numpy.ndarray
        First unit vector(s), shape (3,) or (N, 3)
    v2 : numpy.ndarray
        Second unit vector(s), shape (3,) or (N, 3)

    Returns
    -------
    float or numpy.ndarray
        Angular distance in degrees
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    cross = np.cross(v1, v2)
    dot = np.sum(v1 * v2, axis=-1)

    if cross.ndim == 1:
        cross_norm = np.linalg.norm(cross)
    else:
        cross_norm = np.linalg.norm(cross, axis=-1)

    angle_rad = np.arctan2(cross_norm, dot)
    return np.degrees(angle_rad)


def angular_distance_radians(v1, v2):
    """
    Same as angular_distance but returns radians.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    cross = np.cross(v1, v2)
    dot = np.sum(v1 * v2, axis=-1)

    if cross.ndim == 1:
        cross_norm = np.linalg.norm(cross)
    else:
        cross_norm = np.linalg.norm(cross, axis=-1)

    return np.arctan2(cross_norm, dot)


def quaternion_to_rotation_matrix(q):
    """
    Convert a unit quaternion to a 3×3 rotation matrix.

    The quaternion convention is scalar-first: q = (q0, q1, q2, q3)
    where q0 is the scalar part and (q1, q2, q3) is the vector part.

    The rotation matrix R transforms vectors from the inertial frame
    to the body frame: v_body = R @ v_inertial

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion [q0, q1, q2, q3], norm should be 1

    Returns
    -------
    numpy.ndarray
        3×3 rotation matrix
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)  # Normalize to ensure unit quaternion

    q0, q1, q2, q3 = q

    R = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])

    return R


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3×3 rotation matrix to a unit quaternion (scalar-first).

    Uses Shepperd's method for numerical stability — selects the
    computation path based on which diagonal element is largest.

    Parameters
    ----------
    R : numpy.ndarray
        3×3 rotation matrix

    Returns
    -------
    numpy.ndarray
        Quaternion [q0, q1, q2, q3]
    """
    R = np.asarray(R, dtype=np.float64)

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q0 = 0.25 * s
        q1 = (R[2, 1] - R[1, 2]) / s
        q2 = (R[0, 2] - R[2, 0]) / s
        q3 = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q0 = (R[2, 1] - R[1, 2]) / s
        q1 = 0.25 * s
        q2 = (R[0, 1] + R[1, 0]) / s
        q3 = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q0 = (R[0, 2] - R[2, 0]) / s
        q1 = (R[0, 1] + R[1, 0]) / s
        q2 = 0.25 * s
        q3 = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q0 = (R[1, 0] - R[0, 1]) / s
        q1 = (R[0, 2] + R[2, 0]) / s
        q2 = (R[1, 2] + R[2, 1]) / s
        q3 = 0.25 * s

    q = np.array([q0, q1, q2, q3])

    # Ensure scalar part (q0) is positive (canonical form)
    if q[0] < 0:
        q = -q

    return q / np.linalg.norm(q)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q_result = q1 ⊗ q2

    Convention: scalar-first [q0, q1, q2, q3]

    Parameters
    ----------
    q1, q2 : numpy.ndarray
        Quaternions of shape (4,)

    Returns
    -------
    numpy.ndarray
        Product quaternion [q0, q1, q2, q3]
    """
    a0, a1, a2, a3 = q1
    b0, b1, b2, b3 = q2

    return np.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0
    ])


def quaternion_conjugate(q):
    """
    Compute the conjugate (inverse for unit quaternions) of a quaternion.

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion [q0, q1, q2, q3]

    Returns
    -------
    numpy.ndarray
        Conjugate quaternion [q0, -q1, -q2, -q3]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_angle(q1, q2):
    """
    Compute the angular distance between two quaternion orientations.

    Parameters
    ----------
    q1, q2 : numpy.ndarray
        Quaternions [q0, q1, q2, q3]

    Returns
    -------
    float
        Angular distance in degrees
    """
    # q_error = q1^(-1) ⊗ q2
    q_error = quaternion_multiply(quaternion_conjugate(q1), q2)

    # Angle = 2 * arccos(|q0|) where q0 is the scalar part of q_error
    angle_rad = 2.0 * np.arccos(np.clip(abs(q_error[0]), 0.0, 1.0))
    return np.degrees(angle_rad)


def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in degrees.

    Uses the ZYX convention (aerospace standard):
        - Yaw (ψ): rotation about Z
        - Pitch (θ): rotation about Y
        - Roll (φ): rotation about X

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion [q0, q1, q2, q3]

    Returns
    -------
    tuple
        (roll, pitch, yaw) in degrees
    """
    q0, q1, q2, q3 = q

    # Roll (rotation about X-axis)
    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1**2 + q2**2)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    # Pitch (rotation about Y-axis)
    sinp = 2.0 * (q0 * q2 - q3 * q1)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    # Yaw (rotation about Z-axis)
    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2**2 + q3**2)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """
    Convert Euler angles (ZYX convention) to quaternion.

    Parameters
    ----------
    roll_deg, pitch_deg, yaw_deg : float
        Euler angles in degrees

    Returns
    -------
    numpy.ndarray
        Quaternion [q0, q1, q2, q3]
    """
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy

    q = np.array([q0, q1, q2, q3])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def pointing_to_quaternion(ra_deg, dec_deg, roll_deg=0.0):
    """
    Create a quaternion representing a camera pointing at a given RA/Dec
    with a given roll angle.

    The camera boresight (body Z-axis) points at (RA, Dec).
    Roll is rotation about the boresight.

    Parameters
    ----------
    ra_deg : float
        Right ascension of boresight in degrees
    dec_deg : float
        Declination of boresight in degrees
    roll_deg : float
        Roll angle about boresight in degrees

    Returns
    -------
    numpy.ndarray
        Quaternion [q0, q1, q2, q3] mapping inertial → body frame
    """
    # Step 1: Rotate about Z-axis by RA
    # Step 2: Rotate about Y-axis by -(90° - Dec) = -(90° - Dec)
    # Step 3: Rotate about Z-axis by roll
    #
    # The camera body frame:
    #   body-Z = boresight direction (pointing at the star field)
    #   body-X = "right" in the image
    #   body-Y = "up" in the image

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    roll_rad = np.radians(roll_deg)

    # Rotation matrix: first align boresight to (RA, Dec), then apply roll
    # R_z(RA) @ R_y(-(90-Dec)) @ R_x(roll)

    # Boresight direction in inertial frame
    boresight = radec_to_unit_vector(ra_deg, dec_deg)

    # Build rotation matrix using Rodrigues-like construction
    # The body Z-axis should map to boresight direction
    # We construct the rotation by defining all three body axes

    # Body Z = boresight direction
    z_body = boresight / np.linalg.norm(boresight)

    # Choose an "up" reference — celestial north pole
    north_pole = np.array([0.0, 0.0, 1.0])

    # If boresight is near a pole, use a different reference
    if abs(np.dot(z_body, north_pole)) > 0.999:
        north_pole = np.array([1.0, 0.0, 0.0])

    # Body X = perpendicular to boresight in the plane containing north
    x_body = np.cross(north_pole, z_body)
    x_body = x_body / np.linalg.norm(x_body)

    # Body Y completes the right-handed frame
    y_body = np.cross(z_body, x_body)

    # Base rotation matrix (before roll)
    R_base = np.column_stack([x_body, y_body, z_body]).T  # body-from-inertial

    # Apply roll (rotation about boresight = body Z-axis)
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad),  np.cos(roll_rad), 0],
        [0, 0, 1]
    ])

    R_total = R_roll @ R_base

    return rotation_matrix_to_quaternion(R_total)


def quaternion_to_pointing(q):
    """
    Extract the boresight pointing direction (RA, Dec) from a quaternion.

    The boresight is the body Z-axis transformed to the inertial frame.

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion [q0, q1, q2, q3]

    Returns
    -------
    tuple
        (ra_deg, dec_deg)
    """
    R = quaternion_to_rotation_matrix(q)

    # Boresight in body frame is [0, 0, 1]
    # In inertial frame: v_inertial = R^T @ v_body
    boresight_inertial = R.T @ np.array([0.0, 0.0, 1.0])

    return unit_vector_to_radec(boresight_inertial)

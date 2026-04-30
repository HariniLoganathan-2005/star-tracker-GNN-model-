"""
Module 7 — QUEST Algorithm (QUaternion ESTimator)

Computes the optimal attitude quaternion from matched star observations
using RANSAC for robustness to mismatched stars.

Given N matched pairs:
    b_i — unit vector in body frame (from camera)
    r_i — unit vector in inertial frame (from catalogue)

Finds the rotation quaternion q that best satisfies: b_i = R(q) * r_i

Algorithm:
1. RANSAC: Sample random minimal subsets of 3 matched stars
2. Solve QUEST on each subset (eigenvector of the K matrix)
3. Count inliers (stars with residual < threshold)
4. Keep the solution with the most inliers
5. Re-run QUEST on all inliers for the final estimate

Reference:
Shuster, M.D. and Oh, S.D., "Three-Axis Attitude Determination from
Vector Observations," J. Guidance and Control, 1981.
"""

import logging
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def _solve_quest_core(body_vectors, reference_vectors, weights=None):
    """
    Core QUEST solver — eigenvector of the K matrix.

    Parameters
    ----------
    body_vectors : numpy.ndarray, shape (N, 3)
    reference_vectors : numpy.ndarray, shape (N, 3)
    weights : numpy.ndarray or None

    Returns
    -------
    numpy.ndarray: quaternion [q0, q1, q2, q3]
    float: largest eigenvalue
    """
    n_obs = len(body_vectors)
    if weights is None:
        weights = np.ones(n_obs) / n_obs
    else:
        weights = weights / np.sum(weights)

    # B = sum_i w_i * r_i * b_i^T  (note: reference × body^T)
    # This gives the rotation R such that b = R @ r (body-from-inertial)
    B = np.zeros((3, 3))
    for i in range(n_obs):
        B += weights[i] * np.outer(reference_vectors[i], body_vectors[i])

    # K matrix
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])

    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = Z
    K[1:, 0] = Z
    K[1:, 1:] = S - sigma * np.eye(3)

    # Eigenvector for largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    max_idx = np.argmax(eigenvalues)
    q = eigenvectors[:, max_idx].copy()
    q = q / np.linalg.norm(q)

    if q[0] < 0:
        q = -q

    return q, eigenvalues[max_idx]


def _compute_residuals(q, body_vectors, reference_vectors):
    """
    Compute per-star angular residuals in arcseconds.

    Parameters
    ----------
    q : quaternion
    body_vectors : (N, 3) body-frame vectors
    reference_vectors : (N, 3) inertial vectors

    Returns
    -------
    numpy.ndarray: per-star residuals in arcseconds
    """
    from catalogue.catalogue_utils import quaternion_to_rotation_matrix
    R = quaternion_to_rotation_matrix(q)

    residuals = []
    for i in range(len(body_vectors)):
        predicted = R @ reference_vectors[i]
        # Angular error between predicted and observed body vector
        dot = np.clip(np.dot(predicted, body_vectors[i]), -1.0, 1.0)
        err_rad = np.arccos(dot)
        residuals.append(np.degrees(err_rad) * 3600)

    return np.array(residuals)


def quest(body_vectors, reference_vectors, weights=None):
    """
    QUEST with RANSAC for robustness to outlier matches.

    Parameters
    ----------
    body_vectors : numpy.ndarray, shape (N, 3)
    reference_vectors : numpy.ndarray, shape (N, 3)
    weights : numpy.ndarray or None

    Returns
    -------
    dict with quaternion, rotation_matrix, residual, etc.
    """
    from catalogue.catalogue_utils import quaternion_to_rotation_matrix

    body_vectors = np.asarray(body_vectors, dtype=np.float64)
    reference_vectors = np.asarray(reference_vectors, dtype=np.float64)
    n_obs = len(body_vectors)

    if n_obs < 2:
        raise ValueError(f"QUEST requires >= 2 matched pairs, got {n_obs}")

    if weights is None:
        weights = np.ones(n_obs)
    weights = np.asarray(weights, dtype=np.float64)

    logger.info(f"Running QUEST with {n_obs} matched star pairs...")

    # =========================================================================
    # RANSAC: Sample subsets, find best consensus
    # =========================================================================
    inlier_threshold_arcsec = 60.0  # Stars within 1 arcmin are inliers
    min_sample = min(3, n_obs)
    max_trials = min(100, int(n_obs * (n_obs - 1) * (n_obs - 2) / 6))
    max_trials = max(max_trials, 10)

    best_q = None
    best_inlier_count = 0
    best_inliers = None

    rng = np.random.default_rng(42)

    if n_obs <= 4:
        # Too few stars for RANSAC, just solve directly
        q, lam = _solve_quest_core(body_vectors, reference_vectors, weights)
        best_q = q
        best_inliers = np.arange(n_obs)
        best_inlier_count = n_obs
    else:
        for trial in range(max_trials):
            # Random minimal sample
            sample = rng.choice(n_obs, size=min_sample, replace=False)

            try:
                q, lam = _solve_quest_core(
                    body_vectors[sample],
                    reference_vectors[sample],
                    weights[sample]
                )
            except Exception:
                continue

            # Count inliers
            residuals = _compute_residuals(q, body_vectors, reference_vectors)
            inliers = np.where(residuals < inlier_threshold_arcsec)[0]

            if len(inliers) > best_inlier_count:
                best_inlier_count = len(inliers)
                best_q = q
                best_inliers = inliers

                # If all stars are inliers, we're done
                if best_inlier_count == n_obs:
                    break

        logger.info(f"RANSAC: {best_inlier_count}/{n_obs} inliers after "
                    f"{min(trial+1, max_trials)} trials")

    # =========================================================================
    # Refine: Re-run QUEST on all inliers
    # =========================================================================
    if best_inliers is not None and len(best_inliers) >= 2:
        q_refined, lam = _solve_quest_core(
            body_vectors[best_inliers],
            reference_vectors[best_inliers],
            weights[best_inliers]
        )
    else:
        q_refined = best_q
        best_inliers = np.arange(n_obs)

    R = quaternion_to_rotation_matrix(q_refined)

    # Compute final residual on inliers
    residuals = _compute_residuals(q_refined, body_vectors[best_inliers],
                                    reference_vectors[best_inliers])
    residual_rms = np.sqrt(np.mean(residuals**2)) if len(residuals) > 0 else 0
    total_residual = np.sum((residuals / 3600 / 57.2958)**2)  # radians^2

    logger.info(f"QUEST result: q = [{q_refined[0]:.6f}, {q_refined[1]:.6f}, "
                f"{q_refined[2]:.6f}, {q_refined[3]:.6f}]")
    logger.info(f"QUEST residual: {residual_rms:.2f} arcsec RMS "
                f"({len(best_inliers)} inliers)")

    # Log per-star residuals
    all_residuals = _compute_residuals(q_refined, body_vectors, reference_vectors)
    for i in range(n_obs):
        status = "INLIER" if i in best_inliers else "OUTLIER"
        logger.info(f"  Star {i}: {all_residuals[i]:.2f} arcsec [{status}]")

    return {
        'quaternion': q_refined,
        'rotation_matrix': R,
        'residual': total_residual,
        'residual_arcsec': residual_rms,
        'eigenvalue': lam,
        'n_observations': len(best_inliers),
        'n_total_matches': n_obs,
        'inlier_indices': best_inliers,
    }


def quest_from_matches(star_vectors, matches, catalogue):
    """
    Run QUEST using star identification results.

    Parameters
    ----------
    star_vectors : list of StarVector
    matches : list of StarMatch
    catalogue : dict

    Returns
    -------
    dict: QUEST result
    """
    n_matches = len(matches)
    if n_matches < 2:
        raise ValueError(f"Need at least 2 matches, got {n_matches}")

    body_vecs = np.zeros((n_matches, 3))
    ref_vecs = np.zeros((n_matches, 3))
    weights = np.zeros(n_matches)

    for i, match in enumerate(matches):
        star = star_vectors[match.star_index]
        body_vecs[i] = star.body_vector
        ref_vecs[i] = catalogue['unit_vectors'][match.catalogue_index]
        weights[i] = match.confidence * star.snr

    result = quest(body_vecs, ref_vecs, weights)

    # Log match details
    for i, match in enumerate(matches):
        status = "USED" if i in result['inlier_indices'] else "REJECTED"
        logger.info(f"  Match: Det#{match.star_index} -> HIP {match.hip_id} "
                    f"[{status}]")

    return result

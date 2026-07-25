"""
Module 5 — Star Pattern Matching (Primary Star Identifier)

Identifies detected stars by matching angular distance patterns against
the Hipparcos star catalogue. Uses a two-stage approach:

Stage 1: For each pair of detected stars, find catalogue pairs with
    matching angular distance. Each match generates votes for specific
    (detected_star → catalogue_star) mappings.

Stage 2: For each detected star, collect its vote histogram. The correct
    catalogue star should receive far more votes than random candidates
    because it appears consistently across MULTIPLE pair queries, while
    noise candidates only appear in one or two queries.

Key insight: A single angular distance matches ~1000 catalogue pairs,
but requiring the SAME catalogue star to appear as a match partner for
MULTIPLE different detected stars greatly narrows it down.
"""

import os
import pickle
import logging
import numpy as np
from itertools import combinations
from scipy.spatial import KDTree
from dataclasses import dataclass
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from catalogue.catalogue_utils import angular_distance

logger = logging.getLogger(__name__)


@dataclass
class StarMatch:
    """A matched pair: detected star -> catalogue star."""
    star_index: int
    hip_id: int
    catalogue_index: int
    confidence: float = 0.0
    vote_count: int = 0


class TriangleDatabase:
    """Precomputed star pair angular distance database with KD-tree index."""

    def __init__(self):
        self.pair_distances = None
        self.pair_ids = None
        self.kdtree = None
        self.catalogue_hip_ids = None
        self.catalogue_vectors = None
        self.catalogue_vmag = None
        # Index: for each catalogue star, which pairs include it
        self.star_to_pairs = None
        self.is_built = False

    def build(self, catalogue, max_angle_deg=None, min_angle_deg=None):
        """Build the pair distance database from the star catalogue."""
        if max_angle_deg is None:
            max_angle_deg = config.TRIANGLE_MAX_PAIR_ANGLE_DEG
        if min_angle_deg is None:
            min_angle_deg = config.TRIANGLE_MIN_PAIR_ANGLE_DEG

        vectors = catalogue['unit_vectors']
        hip_ids = catalogue['hip_ids']
        vmag = catalogue['vmag']
        n_stars = len(hip_ids)

        logger.info(f"Building triangle database from {n_stars} catalogue stars...")
        logger.info(f"Pair angle range: [{min_angle_deg:.2f}, {max_angle_deg:.2f}] deg")

        cos_max = np.cos(np.radians(max_angle_deg))
        cos_min = np.cos(np.radians(min_angle_deg))

        pairs = []
        distances = []

        block_size = 500
        n_blocks = (n_stars + block_size - 1) // block_size

        for bi in range(n_blocks):
            i_start = bi * block_size
            i_end = min(i_start + block_size, n_stars)
            for bj in range(bi, n_blocks):
                j_start = max(bj * block_size, i_start)
                j_end = min(j_start + block_size, n_stars)
                if bi == bj:
                    for i in range(i_start, i_end):
                        for j in range(max(i + 1, j_start), j_end):
                            dot = np.dot(vectors[i], vectors[j])
                            if cos_max <= dot <= cos_min:
                                ang = np.degrees(np.arccos(np.clip(dot, -1, 1)))
                                pairs.append((i, j))
                                distances.append(ang)
                else:
                    dots = vectors[i_start:i_end] @ vectors[j_start:j_end].T
                    valid = (dots >= cos_max) & (dots <= cos_min)
                    ii, jj = np.where(valid)
                    for k in range(len(ii)):
                        i_abs = i_start + ii[k]
                        j_abs = j_start + jj[k]
                        if i_abs < j_abs:
                            ang = np.degrees(np.arccos(
                                np.clip(dots[ii[k], jj[k]], -1, 1)))
                            pairs.append((i_abs, j_abs))
                            distances.append(ang)
            if (bi + 1) % 5 == 0:
                logger.info(f"  Block {bi+1}/{n_blocks}: {len(pairs)} pairs so far")

        self.pair_distances = np.array(distances, dtype=np.float64)
        self.pair_ids = np.array(pairs, dtype=np.int32)

        logger.info(f"Found {len(self.pair_distances)} star pairs")

        self.kdtree = KDTree(self.pair_distances.reshape(-1, 1))

        # Build reverse index: cat_star_idx -> list of pair indices
        self.star_to_pairs = defaultdict(list)
        for p_idx, (ci, cj) in enumerate(self.pair_ids):
            self.star_to_pairs[int(ci)].append(p_idx)
            self.star_to_pairs[int(cj)].append(p_idx)

        self.catalogue_hip_ids = hip_ids.copy()
        self.catalogue_vectors = vectors.copy()
        self.catalogue_vmag = vmag.copy()
        self.is_built = True
        logger.info("Triangle database built successfully")

    def save(self, filepath):
        """Save the database to disk."""
        logger.info(f"Saving triangle database to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pair_distances': self.pair_distances,
                'pair_ids': self.pair_ids,
                'catalogue_hip_ids': self.catalogue_hip_ids,
                'catalogue_vectors': self.catalogue_vectors,
                'catalogue_vmag': self.catalogue_vmag,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved ({os.path.getsize(filepath) / 1e6:.1f} MB)")

    def load(self, filepath):
        """Load the database from disk."""
        logger.info(f"Loading triangle database from {filepath}...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.pair_distances = data['pair_distances']
        self.pair_ids = data['pair_ids']
        self.catalogue_hip_ids = data['catalogue_hip_ids']
        self.catalogue_vectors = data['catalogue_vectors']
        self.catalogue_vmag = data['catalogue_vmag']
        self.kdtree = KDTree(self.pair_distances.reshape(-1, 1))

        # Rebuild reverse index
        self.star_to_pairs = defaultdict(list)
        for p_idx, (ci, cj) in enumerate(self.pair_ids):
            self.star_to_pairs[int(ci)].append(p_idx)
            self.star_to_pairs[int(cj)].append(p_idx)

        self.is_built = True
        logger.info(f"Loaded {len(self.pair_distances)} pairs")


def match_stars(star_vectors, tri_db, catalogue, tolerance_deg=None):
    """
    Match detected stars against the catalogue using pair-based voting.

    Parameters
    ----------
    tolerance_deg : float or None
        Angular matching tolerance in degrees.  If None, uses
        config.TRIANGLE_ANGLE_TOLERANCE_DEG (0.01° default).
        Pass a tighter value (e.g. 0.005°) for narrow-FOV images
        to reduce catalogue ambiguity.
    """
    if not tri_db.is_built:
        raise RuntimeError("Triangle database not built.")

    n_stars = len(star_vectors)
    if n_stars < config.MIN_STARS_FOR_MATCH:
        logger.warning(f"Only {n_stars} stars, need >= {config.MIN_STARS_FOR_MATCH}")
        return [], False

    n_use = min(n_stars, config.TRIANGLE_TOP_N_STARS)
    selected = star_vectors[:n_use]
    body_vecs = np.array([s.body_vector for s in selected])

    logger.info(f"Matching with {n_use} brightest stars (of {n_stars} detected)")

    # Use provided tolerance or fall back to config default
    tolerance = tolerance_deg if tolerance_deg is not None else config.TRIANGLE_ANGLE_TOLERANCE_DEG
    logger.info(f"Pair-matching tolerance: {tolerance:.4f}°")

    # ===============================================================
    # Stage 1: Pair-based voting
    # For each detected pair, query the KD-tree and cast votes
    # ===============================================================
    # votes[det_idx] -> {cat_idx: vote_count}
    # Each pair query where cat_star appears as a match partner adds 1 vote
    votes = defaultdict(lambda: defaultdict(int))
    # Track which detected pair generated each vote for verification
    pair_evidence = defaultdict(lambda: defaultdict(list))

    n_pairs = 0
    total_matches = 0

    for i in range(n_use):
        for j in range(i + 1, n_use):
            obs_angle = angular_distance(body_vecs[i], body_vecs[j])

            # Query KD-tree
            pair_indices = tri_db.kdtree.query_ball_point(
                [obs_angle], r=tolerance
            )

            n_pairs += 1
            total_matches += len(pair_indices)

            for p_idx in pair_indices:
                cat_a, cat_b = tri_db.pair_ids[p_idx]
                cat_a, cat_b = int(cat_a), int(cat_b)

                # Vote: det_i could be cat_a, det_j could be cat_b
                votes[i][cat_a] += 1
                votes[j][cat_b] += 1
                pair_evidence[i][cat_a].append((j, cat_b))
                pair_evidence[j][cat_b].append((i, cat_a))

                # Or: det_i could be cat_b, det_j could be cat_a
                votes[i][cat_b] += 1
                votes[j][cat_a] += 1
                pair_evidence[i][cat_b].append((j, cat_a))
                pair_evidence[j][cat_a].append((i, cat_b))

    logger.info(f"Searched {n_pairs} pairs, {total_matches} catalogue matches total")

    if not votes:
        logger.warning("No matching pairs found")
        return [], False

    # ===============================================================
    # Stage 2: Extract top candidates per detected star
    # ===============================================================
    # For each detected star, sort candidates by vote count
    # The correct catalogue star should have significantly more votes
    # than random candidates

    candidates = {}
    for det_idx in range(n_use):
        if det_idx not in votes:
            continue

        star_votes = votes[det_idx]
        if not star_votes:
            continue

        # Sort by votes
        vote_counts = sorted(star_votes.items(), key=lambda x: -x[1])
        
        # Ratio test
        best_cat = None
        best_votes = 0
        second_votes = 0
        
        if len(vote_counts) == 1:
            best_cat, best_votes = vote_counts[0]
            vote_ratio = float('inf')
        elif len(vote_counts) > 1:
            best_cat, best_votes = vote_counts[0]
            second_votes = vote_counts[1][1]
            if best_votes >= config.QUEST_MIN_MATCHES and (best_votes >= second_votes * 1.5 or best_votes - second_votes >= 5):
                vote_ratio = best_votes / max(second_votes, 1)
            else:
                continue
        else:
            continue

        candidates[det_idx] = {
            'cat_idx': best_cat,
            'votes': best_votes,
            'ratio': vote_ratio,
            'evidence': pair_evidence[det_idx][best_cat],
        }

    logger.info(f"Top candidates for {len(candidates)} detected stars")
    for det_idx, c in sorted(candidates.items()):
        hip = int(tri_db.catalogue_hip_ids[c['cat_idx']])
        logger.info(f"  Det#{det_idx} -> HIP {hip}: "
                     f"votes={c['votes']}, ratio={c['ratio']:.1f}")

    # ===============================================================
    # Stage 3: Verify geometric consistency
    # For each pair of candidate identifications, check that the
    # angular distance between their catalogue counterparts matches
    # the observed angular distance between the detected stars
    # ===============================================================
    matches = _verify_geometric_consistency(
        candidates, selected, body_vecs, tri_db
    )

    success = len(matches) >= config.QUEST_MIN_MATCHES
    logger.info(f"Verified matches: {len(matches)} "
                f"(need >= {config.QUEST_MIN_MATCHES})")

    if success:
        for m in matches:
            logger.info(f"  FINAL: Det#{m.star_index} -> HIP {m.hip_id} "
                        f"(confidence={m.confidence:.3f})")

    return matches, success


def _verify_geometric_consistency(candidates, selected_stars, body_vecs, tri_db):
    """
    Verify candidates by checking pairwise angular distance consistency.

    For each candidate identification (det_i -> cat_A, det_j -> cat_B),
    the angular distance from det_i to det_j should match the angular
    distance from cat_A to cat_B within tolerance.
    """
    tolerance = config.TRIANGLE_ANGLE_TOLERANCE_DEG * 5  # Relaxed for verification

    det_indices = sorted(candidates.keys())
    n = len(det_indices)
    if n < 2:
        return []

    cat_indices = [candidates[d]['cat_idx'] for d in det_indices]
    cat_vecs = tri_db.catalogue_vectors[cat_indices]
    det_vecs = body_vecs[det_indices]

    # Compute consistency score: how many pairs match?
    consistency = np.zeros(n)
    n_checks = np.zeros(n)

    for i in range(n):
        for j in range(i + 1, n):
            obs_ang = angular_distance(det_vecs[i], det_vecs[j])
            cat_ang = angular_distance(cat_vecs[i], cat_vecs[j])
            error_deg = abs(obs_ang - cat_ang)

            if error_deg <= tolerance:
                consistency[i] += 1
                consistency[j] += 1
            n_checks[i] += 1
            n_checks[j] += 1

    # Find the largest self-consistent subset
    # Start with stars that have high consistency ratio
    matches = []
    for k in range(n):
        det_idx = det_indices[k]
        cat_idx = candidates[det_idx]['cat_idx']

        if n_checks[k] > 0:
            cons_ratio = consistency[k] / n_checks[k]
        else:
            cons_ratio = 0

        if cons_ratio >= 0.4 and candidates[det_idx]['ratio'] > 1.1:
            hip_id = int(tri_db.catalogue_hip_ids[cat_idx])
            match = StarMatch(
                star_index=det_idx,
                hip_id=hip_id,
                catalogue_index=cat_idx,
                confidence=cons_ratio,
                vote_count=candidates[det_idx]['votes'],
            )
            matches.append(match)

    # If we have too few matches, try being more lenient
    if len(matches) < config.QUEST_MIN_MATCHES:
        matches = []
        for k in range(n):
            det_idx = det_indices[k]
            cat_idx = candidates[det_idx]['cat_idx']

            if n_checks[k] > 0:
                cons_ratio = consistency[k] / n_checks[k]
            else:
                cons_ratio = 0

            if cons_ratio >= 0.3:
                hip_id = int(tri_db.catalogue_hip_ids[cat_idx])
                match = StarMatch(
                    star_index=det_idx,
                    hip_id=hip_id,
                    catalogue_index=cat_idx,
                    confidence=cons_ratio,
                    vote_count=candidates[det_idx]['votes'],
                )
                matches.append(match)

    return matches


def build_triangle_database(catalogue, output_path=None):
    """Build and save the triangle database."""
    if output_path is None:
        output_path = config.TRIANGLE_DB_FILE
    tri_db = TriangleDatabase()
    tri_db.build(catalogue)
    tri_db.save(output_path)
    return tri_db


def load_triangle_database(filepath=None):
    """Load a precomputed triangle database from disk."""
    if filepath is None:
        filepath = config.TRIANGLE_DB_FILE
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Triangle database not found: {filepath}\n"
            f"Run setup_data.py to build it."
        )
    tri_db = TriangleDatabase()
    tri_db.load(filepath)
    return tri_db

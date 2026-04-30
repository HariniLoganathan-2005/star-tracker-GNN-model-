"""
Hipparcos Star Catalogue Handler

Downloads the Hipparcos catalogue from VizieR (CDS),
filters to stars brighter than magnitude 6.5,
and provides functions to load, query, and convert catalogue data.
"""

import os
import numpy as np
import pandas as pd
import logging

from catalogue.catalogue_utils import radec_to_unit_vector

logger = logging.getLogger(__name__)


def download_hipparcos(output_path, mag_limit=6.5):
    """
    Download the Hipparcos catalogue from VizieR and filter by magnitude.

    Retrieves the I/239/hip_main catalogue (Hipparcos Main Catalogue)
    which contains 118,218 stars with position, proper motion, and magnitude.

    Parameters
    ----------
    output_path : str
        Path to save the filtered catalogue CSV
    mag_limit : float
        Maximum visual magnitude to include (default: 6.5)

    Returns
    -------
    pandas.DataFrame
        Filtered catalogue with columns: HIP, RA_deg, Dec_deg, Vmag
    """
    from astroquery.vizier import Vizier

    logger.info(f"Downloading Hipparcos catalogue from VizieR (mag ≤ {mag_limit})...")

    # Configure Vizier query
    # I/239/hip_main is the Hipparcos main catalogue
    v = Vizier(
        columns=['HIP', 'RAhms', 'DEdms', 'Vmag', 'RAICRS', 'DEICRS'],
        column_filters={f"Vmag": f"<{mag_limit}"},
        row_limit=-1  # No row limit — get all matching stars
    )

    result = v.query_constraints(catalog="I/239/hip_main", Vmag=f"<{mag_limit}")

    if not result:
        raise RuntimeError("Failed to download Hipparcos catalogue from VizieR. "
                           "Check internet connection.")

    table = result[0]
    logger.info(f"Downloaded {len(table)} stars from Hipparcos (Vmag < {mag_limit})")

    # Convert to pandas DataFrame
    df = pd.DataFrame({
        'HIP': table['HIP'].data.data if hasattr(table['HIP'].data, 'data') else np.array(table['HIP']),
        'RA_deg': np.array(table['RAICRS'], dtype=np.float64),
        'Dec_deg': np.array(table['DEICRS'], dtype=np.float64),
        'Vmag': np.array(table['Vmag'], dtype=np.float64),
    })

    # Drop any rows with missing data
    df = df.dropna().reset_index(drop=True)

    # Convert HIP ID to integer
    df['HIP'] = df['HIP'].astype(int)

    # Sort by magnitude (brightest first)
    df = df.sort_values('Vmag').reset_index(drop=True)

    logger.info(f"After filtering: {len(df)} stars with valid data")

    # Verify with known bright stars
    _verify_catalogue(df)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved catalogue to {output_path}")

    return df


def _verify_catalogue(df):
    """
    Verify catalogue integrity by checking known bright stars.
    """
    known_stars = {
        # HIP ID: (name, expected_ra_approx, expected_dec_approx, expected_vmag_approx)
        32349: ("Sirius", 101.3, -16.7, -1.46),
        30438: ("Canopus", 95.99, -52.7, -0.74),
        69673: ("Arcturus", 213.9, 19.2, -0.05),
        91262: ("Vega", 279.2, 38.8, 0.03),
        24436: ("Rigel", 78.6, -8.2, 0.13),
    }

    for hip_id, (name, exp_ra, exp_dec, exp_vmag) in known_stars.items():
        match = df[df['HIP'] == hip_id]
        if len(match) == 0:
            logger.warning(f"Known star {name} (HIP {hip_id}) not found in catalogue!")
            continue

        star = match.iloc[0]
        ra_err = abs(star['RA_deg'] - exp_ra)
        dec_err = abs(star['Dec_deg'] - exp_dec)
        vmag_err = abs(star['Vmag'] - exp_vmag)

        if ra_err > 1.0 or dec_err > 1.0:
            logger.warning(f"{name}: position error > 1° — RA_err={ra_err:.2f}°, "
                           f"Dec_err={dec_err:.2f}°")
        else:
            logger.info(f"✓ {name} (HIP {hip_id}): RA={star['RA_deg']:.3f}°, "
                        f"Dec={star['Dec_deg']:.3f}°, Vmag={star['Vmag']:.2f}")


def load_catalogue(catalogue_path, compute_vectors=True):
    """
    Load the Hipparcos catalogue from a CSV file.

    Parameters
    ----------
    catalogue_path : str
        Path to the catalogue CSV file
    compute_vectors : bool
        If True, compute unit vectors for each star

    Returns
    -------
    dict with keys:
        'df': pandas DataFrame with star data
        'hip_ids': numpy array of HIP IDs
        'ra_deg': numpy array of RA in degrees
        'dec_deg': numpy array of Dec in degrees
        'vmag': numpy array of visual magnitudes
        'unit_vectors': numpy array of shape (N, 3) — unit vectors in J2000
        'id_to_index': dict mapping HIP ID to array index
    """
    if not os.path.exists(catalogue_path):
        raise FileNotFoundError(
            f"Catalogue file not found: {catalogue_path}\n"
            f"Run setup_data.py first to download the Hipparcos catalogue."
        )

    df = pd.read_csv(catalogue_path)
    logger.info(f"Loaded catalogue with {len(df)} stars from {catalogue_path}")

    result = {
        'df': df,
        'hip_ids': df['HIP'].values.astype(int),
        'ra_deg': df['RA_deg'].values.astype(np.float64),
        'dec_deg': df['Dec_deg'].values.astype(np.float64),
        'vmag': df['Vmag'].values.astype(np.float64),
        'id_to_index': {int(hip): idx for idx, hip in enumerate(df['HIP'].values)},
    }

    if compute_vectors:
        result['unit_vectors'] = radec_to_unit_vector(
            result['ra_deg'], result['dec_deg']
        )

    return result


def get_stars_in_fov(catalogue, boresight_vec, fov_radius_deg):
    """
    Get all catalogue stars within a circular field of view.

    Parameters
    ----------
    catalogue : dict
        Loaded catalogue from load_catalogue()
    boresight_vec : numpy.ndarray
        Unit vector of the boresight direction (camera center)
    fov_radius_deg : float
        Radius of the field of view in degrees

    Returns
    -------
    dict with keys:
        'indices': array indices of visible stars
        'hip_ids': HIP IDs of visible stars
        'unit_vectors': unit vectors of visible stars
        'vmag': magnitudes of visible stars
        'angular_distances': distances from boresight in degrees
    """
    # Compute angular distance from boresight to all catalogue stars
    dot_products = catalogue['unit_vectors'] @ boresight_vec
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_distances = np.degrees(np.arccos(dot_products))

    # Select stars within FOV
    mask = angular_distances <= fov_radius_deg
    indices = np.where(mask)[0]

    return {
        'indices': indices,
        'hip_ids': catalogue['hip_ids'][indices],
        'unit_vectors': catalogue['unit_vectors'][indices],
        'vmag': catalogue['vmag'][indices],
        'angular_distances': angular_distances[indices],
    }

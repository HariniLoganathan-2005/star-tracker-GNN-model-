"""
Synthetic Star Tracker Image Generator

Generates realistic synthetic FITS images of star fields by projecting
Hipparcos catalogue stars through a virtual star tracker camera model.

Each generated image includes:
- Realistic star PSFs (Gaussian point spread function)
- Poisson photon noise
- Gaussian read noise
- Dark current noise
- Controllable sky background
- Ground truth metadata in FITS header (true attitude, star list)

Used for: GNN training data, pipeline testing, occlusion experiments.
"""

import os
import logging
import numpy as np
from astropy.io import fits

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from catalogue.catalogue_utils import (
    pointing_to_quaternion, quaternion_to_pointing, radec_to_unit_vector
)
from camera.camera_model import StarTrackerCamera

logger = logging.getLogger(__name__)


def magnitude_to_flux(vmag, base_flux=None, exposure_time=None):
    """
    Convert visual magnitude to expected flux (ADU) for the star tracker.

    Uses the relation: flux = base_flux * 10^(-0.4 * vmag)

    Parameters
    ----------
    vmag : float or array
        Visual magnitude (smaller = brighter)
    base_flux : float
        Flux of a magnitude-0 star in ADU
    exposure_time : float
        Exposure time in seconds

    Returns
    -------
    float or array
        Expected flux in ADU
    """
    if base_flux is None:
        base_flux = config.BASE_STAR_FLUX
    if exposure_time is None:
        exposure_time = config.EXPOSURE_TIME

    return base_flux * exposure_time * 10.0 ** (-0.4 * vmag)


def render_star_psf(image, u, v, flux, psf_sigma=None):
    """
    Render a single star onto the image as a 2D Gaussian PSF.

    Parameters
    ----------
    image : numpy.ndarray
        2D image array to render onto (modified in-place)
    u, v : float
        Sub-pixel star position (column, row)
    flux : float
        Total integrated flux of the star (ADU)
    psf_sigma : float
        Gaussian PSF standard deviation in pixels

    Returns
    -------
    numpy.ndarray
        Modified image with star rendered
    """
    if psf_sigma is None:
        psf_sigma = config.STAR_PSF_SIGMA

    h, w = image.shape

    # Determine rendering region (±4σ is sufficient)
    half_size = int(np.ceil(4 * psf_sigma))
    r_min = max(0, int(v) - half_size)
    r_max = min(h, int(v) + half_size + 1)
    c_min = max(0, int(u) - half_size)
    c_max = min(w, int(u) + half_size + 1)

    if r_min >= r_max or c_min >= c_max:
        return image

    # Create coordinate grids
    rows = np.arange(r_min, r_max)
    cols = np.arange(c_min, c_max)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # 2D Gaussian centered on (u, v)
    gaussian = np.exp(-(
        (col_grid - u) ** 2 + (row_grid - v) ** 2
    ) / (2 * psf_sigma ** 2))

    # Normalize so total flux = specified flux
    gaussian_sum = gaussian.sum()
    if gaussian_sum > 0:
        gaussian = gaussian * (flux / gaussian_sum)

    image[r_min:r_max, c_min:c_max] += gaussian

    return image


def add_noise(image, read_noise_sigma=None, dark_current=None,
              exposure_time=None, gain=None):
    """
    Add realistic sensor noise to a clean image.

    Noise model (per pixel):
        signal_electrons = image * gain
        shot_noise = Poisson(signal_electrons) / gain
        read_noise = Normal(0, read_noise_sigma)
        dark_noise = Poisson(dark_current * exposure_time) / gain

    Parameters
    ----------
    image : numpy.ndarray
        Clean image (stars + background, no noise)
    read_noise_sigma : float
        Read noise standard deviation in ADU
    dark_current : float
        Dark current rate (electrons/pixel/second)
    exposure_time : float
        Exposure time in seconds
    gain : float
        Conversion gain (electrons per ADU)

    Returns
    -------
    numpy.ndarray
        Noisy image
    """
    if read_noise_sigma is None:
        read_noise_sigma = config.READ_NOISE_SIGMA
    if dark_current is None:
        dark_current = config.DARK_CURRENT_RATE
    if exposure_time is None:
        exposure_time = config.EXPOSURE_TIME
    if gain is None:
        gain = config.PHOTON_GAIN

    rng = np.random.default_rng()

    # Convert to electrons
    signal_electrons = np.clip(image * gain, 0, None)

    # Poisson (photon) noise
    noisy_electrons = rng.poisson(signal_electrons.astype(np.float64))

    # Dark current
    dark_electrons = rng.poisson(dark_current * exposure_time,
                                 size=image.shape)
    noisy_electrons = noisy_electrons + dark_electrons

    # Convert back to ADU
    noisy_adu = noisy_electrons.astype(np.float64) / gain

    # Add Gaussian read noise
    read_noise = rng.normal(0, read_noise_sigma, size=image.shape)
    noisy_adu += read_noise

    # Clip to sensor range
    noisy_adu = np.clip(noisy_adu, 0, config.SATURATION_LEVEL)

    return noisy_adu


def generate_synthetic_image(catalogue, ra_deg=None, dec_deg=None,
                             roll_deg=None, occlusion_fraction=0.0,
                             camera=None, add_noise_flag=True,
                             rng=None):
    """
    Generate a single synthetic star tracker image.

    Parameters
    ----------
    catalogue : dict
        Loaded Hipparcos catalogue
    ra_deg : float or None
        Boresight right ascension. Random if None.
    dec_deg : float or None
        Boresight declination. Random if None.
    roll_deg : float or None
        Boresight roll. Random if None.
    occlusion_fraction : float
        Fraction of stars to randomly remove (0 to 1)
    camera : StarTrackerCamera or None
        Camera model. Uses default if None.
    add_noise_flag : bool
        Whether to add sensor noise
    rng : numpy.random.Generator or None
        Random number generator for reproducibility

    Returns
    -------
    dict with keys:
        'image': numpy.ndarray — 2D image array
        'header': astropy.io.fits.Header — FITS header with ground truth
        'true_quaternion': numpy.ndarray — ground truth quaternion
        'true_ra': float — boresight RA
        'true_dec': float — boresight Dec
        'true_roll': float — boresight roll
        'visible_stars': dict — info about stars rendered
        'n_stars_rendered': int — number of stars in the image
    """
    if rng is None:
        rng = np.random.default_rng()

    if camera is None:
        camera = StarTrackerCamera()

    # Random pointing if not specified
    if ra_deg is None:
        ra_deg = rng.uniform(0, 360)
    if dec_deg is None:
        # Uniform on the sphere: Dec ~ arcsin(uniform(-1, 1))
        dec_deg = np.degrees(np.arcsin(rng.uniform(-1, 1)))
    if roll_deg is None:
        roll_deg = rng.uniform(0, 360)

    # Compute attitude quaternion
    q_true = pointing_to_quaternion(ra_deg, dec_deg, roll_deg)

    # Find visible catalogue stars
    visible = camera.get_visible_catalogue_stars(catalogue, q_true)

    n_visible = len(visible['hip_ids'])
    logger.debug(f"Pointing: RA={ra_deg:.2f}°, Dec={dec_deg:.2f}°, "
                 f"Roll={roll_deg:.2f}° → {n_visible} visible stars")

    # Apply occlusion (randomly remove stars)
    if occlusion_fraction > 0 and n_visible > 0:
        n_remove = int(n_visible * occlusion_fraction)
        keep_mask = np.ones(n_visible, dtype=bool)
        remove_indices = rng.choice(n_visible, size=n_remove, replace=False)
        keep_mask[remove_indices] = False

        for key in ['u', 'v', 'hip_ids', 'vmag', 'body_vectors', 'inertial_vectors']:
            visible[key] = visible[key][keep_mask]

        n_visible = len(visible['hip_ids'])
        logger.debug(f"After {occlusion_fraction*100:.0f}% occlusion: "
                     f"{n_visible} stars remain")

    # Create the image
    image = np.full((camera.height, camera.width),
                    config.SKY_BACKGROUND_LEVEL, dtype=np.float64)

    # Render each star
    star_info = []
    for i in range(n_visible):
        u = visible['u'][i]
        v = visible['v'][i]
        vmag = visible['vmag'][i]
        hip_id = visible['hip_ids'][i]

        flux = magnitude_to_flux(vmag)
        render_star_psf(image, u, v, flux)

        star_info.append({
            'hip_id': int(hip_id),
            'u': float(u),
            'v': float(v),
            'vmag': float(vmag),
            'flux': float(flux),
        })

    # Add noise
    if add_noise_flag:
        image = add_noise(image, rng=rng) if False else add_noise(image)

    # Build FITS header with ground truth
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = -64  # 64-bit floating point
    header['NAXIS'] = 2
    header['NAXIS1'] = camera.width
    header['NAXIS2'] = camera.height

    # Camera parameters
    header['FOCAL_L'] = (camera.focal_length, 'Focal length in pixels')
    header['CX'] = (camera.cx, 'Principal point X')
    header['CY'] = (camera.cy, 'Principal point Y')
    header['FOV'] = (camera.fov_deg, 'Field of view in degrees')

    # Ground truth attitude
    header['TRUE_Q0'] = (float(q_true[0]), 'True quaternion scalar part')
    header['TRUE_Q1'] = (float(q_true[1]), 'True quaternion X')
    header['TRUE_Q2'] = (float(q_true[2]), 'True quaternion Y')
    header['TRUE_Q3'] = (float(q_true[3]), 'True quaternion Z')
    header['TRUE_RA'] = (float(ra_deg), 'True boresight RA (degrees)')
    header['TRUE_DEC'] = (float(dec_deg), 'True boresight Dec (degrees)')
    header['TRUE_ROL'] = (float(roll_deg), 'True boresight roll (degrees)')

    # Star count
    header['NSTARS'] = (n_visible, 'Number of stars rendered')
    header['OCCL_FR'] = (float(occlusion_fraction), 'Occlusion fraction applied')

    # Star IDs (store as comma-separated string in header)
    if star_info:
        hip_ids_str = ','.join(str(s['hip_id']) for s in star_info[:100])
        header['STARIDS'] = (hip_ids_str[:68], 'HIP IDs of rendered stars (truncated)')

    # Simulation flag
    header['SYNTHIMG'] = (True, 'Synthetic image flag')
    header['COMMENT'] = 'Generated by Star Tracker Pipeline - Synthetic Image Module'

    return {
        'image': image,
        'header': header,
        'true_quaternion': q_true,
        'true_ra': ra_deg,
        'true_dec': dec_deg,
        'true_roll': roll_deg,
        'visible_stars': star_info,
        'n_stars_rendered': n_visible,
    }


def save_synthetic_image(result, filepath):
    """
    Save a synthetic image as a FITS file.

    Parameters
    ----------
    result : dict
        Output from generate_synthetic_image()
    filepath : str
        Path to save the FITS file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    hdu = fits.PrimaryHDU(data=result['image'], header=result['header'])
    hdul = fits.HDUList([hdu])
    hdul.writeto(filepath, overwrite=True)

    logger.debug(f"Saved synthetic image to {filepath}")


def generate_dataset(catalogue, output_dir, n_images, occlusion_fraction=0.0,
                     prefix="synth", seed=42):
    """
    Generate a set of synthetic star tracker images.

    Parameters
    ----------
    catalogue : dict
        Loaded Hipparcos catalogue
    output_dir : str
        Directory to save images
    n_images : int
        Number of images to generate
    occlusion_fraction : float
        Fraction of stars to occlude
    prefix : str
        Filename prefix
    seed : int
        Random seed for reproducibility

    Returns
    -------
    list of dict
        Metadata for each generated image
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    camera = StarTrackerCamera()

    logger.info(f"Generating {n_images} synthetic images in {output_dir}...")
    logger.info(f"Occlusion fraction: {occlusion_fraction:.0%}")

    metadata = []

    for i in range(n_images):
        result = generate_synthetic_image(
            catalogue,
            occlusion_fraction=occlusion_fraction,
            camera=camera,
            rng=rng,
        )

        filename = f"{prefix}_{i+1:04d}.fits"
        filepath = os.path.join(output_dir, filename)
        save_synthetic_image(result, filepath)

        meta = {
            'filename': filename,
            'filepath': filepath,
            'true_ra': result['true_ra'],
            'true_dec': result['true_dec'],
            'true_roll': result['true_roll'],
            'true_quaternion': result['true_quaternion'].tolist(),
            'n_stars': result['n_stars_rendered'],
            'occlusion': occlusion_fraction,
        }
        metadata.append(meta)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  Generated {i+1}/{n_images}: {filename} "
                        f"({result['n_stars_rendered']} stars)")

    logger.info(f"Dataset generation complete: {n_images} images saved")
    return metadata

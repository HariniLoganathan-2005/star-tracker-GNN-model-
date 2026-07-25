"""
Star Tracker Attitude Determination Pipeline — Global Configuration

All constants, camera parameters, thresholds, and file paths are defined here.
"""

import os
import numpy as np

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CATALOGUE_FILE = os.path.join(DATA_DIR, "hipparcos_catalogue.csv")
TRIANGLE_DB_FILE = os.path.join(DATA_DIR, "triangle_db.pkl")
SKYVIEW_DIR = os.path.join(DATA_DIR, "skyview_fits")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
SYNTHETIC_TRAIN_DIR = os.path.join(SYNTHETIC_DIR, "train")
SYNTHETIC_VAL_DIR = os.path.join(SYNTHETIC_DIR, "val")
SYNTHETIC_TEST_DIR = os.path.join(SYNTHETIC_DIR, "test")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create directories if they don't exist
for d in [DATA_DIR, SKYVIEW_DIR, SYNTHETIC_TRAIN_DIR, SYNTHETIC_VAL_DIR,
          SYNTHETIC_TEST_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# Star Tracker Camera Parameters
# =============================================================================
# Modeled after a realistic narrow-FOV star tracker (e.g., ST-16 class)

SENSOR_WIDTH_PX = 1024           # pixels
SENSOR_HEIGHT_PX = 1024          # pixels
FOV_DEG = 20.0                   # field of view in degrees (diagonal)

# Focal length derived from FOV: f = (sensor_width/2) / tan(FOV/2)
FOCAL_LENGTH_PX = (SENSOR_WIDTH_PX / 2.0) / np.tan(np.radians(FOV_DEG / 2.0))
# ≈ 2919.2 pixels for 20° FOV with 1024px sensor

PRINCIPAL_POINT = (SENSOR_WIDTH_PX / 2.0, SENSOR_HEIGHT_PX / 2.0)  # (cx, cy)

# Lens distortion coefficients (radial distortion)
# For a well-calibrated star tracker, these are small
DISTORTION_K1 = -1.0e-7   # First radial distortion coefficient
DISTORTION_K2 = 1.0e-13   # Second radial distortion coefficient

# =============================================================================
# Star Detection Parameters
# =============================================================================
DETECTION_SIGMA_THRESHOLD = 5.0    # Detect stars above N × σ_background
MIN_STAR_PIXELS = 5                # Minimum blob size (pixels) to be a star
MAX_STAR_PIXELS = 225              # Maximum blob size (15×15) — reject artifacts
MAX_ECCENTRICITY = 0.6             # Reject elongated blobs (satellite trails)
CENTROID_BOX_HALF_SIZE = 5         # Half-size of box for Gaussian centroiding
MIN_STARS_FOR_MATCH = 4            # Minimum detected stars for triangle matching

# =============================================================================
# Catalogue Parameters
# =============================================================================
HIPPARCOS_MAG_LIMIT = 6.5          # Only use stars brighter than this magnitude
# Magnitude 6.5 is approximately the naked-eye limit
# Results in ~9,000–10,000 usable stars

# =============================================================================
# Triangle Matching Parameters
# =============================================================================
TRIANGLE_ANGLE_TOLERANCE_DEG = 0.01   # Angular matching tolerance in degrees
TRIANGLE_MAX_PAIR_ANGLE_DEG = 20.0    # Max angular distance for catalogue pairs
# Should match FOV — pairs farther apart can't both be in the same image
TRIANGLE_MIN_PAIR_ANGLE_DEG = 0.1     # Min angular distance (avoid too-close pairs)
TRIANGLE_BRIGHTNESS_TOLERANCE = 1.5   # Magnitude ratio tolerance for verification
TRIANGLE_TOP_N_STARS = 15             # Use brightest N stars for matching
TRIANGLE_MIN_VOTES = 3                # Minimum vote count for a confident match
TRIANGLE_CONFIDENCE_THRESHOLD = 0.6   # Fraction of triplets that must agree

# =============================================================================
# QUEST Parameters
# =============================================================================
QUEST_MAX_ITERATIONS = 10     # Newton-Raphson max iterations
QUEST_CONVERGENCE_TOL = 1e-12 # Convergence tolerance for eigenvalue
QUEST_MIN_MATCHES = 3         # Minimum matched star pairs for QUEST

# =============================================================================
# Image Preprocessing Parameters
# =============================================================================
BACKGROUND_BOX_SIZE = 64      # Box size for median filter background estimation
BACKGROUND_FILTER_SIZE = 3    # Sigma for background smoothing

# =============================================================================
# Synthetic Image Generation Parameters
# =============================================================================
STAR_PSF_SIGMA = 1.2          # Gaussian PSF sigma in pixels (star point spread)
READ_NOISE_SIGMA = 1.0        # Gaussian read noise standard deviation (ADU)
DARK_CURRENT_RATE = 0.1       # Dark current electrons per pixel per second
EXPOSURE_TIME = 0.5           # Exposure time in seconds
PHOTON_GAIN = 2.0             # Electrons per ADU
BASE_STAR_FLUX = 5e5          # Flux of a magnitude-0 star (ADU in exposure)
SKY_BACKGROUND_LEVEL = 10.0   # Constant sky background level (ADU)
BIT_DEPTH = 16                # Sensor bit depth
SATURATION_LEVEL = 2**16 - 1  # Maximum pixel value (65535 for 16-bit)

# =============================================================================
# Validation Parameters
# =============================================================================
SKYVIEW_NUM_IMAGES = 50          # Number of SkyView images to download
SKYVIEW_SURVEY = "DSS2 Red"     # Survey to use (good for star detection)
SKYVIEW_IMAGE_PIXELS = 512      # SkyView image size in pixels
SKYVIEW_FOV_DEG = 1.0           # SkyView image field of view

OCCLUSION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Fraction of stars removed

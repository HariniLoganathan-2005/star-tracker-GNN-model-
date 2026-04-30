"""
Setup Data — One-Time Data Preparation Script

Downloads the Hipparcos catalogue and builds the triangle matching database.
This needs to be run once before the pipeline can be used.

Usage:
    python setup_data.py

This will:
    1. Download Hipparcos catalogue from VizieR (≤ mag 6.5)
    2. Build the triangle matching KD-tree database
    3. Generate a small set of synthetic test images
"""

import os
import sys
import logging
import time

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from catalogue.hipparcos import download_hipparcos, load_catalogue
from modules.m5_triangle_match import build_triangle_database
from synthetic.image_generator import generate_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('setup_data')


def main():
    start_time = time.time()

    print("=" * 60)
    print("  STAR TRACKER PIPELINE — DATA SETUP")
    print("=" * 60)
    print()

    # =========================================================================
    # Step 1: Download Hipparcos Catalogue
    # =========================================================================
    print("[1/3] Downloading Hipparcos catalogue...")
    print(f"      Magnitude limit: ≤ {config.HIPPARCOS_MAG_LIMIT}")

    if os.path.exists(config.CATALOGUE_FILE):
        print(f"      Catalogue already exists at {config.CATALOGUE_FILE}")
        print("      Loading existing catalogue...")
        catalogue = load_catalogue(config.CATALOGUE_FILE)
    else:
        try:
            df = download_hipparcos(config.CATALOGUE_FILE, config.HIPPARCOS_MAG_LIMIT)
            catalogue = load_catalogue(config.CATALOGUE_FILE)
        except Exception as e:
            logger.error(f"Failed to download catalogue: {e}")
            logger.error("Check your internet connection and try again.")
            sys.exit(1)

    n_stars = len(catalogue['hip_ids'])
    print(f"      ✓ Catalogue loaded: {n_stars} stars")
    print()

    # =========================================================================
    # Step 2: Build Triangle Matching Database
    # =========================================================================
    print("[2/3] Building triangle matching database...")
    print(f"      Max pair angle: {config.TRIANGLE_MAX_PAIR_ANGLE_DEG}°")
    print("      This may take several minutes for ~9000 stars...")

    if os.path.exists(config.TRIANGLE_DB_FILE):
        print(f"      Database already exists at {config.TRIANGLE_DB_FILE}")
        print("      Delete it to rebuild.")
    else:
        try:
            tri_db = build_triangle_database(catalogue, config.TRIANGLE_DB_FILE)
            n_pairs = len(tri_db.pair_distances)
            print(f"      ✓ Database built: {n_pairs} star pairs")
            size_mb = os.path.getsize(config.TRIANGLE_DB_FILE) / 1e6
            print(f"      ✓ Saved: {config.TRIANGLE_DB_FILE} ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to build triangle database: {e}")
            sys.exit(1)
    print()

    # =========================================================================
    # Step 3: Generate Synthetic Test Images
    # =========================================================================
    print("[3/3] Generating synthetic test images...")
    n_test = 20  # Small test set for initial validation

    existing = len([f for f in os.listdir(config.SYNTHETIC_TEST_DIR)
                    if f.endswith('.fits')]) if os.path.exists(config.SYNTHETIC_TEST_DIR) else 0

    if existing >= n_test:
        print(f"      {existing} test images already exist, skipping")
    else:
        try:
            metadata = generate_dataset(catalogue, config.SYNTHETIC_TEST_DIR,
                                         n_images=n_test, prefix="test", seed=42)
            print(f"      ✓ Generated {len(metadata)} test images")
        except Exception as e:
            logger.error(f"Failed to generate synthetic images: {e}")
            sys.exit(1)

    # =========================================================================
    # Done
    # =========================================================================
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  SETUP COMPLETE ({elapsed:.1f} seconds)")
    print("=" * 60)
    print()
    print("  You can now run the pipeline:")
    print(f"    python main.py --image data/synthetic/test/test_0001.fits")
    print()
    print("  Or run the full validation suite:")
    print(f"    python main.py --validate")
    print()


if __name__ == "__main__":
    main()

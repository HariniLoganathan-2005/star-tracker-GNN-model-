import sys
import os
import logging
from catalogue.hipparcos import load_catalogue
from synthetic.image_generator import generate_synthetic_image, save_synthetic_image
from camera.camera_model import StarTrackerCamera
import config

logging.basicConfig(level=logging.INFO)

# Load catalogue
catalogue = load_catalogue(config.CATALOGUE_FILE)
camera = StarTrackerCamera()

constellations = [
    ("ursa_major", 180.0, 55.0),
    ("cassiopeia", 15.0, 60.0),
    ("scorpius", 247.5, -30.0),
    ("cygnus", 307.5, 40.0),
    ("leo", 157.5, 15.0)
]

for name, ra, dec in constellations:
    print(f"Generating synthetic image for {name} (RA={ra}, Dec={dec})")
    result = generate_synthetic_image(
        catalogue=catalogue,
        ra_deg=ra,
        dec_deg=dec,
        roll_deg=0.0,
        camera=camera,
        add_noise_flag=True
    )
    save_synthetic_image(result, os.path.join(os.getcwd(), f"synthetic_{name}.fits"))
    print(f"Saved synthetic_{name}.fits with {result['n_stars_rendered']} stars.")

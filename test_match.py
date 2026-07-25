import sys
import logging
import config
from modules.m1_image_input import load_fits_image
from modules.m2_preprocessing import preprocess_image
from modules.m3_star_detection import detect_stars
from modules.m4_pixel_to_vector import convert_pixels_to_vectors
from modules.m5_triangle_match import match_stars, load_triangle_database
from catalogue.hipparcos import load_catalogue
from camera.camera_model import StarTrackerCamera

logging.basicConfig(level=logging.DEBUG)
config.HIPPARCOS_MAG_LIMIT = 6.5
config.TRIANGLE_ANGLE_TOLERANCE_DEG = 0.010

fits = load_fits_image('synthetic_cassiopeia.fits')
cam = StarTrackerCamera()
img = preprocess_image(fits.data)
stars = detect_stars(img)
vecs = convert_pixels_to_vectors(stars, cam)
cat = load_catalogue(config.CATALOGUE_FILE)
db = load_triangle_database(config.TRIANGLE_DB_FILE)
matches, ok = match_stars(vecs, db, cat)
print("Match success:", ok)
print("Matches:", len(matches))

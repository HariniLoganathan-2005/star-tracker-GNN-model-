"""Quick diagnostic: check body vector convention."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from catalogue.catalogue_utils import *
from camera.camera_model import StarTrackerCamera
from catalogue.hipparcos import load_catalogue
import config

catalogue = load_catalogue(config.CATALOGUE_FILE)
camera = StarTrackerCamera()

q_true = pointing_to_quaternion(84, -1, 0)
R_true = quaternion_to_rotation_matrix(q_true)

vis = camera.get_visible_catalogue_stars(catalogue, q_true)
print(f"Visible: {len(vis['hip_ids'])} stars")

# Take Rigel (HIP 24436)
cat_idx = np.where(catalogue['hip_ids'] == 24436)[0][0]
r_inertial = catalogue['unit_vectors'][cat_idx]
print(f"Rigel inertial: {r_inertial}")

b_from_R = R_true @ r_inertial
print(f"Body (R@r):     {b_from_R}")

b_from_cam = camera.pixel_to_unit_vector(244.4, 149.4)
print(f"Body (camera):  {b_from_cam}")

ang = angular_distance(b_from_R, b_from_cam)
print(f"Angle: {ang*3600:.1f} arcsec")

# Test with 5 brightest
idx = np.argsort(vis['vmag'])[:5]
print(f"\n5 brightest stars:")
for k, i in enumerate(idx):
    body_R = vis['body_vectors'][i]
    body_cam = camera.pixel_to_unit_vector(vis['u'][i], vis['v'][i])
    ref = vis['inertial_vectors'][i]
    ang1 = angular_distance(body_R, body_cam)
    hip = vis['hip_ids'][i]
    
    # What does QUEST see?
    predicted = R_true @ ref
    err = angular_distance(predicted, body_cam)
    
    print(f"  HIP {hip}: R@r vs camera = {ang1*3600:.1f} arcsec, "
          f"QUEST predicted vs camera = {err*3600:.1f} arcsec")

# Run mini QUEST on the 5 brightest with TRUE identifications
print(f"\nMini QUEST test with 5 KNOWN-CORRECT matches:")
body_vecs = np.array([camera.pixel_to_unit_vector(vis['u'][i], vis['v'][i]) for i in idx])
ref_vecs = np.array([vis['inertial_vectors'][i] for i in idx])

# Construct B matrix manually
B = np.zeros((3, 3))
for i in range(5):
    B += np.outer(body_vecs[i], ref_vecs[i])
B /= 5

S = B + B.T
sigma = np.trace(B)
Z = np.array([B[1,2]-B[2,1], B[2,0]-B[0,2], B[0,1]-B[1,0]])

K = np.zeros((4, 4))
K[0,0] = sigma
K[0,1:] = Z
K[1:,0] = Z
K[1:,1:] = S - sigma * np.eye(3)

eigenvalues, eigenvectors = np.linalg.eigh(K)
max_idx = np.argmax(eigenvalues)
q_est = eigenvectors[:, max_idx]
q_est = q_est / np.linalg.norm(q_est)
if q_est[0] < 0:
    q_est = -q_est

print(f"q_true:     {q_true}")
print(f"q_est:      {q_est}")
print(f"q_angle:    {quaternion_angle(q_true, q_est):.4f} deg")

R_est = quaternion_to_rotation_matrix(q_est)
ra_est, dec_est = quaternion_to_pointing(q_est)
print(f"Pointing:   RA={ra_est:.4f}, Dec={dec_est:.4f}")
print(f"Truth:      RA=84.0000, Dec=-1.0000")

# Per-star residuals
for i in range(5):
    pred = R_est @ ref_vecs[i]
    err = angular_distance(pred, body_vecs[i])
    print(f"  Star {i} (HIP {vis['hip_ids'][idx[i]]}): residual = {err*3600:.2f} arcsec")

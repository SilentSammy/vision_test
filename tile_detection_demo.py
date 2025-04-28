import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2
from pose_estimation import estimate_square_pose, estimate_marker_pose, find_arucos, find_quadrilaterals, estimate_camera_pose, get_camera_matrix, draw_quad
from sim_tools import orient_object, move_object, sim, get_image
from keybrd import rising_edge

# Connect and get simulator objects
cams = [
    sim.getObject('/VisionTest[1]/visionSensor'),
    sim.getObject('/VisionTest[2]/visionSensor')
]
test_cam = cams[0]
cone = sim.getObject('/VisionTest[0]/Cone')

# Compute the camera matrix and distortion coefficients
camera_matrix = get_camera_matrix(x_res=1024, y_res=1024, fov_x_deg=60)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# --- Main program ---
tile_colors = [
    {'lbl': '1', 'lh': (0, 0, 80), 'uh': (0, 0, 90)},  # Dark Gray
    {'lbl': '2', 'lh': (0, 0, 190), 'uh': (0, 0, 200)}, # Light Gray
]

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        # Get image from the camera
        if rising_edge('1'):
            test_cam = cams[(cams.index(test_cam) + 1) % len(cams)]
            print(f"Switched to camera {test_cam}")
        frame = get_image(test_cam)

        # Optionally save the image
        if rising_edge('2'):
            cv2.imwrite('screenshot.png', frame)
            print("Screenshot saved as screenshot.png")

        # Find tiles
        tiles = []
        for idx, color in enumerate(tile_colors):
            sqrs = find_quadrilaterals(frame, lower_hsv=color['lh'], upper_hsv=color['uh'])
            for sqr in sqrs:
                tiles.append({'lbl': color['lbl'], 'shape': sqr})
        
        # Show tiles
        for tile in tiles:
            draw_quad(frame, tile['shape'])
            cv2.putText(frame, tile['lbl'], tuple(np.mean(tile['shape'], axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        cv2.namedWindow("Vision Sensor", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Vision Sensor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

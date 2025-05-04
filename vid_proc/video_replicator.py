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
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Connect and get simulator objects
cone = sim.getObject('/VisionTest[0]/Cone')
sim_anchor = sim.getObject('/VisionTest[0]/Anchor')
anchors = json.load(open('anchors.json')) # It's a dictionary of anchor_id -> anchor, each anchor is a dict x and y offsets
cam_matrix = get_camera_matrix(x_res=848, y_res=480, fov_x_deg=60)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
tile_size = 0.3

last_anchor_id = None
def update_anchor_position(anchor_id):
    global last_anchor_id
    if last_anchor_id == anchor_id:
        return
    if str(anchor_id) not in anchors:
        print(f"Anchor ID {anchor_id} not found in anchors.json")
        return
    anchor = anchors[str(anchor_id)]
    x_off = anchor['x'] * tile_size
    y_off = anchor['y'] * tile_size
    move_object(sim_anchor, x_off, y_off)
    last_anchor_id = anchor_id

def update_cone_position(anchor_id, quad, frame):
    if str(anchor_id) not in anchors:
        print(f"Anchor ID {anchor_id} not found in anchors.json")
        return
    anchor = anchors[str(anchor_id)]

    quad = np.array(quad, dtype=np.int32).reshape(1, 4, 2)
    yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = estimate_square_pose(quad, frame, cam_matrix, dist_coeffs, tile_size)
    cam_pose = estimate_camera_pose(yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw)

    x, y, z, alpha, beta, gamma = cam_pose
    x_off = anchor['x'] * tile_size
    y_off = anchor['y'] * tile_size
    x += x_off
    y += y_off
    
    move_object(cone, x=x, y=y, z=z)
    orient_object(cone, alpha=alpha, beta=beta, gamma=gamma)
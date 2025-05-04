import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2
from pose_estimation import estimate_square_pose, estimate_marker_pose, find_arucos, find_quadrilaterals, estimate_camera_pose, get_camera_matrix, draw_quad, project_point_to_plane
from sim_tools import orient_object, move_object, sim, get_image
from keybrd import rising_edge
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Get simulator objects
cone = sim.getObject('/VisionTest[0]/Cone')
sim_anchor = sim.getObject('/VisionTest[0]/Anchor')
sim_ducks = [
    sim.getObject('/VisionTest[0]/Duck[0]'),
    sim.getObject('/VisionTest[0]/Duck[1]'),
    sim.getObject('/VisionTest[0]/Duck[2]'),
    sim.getObject('/VisionTest[0]/Duck[3]'),
    sim.getObject('/VisionTest[0]/Duck[4]'),
    sim.getObject('/VisionTest[0]/Duck[5]'),
    sim.getObject('/VisionTest[0]/Duck[6]'),
]

# Get stuff
anchors = json.load(open('anchors.json')) # It's a dictionary of anchor_id -> anchor, each anchor is a dict x and y offsets
cam_matrix = get_camera_matrix(x_res=848, y_res=480, fov_x_deg=60)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
tile_size = 0.3

last_anchor_id = None
def update_anchor_position(anchor_id):
    global last_anchor_id
    if last_anchor_id == anchor_id:
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

def sync_to_video(anchor_tile, frame, ducks_pos):
    anchor_id = anchor_tile['id']
    quad = anchor_tile['shape']
    if str(anchor_id) not in anchors:
        print(f"Anchor ID {anchor_id} not found in anchors.json")
        return
    anchor = anchors[str(anchor_id)]
    x_off = anchor['x'] * tile_size
    y_off = anchor['y'] * tile_size
    update_anchor_position(anchor_id)
    update_cone_position(anchor_id, quad, frame)
    if ducks_pos:
        ducks_cnt = min(len(ducks_pos), len(sim_ducks))
        for i in range(ducks_cnt):
            sim_duck = sim_ducks[i]
            duck_pos = ducks_pos[i]
            point_pos = project_point_to_plane(quad, duck_pos, 0.2)
            point_pos = (-float(point_pos[0])+y_off, -float(point_pos[1])+x_off)
            move_object(sim_duck, y=point_pos[0], x=point_pos[1])

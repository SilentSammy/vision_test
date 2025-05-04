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

last_anchor_id = None
def update_anchor_position(anchor_id, tile_size = 0.3):
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
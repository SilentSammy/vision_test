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

duck_count = 7
if __name__ != '__main__': # Get simulator objects
    cone = sim.getObject('/VisionTest[0]/Cone')
    sim_anchor = sim.getObject('/VisionTest[0]/Anchor')
    sim_ducks = [sim.getObject(f'/VisionTest[0]/Duck[{i}]') for i in range(duck_count)]

# Get stuff
anchors = json.load(open('anchors.json')) # It's a dictionary of anchor_id -> anchor, each anchor is a dict x and y offsets
cam_matrix = get_camera_matrix(x_res=848, y_res=480, fov_x_deg=60)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
tile_size = 0.3

def get_cam_pose(anchor, quad, frame):
    quad = np.array(quad, dtype=np.int32).reshape(1, 4, 2)
    yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = estimate_square_pose(quad, frame, cam_matrix, dist_coeffs, tile_size)
    cam_pose = estimate_camera_pose(yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw)

    x, y, z, alpha, beta, gamma = cam_pose
    x += anchor['x'] * tile_size
    y += anchor['y'] * tile_size
    return x, y, z, alpha, beta, gamma

def get_ducks_pos(anchor, quad, ducks_img_pos):
    x_off = anchor['x'] * tile_size
    y_off = anchor['y'] * tile_size

    ducks_pos = []
    if ducks_img_pos:
        ducks_cnt = min(len(ducks_img_pos), duck_count)
        for i in range(ducks_cnt):
            duck_pos = ducks_img_pos[i]
            point_pos = project_point_to_plane(quad, duck_pos, 0.2)
            point_pos = (-float(point_pos[0])+y_off, -float(point_pos[1])+x_off)
            ducks_pos.append(point_pos)
    return ducks_pos

def get_frame_data(anchor_tile, frame, ducks_img_pos):
    anchor_id = anchor_tile['id']
    if str(anchor_id) not in anchors:
        return None
    quad = anchor_tile['shape']
    sim_frame_data = {}

    # Save anchor position
    anchor = anchors[str(anchor_id)]
    x_off = anchor['x'] * tile_size
    y_off = anchor['y'] * tile_size
    sim_frame_data['anchor'] = (x_off, y_off)
    
    # Camera
    x, y, z, alpha, beta, gamma = get_cam_pose(anchor, quad, frame)
    sim_frame_data['camera'] = (x, y, z, alpha, beta, gamma)

    # Ducks
    ducks_pos = get_ducks_pos(anchor, quad, ducks_img_pos)
    sim_frame_data['ducks'] = ducks_pos
    return sim_frame_data

def sync_to_video(sim_frame_data):
    anchor = sim_frame_data['anchor']
    camera = sim_frame_data['camera']
    ducks = sim_frame_data['ducks']

    # Move the anchor
    move_object(sim_anchor, anchor[0], anchor[1])

    # Move the camera
    x, y, z, alpha, beta, gamma = camera
    move_object(cone, x=x, y=y, z=z)
    orient_object(cone, alpha=alpha, beta=beta, gamma=gamma)

    # Move the ducks
    for i, dp in enumerate(ducks):
        move_object(sim_ducks[i], y=dp[0], x=dp[1])

if __name__ == '__main__':
    from video_player import VideoPlayer
    anchor_frames = json.load(open('anchor_tiles.json')) # It's a dictionary of frame_idx -> anchor_tile, each anchor_tile is a dict with 'id' and 'shape' keys
    duck_frames = json.load(open('ducks.json')) # It's a dictionary of frame_idx -> ducks, each duck is a key value pair of id and position
    vp = VideoPlayer('input.mp4')

    sim_frames = {}
    for frame_idx, anchor_tile in anchor_frames.items():
        vp._frame_idx = int(frame_idx)
        frame = vp.get_frame()
        ducks_img_pos = duck_frames[frame_idx] if frame_idx  in duck_frames else {}
        ducks_img_pos = [position for position in ducks_img_pos.values()]
        sim_frame_data = get_frame_data(anchor_tile, frame, ducks_img_pos)
        if sim_frame_data is None:
            continue
        sim_frames[frame_idx] = sim_frame_data
    
    # Save the simulation frames to a file
    with open('sim_frames.json', 'w') as f:
        json.dump(sim_frames, f, indent=4)
    print("Simulation frames saved to sim_frames.json")

    # Extract duck-related data from sim_frames
    duck_pos = []
    for frame_idx, sim_frame_data in sim_frames.items():
        for i, duck in enumerate(sim_frame_data['ducks']):
            duck_pos.append((frame_idx, i, duck[0], duck[1]))
    # Save the duck positions to a CSV file
    with open('duck_positions.csv', 'w') as f:
        f.write("frame_idx,duck_id,x,y\n")
        for frame_idx, duck_id, x, y in duck_pos:
            f.write(f"{frame_idx},{duck_id},{x},{y}\n")


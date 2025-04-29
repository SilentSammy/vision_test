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


def rotate_quad(quad, rot_count):
    # Normalize rot_count to 0,1,2, or 3.
    rot = rot_count % 4

    # If no rotation is needed, return the original quad.
    if rot == 0:
        return quad

    new_quad = quad.copy()
    # Rotate 'rot' times
    for _ in range(rot):
        new_quad = np.array([
            new_quad[3],
            new_quad[0],
            new_quad[1],
            new_quad[2]
        ])
    return new_quad

def rotate_neighbors(neighbors, rot_count):
    if rot_count <= 0:
        return neighbors

    rotation_map = {
        'top_left': 'top_right',
        'top': 'right',
        'top_right': 'bottom_right',
        'right': 'bottom',
        'bottom_right': 'bottom_left',
        'bottom': 'left',
        'bottom_left': 'top_left',
        'left': 'top'
    }

    # Rotate once.
    new_neighbors = { rotation_map[k]: v for k, v in neighbors.items() }
    # Recursively rotate the required additional times.
    return rotate_neighbors(new_neighbors, rot_count - 1)

def get_neighbors(tiles, tile, threshold_px=50):
    neighbors = {}
    t_quad = tile['quad'].reshape(4, 2)
    
    for other in tiles:
        if other is tile:
            continue
        o_quad = other['quad'].reshape(4, 2)

        # Check for for adjacent and diagonal neighbors
        if (np.linalg.norm(t_quad[1] - o_quad[0]) <= threshold_px and np.linalg.norm(t_quad[2] - o_quad[3]) <= threshold_px):
            neighbors['right'] = other
        elif (np.linalg.norm(t_quad[0] - o_quad[1]) <= threshold_px and np.linalg.norm(t_quad[3] - o_quad[2]) <= threshold_px):
            neighbors['left'] = other
        elif (np.linalg.norm(t_quad[0] - o_quad[3]) <= threshold_px and np.linalg.norm(t_quad[1] - o_quad[2]) <= threshold_px):
            neighbors['top'] = other
        elif (np.linalg.norm(t_quad[3] - o_quad[0]) <= threshold_px and np.linalg.norm(t_quad[2] - o_quad[1]) <= threshold_px):
            neighbors['bottom'] = other
        elif (np.linalg.norm(t_quad[0] - o_quad[2]) <= threshold_px):
            neighbors['top_left'] = other
        elif (np.linalg.norm(t_quad[1] - o_quad[3]) <= threshold_px):
            neighbors['top_right'] = other
        elif (np.linalg.norm(t_quad[3] - o_quad[1]) <= threshold_px):
            neighbors['bottom_left'] = other
        elif (np.linalg.norm(t_quad[2] - o_quad[0]) <= threshold_px):
            neighbors['bottom_right'] = other
        
        # Check if all neighbors are found
        if len(neighbors) == 8:
            break

    return neighbors

# --- Main program ---
tile_size = 0.3
tile_colors = [
    {'id': 0, 'lh': (0, 0, 32), 'uh': (180, 30, 40)},  # Dark Gray with slight tints
    {'id': 1, 'lh': (0, 0, 138), 'uh': (180, 30, 146)}, # Light Gray with slight tints
]
anchor_tiles = [
    {
        'id': 0,
        'off_x': -4,
        'off_y': 1,
        'color': 1,
        'neighbors': {
            'top_left': 0,
            'top': 1,
            'top_right': 1,
            'left': 1,
            'right': 0,
            'bottom_left': 1,
            'bottom': 0,
            'bottom_right': 1,
        },
    },
    {
        'id': 1,
        'color': 1,
        'off_x': -4,
        'off_y': 5,
        'neighbors': {
            'top_left': 0,
            'top': 1,
            'top_right': 0,
            'left': 1,
            'right': 0,
            'bottom_left': 0,
            'bottom': 1,
            'bottom_right': 1,
        },
    },
    # {
    #     'id': 2,
    #     'color': 0,
    #     'off_x': 6,
    #     'off_y': 5,
    #     'neighbors': {
    #         'top_left': 0,
    #         'top': 0,
    #         'top_right': 0,
    #         'left': 0,
    #         'right': 0,
    #         'bottom_left': 1,
    #         'bottom': 0,
    #         'bottom_right': 0,
    #     },
    # },
    {
        'id': 3,
        'color': 0,
        'off_x': 4,
        'off_y': 2,
        'neighbors': {
            'top_left': 0,
            'top': 1,
            'top_right': 0,
            'left': 1,
            'right': 1,
            'bottom_left': 0,
            'bottom': 1,
            'bottom_right': 1,
        },
    },
    {
        'id': 4,
        'color': 1,
        'off_x': -2,
        'off_y': -5,
        'neighbors': {
            'top_left': 0,
            'top': 1,
            'top_right': 1,
            'left': 0,
            'right': 1,
            'bottom_left': 1,
            'bottom': 1,
            'bottom_right': 0,
        },
    },
    {
        'id': 5,
        'color': 1,
        'off_x': -4,
        'off_y': -5,
        'neighbors': {
            'top_left': 1,
            'top': 1,
            'top_right': 0,
            'left': 0,
            'right': 1,
            'bottom_left': 0,
            'bottom': 1,
            'bottom_right': 1,
        },
    },
    {
        'id': 6,
        'color': 0,
        'off_x': -3,
        'off_y': 3,
        'neighbors': {
            'top_left': 0,
            'top': 0,
            'top_right': 1,
            'left': 0,
            'right': 1,
            'bottom_left': 0,
            'bottom': 1,
            'bottom_right': 0,
        },
    },
    {
        'id': 7,
        'color': 1,
        'off_x': 3,
        'off_y': 5,
        'neighbors': {
            'top_left': 0,
            'top': 1,
            'top_right': 1,
            'left': 0,
            'right': 1,
            'bottom_left': 0,
            'bottom': 1,
            'bottom_right': 0,
        },
    },
    {
        'id': 8,
        'color': 1,
        'off_x': -2,
        'off_y': 5,
        'neighbors': {
            'top_left': 1,
            'top': 0,
            'top_right': 0,
            'left': 1,
            'right': 0,
            'bottom_left': 0,
            'bottom': 1,
            'bottom_right': 0,
        },
    },
]
for at in anchor_tiles:
    at['off_x'] *= tile_size
    at['off_y'] *= tile_size

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
                tiles.append({'id': None, 'color': color['id'], 'quad': sqr, 'center': np.mean(sqr, axis=0)})
        
        if tiles:
            # Assign ids to known tiles based on their neighbors
            for anchor in anchor_tiles:
                anchor_found = False
                for tile in [t for t in tiles if t['color'] == anchor['color']]:
                    is_match = False
                    rotation_match = None
                    
                    # Ignore tiles with missing neighbors
                    neighbors = get_neighbors(tiles, tile)
                    if len(neighbors) < 8 or any(v is None for v in neighbors.values()):
                        continue


                    # Try all four rotations.
                    for r in range(4):
                        rotated_neighbors = rotate_neighbors(neighbors, r)
                        match = True
                        for side, neighbor in rotated_neighbors.items():
                            # Compare anchor's expected neighbor color with the rotated neighbor's color.
                            if anchor['neighbors'][side] != neighbor['color']:
                                match = False
                                break
                        if match:
                            anchor_found = True
                            is_match = True
                            rotation_match = r
                            break

                    # If we have a match, update the tile.
                    if is_match:
                        tile['id'] = anchor['id']
                        tile['off_x'] = anchor['off_x']
                        tile['off_y'] = anchor['off_y']
                        tile['roll_count'] = rotation_match
                        tile['quad'] = rotate_quad(tile['quad'], rotation_match)
                        break
                
                if anchor_found:
                    break
            # Draw known tiles
            id_tiles = [t for t in tiles if t['id'] is not None]
            for tile in id_tiles:
                # Draw the tile and its ID
                draw_quad(frame, tile['quad'])
                cv2.putText(frame, str(tile['id']), tuple(tile['center'].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

                # Estimate pose of the tile
                tile_pose = estimate_square_pose(tile['quad'], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, square_length=tile_size)
                tile['pose'] = tile_pose
            
            # Get most optimal known tile for camera pose estimation
            if id_tiles:
                # Sort by closest to camera, then by least pitch + yaw
                id_tiles.sort(key=lambda t: (abs(t['pose'][3]) + abs(t['pose'][4]), t['pose'][5]))
                ref_tile = id_tiles[0]
                yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = ref_tile['pose']
                cam_pose = estimate_camera_pose(yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw)
                x_off, y_off = ref_tile['off_x'], ref_tile['off_y']
                cam_pose = list(cam_pose)
                cam_pose[0] += x_off
                cam_pose[1] += y_off
                
                # Move the cone to the estimated camera pose
                x, y, z, alpha, beta, gamma = cam_pose
                move_object(cone, x=x, y=y, z=z)
                orient_object(cone, alpha=alpha, beta=beta, gamma=gamma)

        cv2.namedWindow("Vision Sensor", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Vision Sensor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

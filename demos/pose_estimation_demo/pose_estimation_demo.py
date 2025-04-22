import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2
from pose_estimation import estimate_square_pose, estimate_marker_pose, find_arucos, find_quadrilaterals, estimate_camera_pose
from sim_tools import orient_object, move_object, sim, get_image

# Connect and get simulator objects
test_cam = sim.getObject('/VisionTest/visionSensor')
cone = sim.getObject('/VisionTest[1]/Cone')

# Compute the camera matrix and distortion coefficients
x_res = 1024
y_res = 1024
fov_x_deg = 60
focal_length = (x_res / 2) / math.tan(math.radians(fov_x_deg) / 2)
camera_matrix = np.array([[focal_length,      0.0, x_res / 2],
                          [     0.0, focal_length, y_res / 2],
                          [     0.0,      0.0,      1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# --- Main program ---
ref_objs = [
    {'lh': (60, 200, 200), 'uh': (60, 255, 255), 'x': 0.0, 'y': 0.0},  # Pure green (lower tolerance)
    {'lh': (0, 150, 150), 'uh': (10, 255, 255), 'x': -0.8, 'y': -0.8},  # Pure red
    {'lh': (110, 150, 150), 'uh': (130, 255, 255), 'x': 0.8, 'y': -0.8},  # Pure blue
    {'lh': (15, 100, 100), 'uh': (25, 255, 255), 'x': 0.8, 'y': 0.8},  # Orange
    {'lh': (140, 100, 100), 'uh': (160, 255, 255), 'x': -0.8, 'y': 0.8}  # Magenta
]

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        frame = get_image(test_cam)

        # Find ARUCO markers
        markers, aru_ids = find_arucos(frame)
        # markers = zip(aru_ids, markers)

        # Find squares
        squares = []
        for idx, ref_obj in enumerate(ref_objs):
            sqrs = find_quadrilaterals(frame, lower_hsv=ref_obj['lh'], upper_hsv=ref_obj['uh'])
            if sqrs:
                squares.append((idx, sqrs[0]))

        # Attempt to determine the camera pose based on the detected markers or squares (assuming the marker is at the origin, facing upward)
        cam_pose = None
        if markers:
            yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = estimate_marker_pose(markers[0], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, marker_length=0.2)
            print(f"ARUCO\tZ:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {cam_dist:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")
            cam_pose = estimate_camera_pose(yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw)
        elif squares:
            # Choose the closest square to the camera
            square_poses = [ (sqr_id, quad, *estimate_square_pose(quad, frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, square_length=0.2)) for sqr_id, quad in squares ]
            square_poses.sort(key=lambda x: x[5])  # Sort by cam_dist (5th element)
            sqr_idx, quad, yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = square_poses[0]

            # Estimate the camera pose based on the square
            print(f"SQUARE {sqr_idx}\tZ:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {cam_dist:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")
            cv2.polylines(frame, [quad], isClosed=True, color=(255, 255, 255), thickness=4)
            cam_pose = estimate_camera_pose(yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw)
            x_off, y_off = ref_objs[sqr_idx]['x'], ref_objs[sqr_idx]['y']
            cam_pose = list(cam_pose)  # Convert to list for modification
            cam_pose[0] += x_off
            cam_pose[1] += y_off

        # Visualize the estimated camera pose in the simulation
        if cam_pose:
            x, y, z, alpha, beta, gamma = cam_pose
            move_object(cone, x=x, y=y, z=z)
            orient_object(cone, alpha=alpha, beta=beta, gamma=gamma)
            # print(f"CAM\tX: {x:.2f}\tY:{y:.2f}\tZ:{z:.2f}\tYaw:{math.degrees(alpha):.2f}\tPit:{math.degrees(beta):.2f}\tRol:{math.degrees(gamma):.2f}\n")

        cv2.namedWindow("Vision Sensor", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Vision Sensor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

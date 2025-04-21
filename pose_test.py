from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2
from pose_estimation import estimate_square_pose, estimate_marker_pose, find_arucos, find_quadrilaterals
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

def screenshot_and_exit(cam_handle):
    frame = get_image(cam_handle)
    cv2.imshow('Vision Sensor Image', frame)
    cv2.imwrite('last_frame.png', frame)
    cv2.waitKey(1000)
    raise SystemExit

def unroll_offsets(offset_x, offset_y, roll_deg):
    # Convert roll angle to radians.
    roll_rad = math.radians(roll_deg)  # use positive roll
    # Build the 2D rotation matrix for roll.
    cos_r = math.cos(roll_rad)
    sin_r = math.sin(roll_rad)
    # Rotate the (offset_x, offset_y) vector.
    x_unrolled = offset_x * cos_r - offset_y * sin_r
    y_unrolled = offset_x * sin_r + offset_y * cos_r
    return x_unrolled, y_unrolled

def unroll_angle_offsets(yaw_deg, pitch_deg, roll_deg):
    """
    Given angular offsets (e.g. cam_yaw, cam_pitch) in degrees and a roll angle, 
    rotates the (yaw, pitch) vector by -roll so that the roll’s influence is removed.
    
    Returns:
        A tuple (unrolled_yaw, unrolled_pitch) in degrees.
    """
    # Convert to radians.
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)
    
    # Apply 2D rotation: we rotate the vector (yaw_rad, pitch_rad) by -roll_rad.
    unrolled_yaw_rad = math.cos(-roll_rad)*yaw_rad - math.sin(-roll_rad)*pitch_rad
    unrolled_pitch_rad = math.sin(-roll_rad)*yaw_rad + math.cos(-roll_rad)*pitch_rad
    
    # Convert back to degrees.
    unrolled_yaw = math.degrees(unrolled_yaw_rad)
    unrolled_pitch = math.degrees(unrolled_pitch_rad)
    return unrolled_yaw, unrolled_pitch

def estimate_camera_pose(yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw):
    cam_yaw, cam_pitch = unroll_angle_offsets(cam_yaw, cam_pitch, roll)
    yaw_rad = math.radians(cam_yaw-yaw)
    pitch_rad = math.radians(cam_pitch+pitch)
    y = cam_dist * math.cos(pitch_rad) * math.sin(yaw_rad)  # rightward displacement
    x = cam_dist * math.sin(pitch_rad)                      # forward displacement
    z = cam_dist * math.cos(pitch_rad) * math.cos(yaw_rad)  # upward displacement
    alpha = math.radians(yaw)
    beta = math.radians(pitch)
    gamma = 0

    return x, y, z, alpha, beta, gamma

# --- Main program ---
frame = None
try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        frame = get_image(test_cam)

        # Find green squares and ARUCO markers
        squares = find_quadrilaterals(frame, lower_hsv=(45, 100, 100), upper_hsv=(75, 255, 255))
        markers, _ = find_arucos(frame)

        # Attempt to determine the camera pose based on the detected markers or squares (assuming the marker is at the origin, facing upward)
        cam_pose = None
        if markers:
            yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = estimate_marker_pose(markers[0], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, marker_length=0.1)
            print(f"ARUCO\tZ:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {cam_dist:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")
            cam_pose = estimate_camera_pose(yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw)
        elif squares:
            yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = estimate_square_pose(squares[0], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, square_length=0.1)
            print(f"SQUARE\tZ:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {cam_dist:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")
            cam_pose = estimate_camera_pose(yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw)
            for square in squares: # Draw the detected squares
                cv2.polylines(frame, [square], isClosed=True, color=(255, 255, 255), thickness=2)
        
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

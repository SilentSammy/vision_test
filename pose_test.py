from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2
from pose_estimation import estimate_square_pose, estimate_marker_pose, find_arucos, find_quadrilaterals
from sim_tools import orient_object, move_object, sim

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

def get_image(vision_sensor_handle):
    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def screenshot_and_exit(cam_handle):
    frame = get_image(cam_handle)
    cv2.imshow('Vision Sensor Image', frame)
    cv2.imwrite('last_frame.png', frame)
    cv2.waitKey(1000)
    raise SystemExit

def combine_angles(cam_angle, object_angle):
    """
    Combines the camera’s translation-derived angle and the object's rotation-derived angle.
    If either signal is near zero then we consider that there’s no rotation.
    Otherwise, we return a value whose magnitude is the smaller of the two,
    but whose sign is taken from the object's rotation-derived angle.
    
    This ensures that if the camera or marker rotate oppositely the result will have the correct sign.
    """
    if abs(cam_angle) < 1e-6 or abs(object_angle) < 1e-6:
        return 0.0
    # Use the smaller magnitude and the sign of the object's angle.
    return math.copysign(min(abs(cam_angle), abs(object_angle)), object_angle)

# --- Main program ---
frame = None
try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        frame = get_image(test_cam)

        squares = find_quadrilaterals(frame, lower_hsv=(45, 100, 100), upper_hsv=(75, 255, 255))
        markers, _ = find_arucos(frame)

        if squares:
            yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = estimate_square_pose(squares[0], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, square_length=0.1)
            print(f"SQUARE\tZ:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {cam_dist:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")

        if markers:
            yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = estimate_marker_pose(markers[0], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, marker_length=0.1)
            print(f"ARUCO\tZ:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {cam_dist:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")

            # Use polar coordinates to determine the camera's position
            yaw_rad = math.radians(cam_yaw-yaw)
            pitch_rad = math.radians(cam_pitch+pitch)
            y = cam_dist * math.cos(pitch_rad) * math.sin(yaw_rad)  # rightward displacement
            x = cam_dist * math.sin(pitch_rad)                      # forward displacement
            z = cam_dist * math.cos(pitch_rad) * math.cos(yaw_rad)  # upward displacement
            move_object(cone, x=x, y=y, z=z)

            # Determine the camera's orientation
            # combined_pitch = combine_angles(cam_pitch, -pitch)  # note the sign: often the object pitch comes out with opposite sign
            # combined_yaw   = combine_angles(cam_yaw, yaw)
            alpha = math.radians(yaw)
            beta = math.radians(pitch)
            gamma = 0
            orient_object(cone, alpha=alpha, beta=beta, gamma=gamma)
            
        # Draw the detected squares
        for square in squares:
            cv2.polylines(frame, [square], isClosed=True, color=(255, 255, 255), thickness=2)

        cv2.namedWindow("Vision Sensor", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Vision Sensor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

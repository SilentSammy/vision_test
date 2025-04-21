from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2
from pose_estimation import estimate_square_pose, estimate_marker_pose, find_arucos, find_quadrilaterals

# Connect and get simulator objects
client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')
car_cam = sim.getObject('/LineTracer/visionSensor')
test_cam = sim.getObject('/VisionTest/visionSensor')

# Camera parameters
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
camera_matrix = np.array([[443.4, 0, 256],
                          [0, 443.4, 256],
                          [0, 0, 1]], dtype=np.float32)

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

# --- Main program ---
frame = None

# screenshot_and_exit(test_cam)

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        # frame = cv2.imread('resources/square_1m.png')
        frame = get_image(test_cam)

        lower_hsv = (45, 100, 100)  # Example HSV range for green
        upper_hsv = (75, 255, 255)

        squares = find_quadrilaterals(frame, lower_hsv, upper_hsv)
        markers, _ = find_arucos(frame)

        if squares:
            yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = estimate_square_pose(squares[0], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, square_length=0.1)
            print(f"SQUARE\tZ:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {cam_dist:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")

        if markers:
            yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = estimate_marker_pose(markers[0], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, marker_length=0.1)
            print(f"ARUCO\tZ:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {cam_dist:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")

        # Draw the detected squares
        for square in squares:
            cv2.polylines(frame, [square], isClosed=True, color=(255, 255, 255), thickness=2)

        cv2.imshow("Detected Squares", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

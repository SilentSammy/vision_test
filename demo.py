from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2
from pose_estimation import estimate_square_pose, estimate_marker_pose, find_arucos, find_quadrilaterals, estimate_camera_pose, find_corresponding_point
from sim_tools import orient_object, move_object, sim, get_image

# Connect and get simulator objects
test_cam = sim.getObject('/VisionTest/visionSensor')
demo_cam = sim.getObject('/Demo/visionSensor')
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

# --- Main program ---

def main():
    objs = []
    next_id = 0

    try:
        sim.startSimulation()

        while sim.getSimulationState() != sim.simulation_stopped:
            frame = get_image(demo_cam)
            sqrs = find_quadrilaterals(frame, lower_hsv=(50, 150, 150), upper_hsv=(70, 255, 255))
            new_objs = [{'id': None, 'ignore': False, 'shape': sqr} for sqr in sqrs]
            objs = new_objs
            # Create a list of previous discs of this color
            # prev_objs = objs.copy()
            # for new_obj in new_objs:
            #     corresponding_obj = find_corresponding_point(new_obj, prev_objs, threshold=frame.shape[1]*0.2)
            #     if corresponding_obj is not None: # It's a previously seen disc
            #         new_obj['id'] = corresponding_obj['id'] # update its id in the new list
            #         sqrs.index(corresponding_obj)['shape'] = new_obj['shape'] # update the obj in the old list
            #         prev_objs.remove(corresponding_obj) # remove it so we don't match it again
            #     else: # It's a new disc
            #         new_obj['id'] = next_id
            #         next_id += 1
            #         objs.append(new_obj)
            # Optionally, remove discs that have left the field of view
            # discs[color] = [d for d in discs[color] if d['id'] in (d['id'] for d in new_discs[color])]


            for obj in objs:
                sqr = obj['shape']
                cv2.polylines(frame, [sqr], isClosed=True, color=(255, 255, 255), thickness=2)
                # cv2.putText(frame, str(obj['id']), (sqr[0][0], sqr[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            
            cv2.namedWindow("Vision Sensor", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow("Vision Sensor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.imwrite('last_frame.png', frame)
                break
    finally:
        sim.stopSimulation()

def main1():
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
            if squares:
                # Choose the closest square to the camera
                square_poses = [ (sqr_id, quad, *estimate_square_pose(quad, frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, square_length=0.2)) for sqr_id, quad in squares ]
                square_poses.sort(key=lambda x: x[5])  # Sort by cam_dist (5th element)
                sqr_idx, quad, yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw = square_poses[0]

                # Estimate the camera pose based on the square
                print(f"SQUARE {sqr_idx}\tZ:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {cam_dist:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")
                cv2.polylines(frame, [quad], isClosed=True, color=(255, 255, 255), thickness=2)
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

if __name__ == "__main__":
    main()
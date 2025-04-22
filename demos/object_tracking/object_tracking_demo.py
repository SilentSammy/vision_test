import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2
from pose_estimation import find_ellipses, find_corresponding_point
from sim_tools import sim, get_image

demo_cam = sim.getObject('/Demo/visionSensor')

# Camera parameters
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
camera_matrix = np.array([[443.4, 0, 256],
                          [0, 443.4, 256],
                          [0, 0, 1]], dtype=np.float32)

color = ((45, 100, 100), (75, 255, 255))
next_id = 0
objs = []

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        frame = get_image(demo_cam)

        # Find objects in the image (each object is dict of an id, and a shape in OpenCV format)
        new_objs = [{'id': None, 'shape': e} for e in find_ellipses(frame, lower_hsv=color[0], upper_hsv=color[1])]
        print(len(new_objs), "objects found")
        
        # Create a list of previous objects
        prev_objs = objs.copy()
        for new_obj in new_objs:
            # Find the corresponding object in the previous list
            new_center = new_obj['shape'][0]
            old_centers = [o['shape'][0] for o in prev_objs]
            corresponding_point = find_corresponding_point(new_center, old_centers, threshold=frame.shape[1]*0.2)
            
            if corresponding_point is not None: # It's a previously seen object
                corresponding_obj = prev_objs[old_centers.index(corresponding_point)]
                new_obj['id'] = corresponding_obj['id']
                corresponding_obj['shape'] = new_obj['shape'] # update the shape in the old list
                prev_objs.remove(corresponding_obj) # remove it so we don't match it again
            else: # It's a new object
                new_obj['id'] = next_id
                next_id += 1
                objs.append(new_obj)
        # Optionally, remove objects that have left the field of view
        objs = [o for o in objs if o['id'] in (o['id'] for o in new_objs)]
        
        # Draw the id on the objects
        for new_obj in objs:
            center = new_obj['shape'][0]
            cv2.putText(frame, str(new_obj['id']), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.namedWindow("Vision Sensor", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Vision Sensor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

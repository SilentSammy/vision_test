import math
import time
import sys
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Connect to CoppeliaSim
client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')

# Object Handles
sphere = sim.getObject('/Sphere')
vision_sensor = sim.getObject('/visionSensor')
aruco_tag = sim.getObject('/arucoMarker')

# Setup
dt = 0
start = time.time()
last = start
elapsed = 0

def time_step():
    global dt, last, elapsed
    now = time.time()
    dt = now - last
    last = now
    elapsed = now - start

def translate_object(object_handle, x=0, y=0, z=0):
    """Adds to an object's position."""
    position = sim.getObjectPosition(object_handle, -1)
    sim.setObjectPosition(object_handle, -1, [position[0] + x, position[1] + y, position[2] + z])

def move_object(object_handle, x=None, y=None, z=None):
    """Teleports an object to a specific position."""
    position = sim.getObjectPosition(object_handle, -1)
    position[0] = x if x is not None else position[0]
    position[1] = y if y is not None else position[1]
    position[2] = z if z is not None else position[2]
    sim.setObjectPosition(object_handle, -1, position)

def rotate_object(object_handle, alpha=0, beta=0, gamma=0):
    """Adds to an object's orientation (in radians)."""
    orientation = sim.getObjectOrientation(object_handle, -1)
    sim.setObjectOrientation(object_handle, -1, [orientation[0] + alpha, orientation[1] + beta, orientation[2] + gamma])

def orient_object(object_handle, alpha=None, beta=None, gamma=None):
    """Sets an object's orientation to specific angles (in radians)."""
    orientation = sim.getObjectOrientation(object_handle, -1)
    orientation[0] = alpha if alpha is not None else orientation[0]
    orientation[1] = beta if beta is not None else orientation[1]
    orientation[2] = gamma if gamma is not None else orientation[2]
    sim.setObjectOrientation(object_handle, -1, orientation)

def move_sphere():
    # Move in a circle
    x_pos = math.sin(elapsed)
    y_pos = math.cos(elapsed)
    move_object(sphere, x=x_pos, y=y_pos)

def orient_camera():
    pitch = math.sin(elapsed) * math.pi / 4
    pitch += math.pi
    orient_object(vision_sensor, alpha=0, beta=pitch, gamma=0)

def get_image():
    # Actualizar imagen del sensor
    sim.handleVisionSensor(vision_sensor)
    img, resolution = sim.getVisionSensorImg(vision_sensor)

    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img

def main():
    try:
        sim.startSimulation()
        while sim.getSimulationState() != sim.simulation_stopped: # So we can stop the simulation from CoppeliaSim
            time_step()

            move_sphere()

            orient_camera()

            cv2.imshow('Imagen del Sensor de Visión', get_image())
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        # Stop simulation and cleanup
        cv2.destroyAllWindows()
        sim.stopSimulation()

if __name__ == "__main__":
    main()

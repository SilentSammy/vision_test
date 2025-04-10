import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
import cv2
import numpy as np

class DifferentialCar:
    def __init__(self, left_wheel=None, right_wheel=None):
        # Get wheel handles from the global sim object
        self.left_wheel = left_wheel or sim.getObject('/DynamicLeftJoint')
        self.right_wheel = right_wheel or sim.getObject('/DynamicRightJoint')
        
        # Internal speeds (m/s and rad/s)
        self._linear_speed = 0.0
        self._angular_speed = 0.0

        # Differential car constants
        self.nominalLinearVelocity = 0.3    # nominal linear speed (m/s)
        self.wheelRadius = 0.027            # wheel radius (m)
        self.interWheelDistance = 0.119     # distance between wheels (m)
        
        # Apply initial wheel speeds
        self._update_wheel_velocities()

    def _update_wheel_velocities(self):
        left_speed = (self._linear_speed - (self._angular_speed * self.interWheelDistance / 2)) / self.wheelRadius
        right_speed = (self._linear_speed + (self._angular_speed * self.interWheelDistance / 2)) / self.wheelRadius
        
        # Batch update using stepping
        client.setStepping(True)
        sim.setJointTargetVelocity(self.left_wheel, left_speed)
        sim.setJointTargetVelocity(self.right_wheel, right_speed)
        client.setStepping(False)
    
    @property
    def linear_speed(self):
        return self._linear_speed
    
    @linear_speed.setter
    def linear_speed(self, value):
        self._linear_speed = value
        self._update_wheel_velocities()
    
    @property
    def angular_speed(self):
        return self._angular_speed
    
    @angular_speed.setter
    def angular_speed(self, value):
        self._angular_speed = value
        self._update_wheel_velocities()

# Monitor key presses
pressed_keys = set()
def on_press(key):
    pressed_keys.add(key.char if hasattr(key, 'char') else str(key))
def on_release(key):
    pressed_keys.discard(key.char if hasattr(key, 'char') else str(key))
keyboard.Listener(on_press=on_press, on_release=on_release).start()

# Connect and get simulator objects
client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')
car = DifferentialCar()
car_cam = sim.getObject('/LineTracer/visionSensor')

def get_image(vision_sensor_handle):
    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def control():
    # Maximum speeds (m/s and rad/s)
    max_linear_speed = 0.5
    max_angular_speed = math.radians(90)

    car.linear_speed = max_linear_speed if 'w' in pressed_keys else -max_linear_speed if 's' in pressed_keys else 0
    car.angular_speed = max_angular_speed if 'a' in pressed_keys else -max_angular_speed if 'd' in pressed_keys else 0

try:
    sim.startSimulation()
    
    while sim.getSimulationState() != sim.simulation_stopped:
        control()
        
        cv2.imshow('Vision Sensor Image', get_image(car_cam))
        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.01)
        
finally:
    sim.stopSimulation()

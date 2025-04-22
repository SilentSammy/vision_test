from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2

class DifferentialCar:
    def __init__(self, left_wheel=None, right_wheel=None):
        self._last_time = None  # Initialize last_time to None

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
        
        # Convert to Python float to avoid serialization issues.
        left_speed = float(left_speed)
        right_speed = float(right_speed)
        
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

    def stop(self):
        self.linear_speed = 0.0
        self.angular_speed = 0.0
    
    def slow_down(self, damping_factor=1):
        current_time = time.time()
        dt = current_time - self._last_time if self._last_time and (current_time - self._last_time) <= 0.5 else 0.0
        self._last_time = current_time
        car.linear_speed -= car.linear_speed * damping_factor * dt
    
    def spin_down(self, damping_factor=1):
        current_time = time.time()
        dt = current_time - self._last_time if self._last_time and (current_time - self._last_time) <= 0.5 else 0.0
        self._last_time = current_time
        car.angular_speed -= car.angular_speed * damping_factor * dt

# Monitor key presses
pressed_keys = set()
toggles = {}
def is_toggled(key):
    if key not in toggles:
        toggles[key] = False
    return toggles.get(key, False)
def on_press(key):
    key_repr = key.char if hasattr(key, 'char') else str(key)
    pressed_keys.add(key_repr)
    if key_repr in toggles:
        toggles[key_repr] = not toggles[key_repr]
def on_release(key):
    pressed_keys.discard(key.char if hasattr(key, 'char') else str(key))
keyboard.Listener(on_press=on_press, on_release=on_release).start()

# Connect and get simulator objects
client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')
car_cam = sim.getObject('/LineTracer/visionSensor')
car = DifferentialCar()

# Camera parameters
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
camera_matrix = np.array([[443.4, 0, 256],
                          [0, 443.4, 256],
                          [0, 0, 1]], dtype=np.float32)

# Control
cam_yaw = 0.0
heading_pid = PID(Kp=0.02, Ki=0, Kd=0, setpoint=0.0)
cam_dist = 0.32
distance_pid = PID(Kp=2, Ki=0, Kd=0, setpoint=cam_dist)
distance_pid.output_limits = (-0.2, 0.2)

def get_image(vision_sensor_handle):
    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def control():
    lin_vel = 0.5
    ang_vel = math.radians(45)
    car.linear_speed = (1 if 'w' in pressed_keys else -1 if 's' in pressed_keys else 0.0) * lin_vel
    car.angular_speed = (1 if 'a' in pressed_keys else -1 if 'd' in pressed_keys else 0.0) * ang_vel

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        frame = get_image(car_cam)
        control()
        cv2.imshow('Vision Sensor Image', frame)
        cv2.setWindowProperty('Vision Sensor Image', cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

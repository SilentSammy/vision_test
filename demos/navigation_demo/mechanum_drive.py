import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import math
import cv2
from keybrd import is_pressed
from sim_tools import MechanumCar, DifferentialCar, get_image, sim

# Connect and get simulator objects
car = MechanumCar()

def control():
    max_lin_vel = 0.5
    max_ang_vel = math.radians(45)
    car.v_x = (1 if is_pressed('w') else -1 if is_pressed('s') else 0.0) * max_lin_vel
    car.v_y = (1 if is_pressed('q') else -1 if is_pressed('e') else 0.0) * max_lin_vel
    car.angular_speed = (1 if is_pressed('a') else -1 if is_pressed('d') else 0.0) * max_ang_vel

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        control()
finally:
    sim.stopSimulation()

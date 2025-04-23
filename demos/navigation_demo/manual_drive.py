import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import math
import cv2
from keybrd import is_pressed
from sim_tools import DifferentialCar, get_image, sim

# Connect and get simulator objects
car_cam = sim.getObject('/LineTracer/visionSensor')
car = DifferentialCar()

def control():
    max_lin_vel = 0.5
    max_ang_vel = math.radians(45)
    car.accelerate_to((1 if is_pressed('w') else -1 if is_pressed('s') else 0.0) * max_lin_vel, acc=1.5)
    car.spin_up_to((1 if is_pressed('a') else -1 if is_pressed('d') else 0.0) * max_ang_vel, acc=math.radians(180))

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

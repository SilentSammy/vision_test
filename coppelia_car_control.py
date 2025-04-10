import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Connect and get simulator objects
client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')
left_wheel = sim.getObject('/DynamicLeftJoint')
right_wheel = sim.getObject('/DynamicRightJoint')
speed = 1

try:
    sim.startSimulation()
    sim.setJointTargetVelocity(left_wheel, speed)
    sim.setJointTargetVelocity(right_wheel, speed)
    
    while sim.getSimulationState() != sim.simulation_stopped:
        speed = math.sin(time.time() / 2) * 2

        # Apply the computed speed to both wheels
        client.setStepping(True) # El simulador dejará de hacer su propio stepping brevemente, mientras aplicamos los cambios
        sim.setJointTargetVelocity(left_wheel, speed)
        sim.setJointTargetVelocity(right_wheel, speed)
        client.setStepping(False) # El simulador hará su propio stepping nuevamente, asegurando que ambos cambios se apliquen al mismo tiempo

        # Control the loop speed
        time.sleep(0.2)
finally:
    sim.stopSimulation()

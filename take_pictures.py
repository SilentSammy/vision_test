import os
import cv2
import numpy as np
import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Connect and get simulator objects
client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')
cal_cam = sim.getObject('/calibrationCamera')
chessboard = sim.getObject('/chessboard')

# --- Helper functions from calibrate_cam.py ---
def move_object(object_handle, x=None, y=None, z=None):
    """Teleports an object to a specific position."""
    position = sim.getObjectPosition(object_handle, -1)
    position[0] = x if x is not None else position[0]
    position[1] = y if y is not None else position[1]
    position[2] = z if z is not None else position[2]
    sim.setObjectPosition(object_handle, -1, position)

def orient_object(object_handle, alpha=None, beta=None, gamma=None):
    """Sets an object's orientation to specific angles (in radians)."""
    orientation = sim.getObjectOrientation(object_handle, -1)
    orientation[0] = alpha if alpha is not None else orientation[0]
    orientation[1] = beta if beta is not None else orientation[1]
    orientation[2] = gamma if gamma is not None else orientation[2]
    sim.setObjectOrientation(object_handle, -1, orientation)

def get_image(vision_sensor_handle):
    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def take_picture():
    return get_image(cal_cam)
# --- End helpers ---

def main():
    try:
        sim.startSimulation()
        output_dir = "calibration_captures"
        os.makedirs(output_dir, exist_ok=True)

        image_count = 0
        # Define grid offsets (in meters) relative to some nominal chessboard position.
        # Adjust these values so that the chessboard is always visible in the calibration camera.
        x_positions = [-0.5, -0.25, 0, 0.25, 0.5]
        y_positions = [-0.5, -0.25, 0, 0.25, 0.5]
        # Orientation angles (in degrees) to test different rotations. Convert to radians.
        rots_deg = [-10, 0, 10]
        rots = [math.radians(theta) for theta in rots_deg]

        # For example, assume the chessboard's nominal Z is 3m.
        nominal_z = 3.0

        # Loop over each combination of translation and rotation.
        for x_off in x_positions:
            for y_off in y_positions:
                for theta in rots:
                    # Set chessboard position. For a top-down view, we adjust X and Y.
                    move_object(chessboard, x=x_off, y=y_off, z=nominal_z)
                    # Orient the chessboard. Here we assume rotation about Z (yaw) is sufficient.
                    # If you need more rotations, adjust alpha, beta as needed.
                    orient_object(chessboard, alpha=0, beta=math.radians(30), gamma=theta)
                    
                    # Allow some time for the simulation to settle.
                    time.sleep(0.05)
                    
                    # Capture an image from the calibration camera.
                    frame = take_picture()
                    cv2.imshow("Calibration Capture", frame)
                    filename = os.path.join(output_dir, f"calib_{image_count:03d}_x{x_off}_y{y_off}_rot{rots_deg[rots.index(theta)]}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Captured image saved as {filename}")
                    image_count += 1
                    cv2.waitKey(1)
    finally:
        sim.stopSimulation()
        print("Automated calibration capture complete.")

if __name__ == "__main__":
    main()
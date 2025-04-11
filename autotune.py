import time
import math
import cv2
import numpy as np
from simple_pid import PID
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard

# Minimal DifferentialCar class (assuming the rest of the code is in place)
class DifferentialCar:
    def __init__(self, left_wheel=None, right_wheel=None):
        self.left_wheel = left_wheel or sim.getObject('/DynamicLeftJoint')
        self.right_wheel = right_wheel or sim.getObject('/DynamicRightJoint')
        self._linear_speed = 0.0
        self._angular_speed = 0.0
        self.wheelRadius = 0.027
        self.interWheelDistance = 0.119
        self._update_wheel_velocities()
    
    def _update_wheel_velocities(self):
        left_speed = (self._linear_speed - (self._angular_speed * self.interWheelDistance / 2)) / self.wheelRadius
        right_speed = (self._linear_speed + (self._angular_speed * self.interWheelDistance / 2)) / self.wheelRadius
        left_speed = float(left_speed)
        right_speed = float(right_speed)
        client.setStepping(True)
        sim.setJointTargetVelocity(self.left_wheel, left_speed)
        sim.setJointTargetVelocity(self.right_wheel, right_speed)
        client.setStepping(False)
    
    @property
    def angular_speed(self):
        return self._angular_speed
    
    @angular_speed.setter
    def angular_speed(self, value):
        self._angular_speed = value
        self._update_wheel_velocities()
    
    @property
    def linear_speed(self):
        return self._linear_speed
    
    @linear_speed.setter
    def linear_speed(self, value):
        self._linear_speed = value
        self._update_wheel_velocities()

# Global variables & setup.
pressed_keys = set()
def on_press(key):
    pressed_keys.add(key.char if hasattr(key, 'char') else str(key))
def on_release(key):
    pressed_keys.discard(key.char if hasattr(key, 'char') else str(key))
keyboard.Listener(on_press=on_press, on_release=on_release).start()

client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')
car_cam = sim.getObject('/LineTracer/visionSensor')
car = DifferentialCar()

dist_coeffs = np.zeros((5, 1), dtype=np.float32)
camera_matrix = np.array([[443.4, 0, 256],
                          [0, 443.4, 256],
                          [0, 0, 1]], dtype=np.float32)

# Time globals.
dt = 0
start = time.time()
last = start
elapsed = 0

# Control globals.
cam_yaw = 0.0
# Start with pure proportional control (simulate oscillations)
cam_yaw_pid = PID(Kp=0.5, Ki=0, Kd=0, setpoint=0.0)

def time_step():
    global dt, last, elapsed
    now = time.time()
    dt = now - last
    last = now
    elapsed = now - start

def get_image(vision_sensor_handle):
    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def find_arucos(frame):
    # Detect markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # print(f"Detected {len(corners)} markers")

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    return corners, ids

def estimate_pose(marker_corners, frame, ref_size=None, camera_matrix=None, dist_coeffs=None, marker_length=None):
    """
    Estimates pose and extracts Euler angles (yaw, pitch, roll) and distance.
    
    Two modes are available:
      1. Traditional approach:
         If calibration parameters (camera_matrix, dist_coeffs, and marker_length) are provided,
         this method uses cv2.aruco.estimatePoseSingleMarkers.
         
      2. Custom (ref_size) approach:
         If calibration parameters are not provided but a ref_size is, then default calibration 
         parameters are assumed and the distance is computed based on the marker's apparent size:
             estimated_distance = ref_size / normalized_apparent_size
         (with normalized_apparent_size = apparent_size_pixels / image_width).
         
      In addition, the function calculates the camera’s pitch and yaw relative to the marker 
      based on its position in the image.
      
      Returns a tuple: (yaw, pitch, roll, distance, cam_pitch, cam_yaw)
    """
    import math
    import numpy as np
    import cv2

    def rotationMatrixToEulerAngles(R):
        """
        Converts a rotation matrix to Euler angles (roll, pitch, yaw) using the 
        Tait–Bryan angles convention. Returns a numpy array [roll, pitch, yaw] in radians.
        """
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        return np.array([roll, pitch, yaw])

    h, w, _ = frame.shape

    # Determine which approach to use.
    use_traditional = (camera_matrix is not None and dist_coeffs is not None and marker_length is not None)
    use_custom = (not use_traditional and ref_size is not None)
    
    if not use_traditional and not use_custom:
        raise ValueError("Insufficient parameters. Provide calibration parameters (camera_matrix, dist_coeffs, marker_length) or a ref_size.")
    
    # If calibration parameters are missing, assume defaults.
    if not use_traditional:
        marker_length = 0.05  # default marker size in meters
        fx_default = fy_default = 800.0
        cx_default, cy_default = w / 2.0, h / 2.0
        camera_matrix = np.array([[fx_default, 0, cx_default],
                                  [0, fy_default, cy_default],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # Compute pose using the traditional method.
    ret_vals = cv2.aruco.estimatePoseSingleMarkers(marker_corners, marker_length, camera_matrix, dist_coeffs)
    if ret_vals is None or len(ret_vals[0]) == 0:
        return None

    # Extract rotation vector and convert it to Euler angles.
    rvec = ret_vals[0][0][0]  # rotation vector for the first detected marker
    R, _ = cv2.Rodrigues(rvec)
    euler_rad = rotationMatrixToEulerAngles(R)
    euler_deg = np.degrees(euler_rad)
    # Mapping convention:
    yaw = euler_deg[1]
    new_pitch = ((180 - euler_deg[0] + 180) % 360) - 180  # adjusted pitch
    roll = euler_deg[2]

    # Determine distance.
    if use_custom:
        # Compute apparent marker size.
        corners_array = marker_corners.reshape((4, 2))
        distances = []
        for i in range(4):
            j = (i + 1) % 4
            dx = corners_array[j][0] - corners_array[i][0]
            dy = corners_array[j][1] - corners_array[i][1]
            distances.append(math.hypot(dx, dy))
        apparent_size_pixels = max(distances)
        normalized_apparent_size = apparent_size_pixels / float(w)
        estimated_distance = ref_size / normalized_apparent_size
    else:
        # Use translation vector from the pose.
        tvec = ret_vals[1][0][0]
        estimated_distance = np.linalg.norm(tvec)

    # Calculate the camera's pitch and yaw relative to the marker using both FOV axes.
    # Extract focal lengths from the camera matrix.
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    # Compute horizontal and vertical FOV (in degrees)
    fov_x = math.degrees(2 * math.atan(w / (2 * fx)))
    fov_y = math.degrees(2 * math.atan(h / (2 * fy)))
    
    # Compute the marker center.
    marker_center = np.mean(marker_corners.reshape((4, 2)), axis=0)
    # Get principal point from the camera matrix.
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    # Pixel offsets from the principal point.
    offset_x = marker_center[0] - cx
    offset_y = marker_center[1] - cy
    # Convert pixel offsets to angular offsets: half image width corresponds to half fov_x, etc.
    cam_yaw = (offset_x / (w / 2)) * (fov_x / 2)
    cam_pitch = -(offset_y / (h / 2)) * (fov_y / 2)

    return (yaw, new_pitch, roll, estimated_distance, cam_pitch, cam_yaw)

def sense():
    global frame, cam_pitch, cam_yaw, yaw, pitch, roll, distance
    frame = get_image(car_cam)
    corners, ids = find_arucos(frame)
    if ids is not None and len(corners) > 0:
        # yaw, pitch, roll, distance = estimate_pose(corners[0], frame, ref_size=0.08203125)
        yaw, pitch, roll, distance, cam_pitch, cam_yaw = estimate_pose(corners[0], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, marker_length=0.1)
        print(f"Z:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {distance:.2f}\tCp:{cam_pitch:6.2f}\tCy:{cam_yaw:6.2f}")

    cv2.imshow('Vision Sensor Image', frame)
    cv2.setWindowProperty('Vision Sensor Image', cv2.WND_PROP_TOPMOST, 1)

def actuate():
    # For autotuning, we use auto mode ('r' key pressed)
    if 'r' in pressed_keys:
        car.angular_speed = cam_yaw_pid(cam_yaw)
    else:
        # Manual control fallback.
        max_linear_speed = 0.5
        max_angular_speed = math.radians(90)
        car.linear_speed = max_linear_speed if 'w' in pressed_keys else -max_linear_speed if 's' in pressed_keys else 0
        car.angular_speed = max_angular_speed if 'a' in pressed_keys else -max_angular_speed if 'd' in pressed_keys else 0

def autotune_pid(autotune_duration=30):
    """Run the system in auto mode (assume only P control) for a set duration,
       record cam_yaw and compute oscillation period Pu, then use Ziegler–Nichols tuning."""
    print("Starting autotune... Drive the car in auto mode (press 'r') to induce oscillations.")
    initial_time = time.time()
    time_series = []
    yaw_series = []
    while time.time() - initial_time < autotune_duration:
        time_step()
        sense()
        actuate()  # auto mode should be active (press 'r')
        # Record the current time (relative) and cam_yaw measurement.
        time_series.append(time.time() - initial_time)
        yaw_series.append(cam_yaw)
        cv2.waitKey(1)
    # Simple peak detection: find local maxima in the cam_yaw time series.
    peaks = []
    peak_times = []
    for i in range(1, len(yaw_series)-1):
        if yaw_series[i] > yaw_series[i-1] and yaw_series[i] > yaw_series[i+1]:
            peaks.append(yaw_series[i])
            peak_times.append(time_series[i])
    if len(peak_times) < 2:
        print("Not enough oscillation detected for autotuning.")
        return
    periods = [peak_times[i+1] - peak_times[i] for i in range(len(peak_times)-1)]
    Pu = sum(periods) / len(periods)
    # Ku is the current proportional gain.
    Ku = cam_yaw_pid.Kp
    # Ziegler–Nichols tuning rule for PID.
    new_Kp = 0.6 * Ku
    new_Ki = 1.2 * Ku / Pu
    new_Kd = 0.075 * Ku * Pu
    print(f"Autotune results: Ku = {Ku:.3f}, Pu = {Pu:.3f}")
    print(f"New PID constants => Kp: {new_Kp:.3f}, Ki: {new_Ki:.3f}, Kd: {new_Kd:.3f}")
    # Update the PID controller.
    cam_yaw_pid.Kp = new_Kp
    cam_yaw_pid.Ki = new_Ki
    cam_yaw_pid.Kd = new_Kd

# --- Main program ---
frame = None
try:
    sim.startSimulation()
    # Let the car drive in auto mode for 60 seconds.
    auto_duration = 60  # seconds before autotuning
    start_auto = time.time()
    while time.time() - start_auto < auto_duration:
        time_step()
        sense()
        actuate()
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # Now run autotune (for 30 seconds) and print the optimum PID gains.
    autotune_pid(30)
    # Continue running so you can see the effect.
    while sim.getSimulationState() != sim.simulation_stopped:
        time_step()
        sense()
        actuate()
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()
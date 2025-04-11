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
cal_cam = sim.getObject('/calibrationCamera')
car_cam = sim.getObject('/LineTracer/visionSensor')
sky_cam = sim.getObject('/skyCam')
aru_cam = sim.getObject('/aruCam')
car = DifferentialCar()

# Camera parameters
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
camera_matrix = np.array([[443.4, 0, 256],
                          [0, 443.4, 256],
                          [0, 0, 1]], dtype=np.float32)

def get_image(vision_sensor_handle):
    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def control():
    # If 'r' key is held down, auto control the car.
    if 'r' in pressed_keys:
        # TODO: Replace with actual automatic control logic.
        auto_linear_speed = 0.3   # e.g., maintain a constant linear speed
        auto_angular_speed = 0.0    # e.g., no turning for now
        car.linear_speed = auto_linear_speed
        car.angular_speed = auto_angular_speed
    else:
        # Manual control.
        max_linear_speed = 0.5
        max_angular_speed = math.radians(90)
        car.linear_speed = max_linear_speed if 'w' in pressed_keys else -max_linear_speed if 's' in pressed_keys else 0
        car.angular_speed = max_angular_speed if 'a' in pressed_keys else -max_angular_speed if 'd' in pressed_keys else 0

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
         If all three calibration parameters (camera_matrix, dist_coeffs, and marker_length)
         are provided, this method uses cv2.aruco.estimatePoseSingleMarkers.
         
      2. Custom (ref_size) approach:
         If the calibration parameters are not provided but a ref_size is, then default
         calibration parameters are assumed and the distance is computed based on the marker's
         apparent size:
             estimated_distance = ref_size / normalized_apparent_size
         (with normalized_apparent_size = apparent_size_pixels / image_width).
         
      In addition, the function calculates the camera’s pitch and yaw relative to the marker 
      based on its position in the image. For a camera with a 60° field-of-view (FOV),
      if the marker’s center is at the top edge, cam_pitch will be about -30°.
      
      If neither option is available, a ValueError is raised.
      
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
        raise ValueError("Insufficient parameters. Provide either calibration parameters (camera_matrix, dist_coeffs, marker_length) or a ref_size.")
    
    # If calibration parameters are missing, assume defaults.
    if not use_traditional:
        marker_length = 0.05  # default marker size in meters
        fx = fy = 800.0
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0,  0,  1]], dtype=np.float32)
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
            j = (i + 1) % 4  # wrap-around
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

    # Calculate the camera's pitch and yaw relative to the marker.
    # Assume the camera's FOV is 60° (both vertical and horizontal for a square image).
    fov = 60.0  # degrees
    # Compute the marker center.
    marker_center = np.mean(marker_corners.reshape((4, 2)), axis=0)
    # Get principal point from the camera matrix.
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    # Compute the offset from the principal point.
    offset_x = marker_center[0] - cx
    offset_y = marker_center[1] - cy
    # Convert pixel offsets to angular offsets. (Half FOV corresponds to half image width/height.)
    cam_yaw = (offset_x / (w / 2.0)) * (fov / 2.0)
    cam_pitch = -(offset_y / (h / 2.0)) * (fov / 2.0)

    return (yaw, new_pitch, roll, estimated_distance, -cam_pitch, -cam_yaw)

frame = None
try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        control()
        
        frame = get_image(car_cam)

        corners, ids = find_arucos(frame)

        if ids is not None and len(corners) > 0:
            # yaw, pitch, roll, distance = estimate_pose(corners[0], frame, ref_size=0.08203125)
            yaw, pitch, roll, distance, cam_pitch, cam_yaw = estimate_pose(corners[0], frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, marker_length=0.1)
            print(f"Z:{yaw:6.2f}\tY:{pitch:6.2f}\tX:{roll:6.2f}\tD: {distance:.2f}\tCamPitch: {cam_pitch:.2f}\tCamYaw: {cam_yaw:.2f}")

        cv2.imshow('Vision Sensor Image', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    sim.stopSimulation()
    cv2.imwrite('last_frame.png', frame)

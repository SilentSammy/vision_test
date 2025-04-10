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
aru_cam = sim.getObject('/arucoMarker/visionSensor')
car = DifferentialCar()

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

def find_arucos(frame):
    # Detect markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    print(f"Detected {len(corners)} markers")

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    return corners, ids

def estimate_pose(marker_corners, frame, ref_size=0.08203125):
    """Estimates pose and extracts Euler angles (yaw, pitch, roll) and distance.
    The returned tuple is (yaw, pitch, roll, distance).
    """

    def rotationMatrixToEulerAngles(R):
        """
        Converts a rotation matrix to Euler angles (roll, pitch, yaw) using the 
        Tait–Bryan angles convention. Returns a numpy array [roll, pitch, yaw] in radians.
        """
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2,1], R[2,2])
            pitch = math.atan2(-R[2,0], sy)
            yaw = math.atan2(R[1,0], R[0,0])
        else:
            roll = math.atan2(-R[1,2], R[1,1])
            pitch = math.atan2(-R[2,0], sy)
            yaw = 0

        return np.array([roll, pitch, yaw])

    h, w, _ = frame.shape
    marker_length = 0.05  # in meters; adjust as needed
    fx, fy = 800.0, 800.0
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    ret = cv2.aruco.estimatePoseSingleMarkers(marker_corners, marker_length, camera_matrix, dist_coeffs)
    if ret is not None and len(ret[0]) > 0:
        rvec = ret[0][0][0]  # rotation vector
        R, _ = cv2.Rodrigues(rvec)
        euler_rad = rotationMatrixToEulerAngles(R)
        euler_deg = np.degrees(euler_rad)
        # Mapping: yaw = computed yaw (euler_deg[1]),
        #          pitch = adjusted computed pitch,
        #          roll = computed roll (euler_deg[2])
        new_pitch = ((180 - euler_deg[0] + 180) % 360) - 180
        
        # Compute the apparent size from marker_corners in pixels.
        corners_array = marker_corners.reshape((4, 2))
        distances = []
        for i in range(4):
            j = (i + 1) % 4  # wrap-around index
            dx = corners_array[j][0] - corners_array[i][0]
            dy = corners_array[j][1] - corners_array[i][1]
            distances.append(math.hypot(dx, dy))
        apparent_size_pixels = max(distances)
        # Convert to normalized units wrt frame width.
        normalized_apparent_size = apparent_size_pixels / w
        # Assuming self.ref_size holds the normalized apparent size when 1 meter away:
        # Estimated distance = 1m * (ref_size / measured_size)
        estimated_distance = ref_size / normalized_apparent_size
        
        return (euler_deg[1], new_pitch, euler_deg[2], estimated_distance)
    return None

frame = None
try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        control()
        
        # frame = get_image(aru_cam)
        frame = get_image(car_cam)
        # frame = get_image(sky_cam)
        # _, frame = webcam.read()

        corners, ids = find_arucos(frame)

        if ids is not None and len(corners) > 0:
            yaw, pitch, roll, distance = estimate_pose(corners[0], frame, ref_size=0.08203125)
            print(f"Z\t{yaw:6.2f}\tY: {pitch:6.2f}\tX: {roll:6.2f}, Distance: {distance:.2f}m", end='\r')

        cv2.imshow('Vision Sensor Image', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    sim.stopSimulation()
    cv2.imwrite('last_frame.png', frame)

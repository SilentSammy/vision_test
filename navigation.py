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

def find_ellipses(frame, lower_hsv, upper_hsv):
    # Find contours that match the specified color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ellipses = []
    for cnt in contours:
        # 1. Filter out polygons with fewer than 5 points.
        if len(cnt) < 5:
            continue

        # 2. Fit an ellipse to the contour.
        ellipse = cv2.fitEllipse(cnt)
        
        # 3. Check if the ellipse has valid dimensions to avoid raising an (-215:Assertion failed) error.
        (center_x, center_y), (axis1, axis2), angle = ellipse
        if axis1 <= 0 or axis2 <= 0:  # Skip invalid ellipses
            print("Invalid ellipse dimensions:", axis1, axis2)
            continue

        # 4. Filter out small areas
        ellipse_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(ellipse_mask, ellipse, 255, -1)
        occluded_area = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=ellipse_mask))
        if occluded_area < 50:
            continue

        # 5. Filter out non-elliptical shapes
        contour_area = cv2.contourArea(cnt)
        (center_x, center_y), (axis1, axis2), angle = ellipse
        ellipse_area = math.pi * (axis1/2) * (axis2/2)
        area_ratio = contour_area / ellipse_area
        if area_ratio < 0.8 or area_ratio > 1.2:
            continue
        
        # 6. Add the ellipse to the list.
        ellipses.append(ellipse)
    
    return ellipses

def estimate_circle_pose(ellipse, frame, camera_matrix=None, real_diameter=None, ref_size=None, fov_x=None, fov_y=None):
    """
    Estimates the distance, yaw, and pitch of the camera relative to a circle
    (detected as an ellipse). Two approaches are available:
    
      Traditional approach:
         If calibration parameters (i.e. a camera_matrix) and real_diameter are provided,
         this method uses the pinhole camera model:
             estimated_distance = (real_diameter * fx) / (major_axis * corrective_factor)
         (Optionally, you may apply a corrective_factor to the measured ellipse axes.)
      
      Custom (ref_size) approach:
         If calibration parameters are not provided but a ref_size is supplied,
         then default calibration parameters are assumed to compute the normalized major axis.
         The distance is then computed as:
             estimated_distance = ref_size / normalized_major
         (where normalized_major = (corrected_major_axis) / image_width)
    
    In either case, the function also computes angular offsets (cam_pitch and cam_yaw)
    based on the ellipse center, the assumed (or provided) camera matrix, and the frame dimensions.
    
    Parameters:
      ellipse: Tuple ((center_x, center_y), (axis1, axis2), angle)
      frame: The current image frame.
      camera_matrix: (Optional) The camera calibration matrix.
      real_diameter: (Optional) The true diameter of the circle (meters) for the traditional approach.
      ref_size: (Optional) The normalized apparent size of a reference object (as used in marker pose)
                for the custom approach.
      
    Returns:
      Tuple (estimated_distance, cam_pitch, cam_yaw)
    """

    def estimate_circle_orientation(ellipse):
        """
        Given an ellipse (from cv2.fitEllipse), compute the circle's orientation.
        It first computes the tilt based on the axis ratio (minor/major = cos(tilt)).
        Then, using the ellipse's angle, it distributes this tilt into yaw and pitch components.
        Returns a tuple (yaw, pitch) in degrees (with roll assumed to be zero).
        """
        # Unpack ellipse parameters.
        # ellipse = ((center_x, center_y), (axis1, axis2), angle)
        (center_x, center_y), (axis1, axis2), angle = ellipse
        major = max(axis1, axis2)
        minor = min(axis1, axis2)
        # Avoid division by zero.
        if major == 0:
            tilt_deg = 0
        else:
            # Compute the tilt angle from the ratio.
            tilt_rad = math.acos(minor / major)
            tilt_deg = math.degrees(tilt_rad)
        
        # Convert the ellipse's angle to radians.
        angle_rad = math.radians(angle)
        
        # Distribute the tilt: if angle=0, all tilt goes to yaw; if angle=90, all goes to pitch.
        yaw   = tilt_deg * math.cos(angle_rad)
        pitch = tilt_deg * math.sin(angle_rad)
        
        return yaw, pitch, 0

    h, w, _ = frame.shape

    # Unpack ellipse. We'll use the larger axis as the apparent diameter.
    (center_x, center_y), axes, angle = ellipse
    measured_major = max(axes)
    
    # Optionally, apply a corrective factor to compensate for ellipse overestimation.
    # corrective_factor = 0.84
    corrective_factor = 1.0
    corrected_major = measured_major * corrective_factor

    # Determine which approach to use.
    # Traditional required both a camera_matrix and a real_diameter.
    use_traditional = (camera_matrix is not None and real_diameter is not None)
    use_custom = (not use_traditional and ref_size is not None)
    
    if not use_traditional and not use_custom:
        raise ValueError("Insufficient parameters. Provide either (camera_matrix and real_diameter) or ref_size.")

    if not use_traditional:
        # Custom approach: assume default calibration parameters.
        # These defaults should match those used when creating the ref_size.
        fx_default = fy_default = 800.0
        cx_default, cy_default = w / 2.0, h / 2.0
        camera_matrix = np.array([[fx_default, 0, cx_default],
                                  [0, fy_default, cy_default],
                                  [0, 0, 1]], dtype=np.float32)
        # Calculate normalized major axis: ratio of corrected_major to image width.
        normalized_major = corrected_major / float(w)
        estimated_distance = ref_size / normalized_major
    else:
        # Traditional approach using the pinhole camera model.
        fx = camera_matrix[0, 0]
        # Use the corrected major axis for distance estimation.
        estimated_distance = (real_diameter * fx) / corrected_major if corrected_major > 0 else 0.0

    # In either case, use the (assumed or provided) camera_matrix to compute angular offsets.
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Use externally provided fov values if they exist; otherwise compute.
    if fov_x is None:
        fov_x = math.degrees(2 * math.atan(w / (2 * fx)))
    if fov_y is None:
        fov_y = math.degrees(2 * math.atan(h / (2 * fy)))
    
    # Compute pixel offsets from the principal point.
    offset_x = center_x - cx
    offset_y = center_y - cy
    
    # Convert these pixel offsets to angular offsets.
    cam_yaw = -(offset_x / (w / 2)) * (fov_x / 2)
    cam_pitch = (offset_y / (h / 2)) * (fov_y / 2)
    
    yaw, pitch, roll = estimate_circle_orientation(ellipse)

    # return estimated_distance, cam_pitch, cam_yaw
    return yaw, pitch, roll, estimated_distance, cam_pitch, cam_yaw

def find_corresponding_ellipse(new_ellipse, old_ellipses, threshold=2.5, relative_to_ellipse=True):
    new_center = new_ellipse[0]
    major_axis = max(new_ellipse[1])

    # Determine the base threshold
    base_threshold = major_axis if relative_to_ellipse else 1.0

    # Sort old ellipses by distance to the new ellipse
    sorted_ellipses = sorted(
        ((old_ellipse, math.hypot(new_center[0] - old_ellipse[0][0], new_center[1] - old_ellipse[0][1]))
         for old_ellipse in old_ellipses),
        key=lambda item: item[1]
    )

    # Find the first old ellipse within the threshold
    for old_ellipse, dist in sorted_ellipses:
        if dist < threshold * base_threshold:
            return old_ellipse
    return None

def find_corresponding_disc(new_disc, prev_discs, threshold=2.5, relative_to_ellipse=True):
    prev_ellipses = [d['ellipse'] for d in prev_discs]
    corresponding_ellipse = find_corresponding_ellipse(new_disc['ellipse'], prev_ellipses, threshold, relative_to_ellipse)
    return prev_discs[prev_ellipses.index(corresponding_ellipse)] if corresponding_ellipse is not None else None

# --- Main program ---
# State 0: Stop
# State 1: Move towards green disc (green disc detected)
# State 2: Pause (yellow disc detected)
# State 3: Spin clockwise (orange disc detected)
# State 4: Spin counter-clockwise (blue disc detected)
current_time = time.time()
frame = None
state = 0
stop_until = 0.0
scan_until = 0.0
disc_colors = {
    'green': ((45, 100, 100), (75, 255, 255)),
    'orange': ((10, 100, 100), (25, 255, 255)),
    'blue': ((100, 100, 100), (140, 255, 255)),
    'yellow': ((20, 100, 100), (40, 255, 255))
}
disc_ids = {color: 0 for color in disc_colors.keys()}
discs = {color: [] for color in disc_colors.keys()}

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        # Iteration inputs
        current_time = time.time()
        frame = get_image(car_cam)

        # Find discs in the image
        new_discs = {color: [{'id': None, 'ignore': False, 'ellipse': e} for e in find_ellipses(frame, lh, uh)] for color, (lh, uh) in disc_colors.items()}
        
        # Find corresponding discs for all colors
        for color in disc_colors.keys():
            # Create a list of previous discs of this color
            prev_discs = discs[color].copy()
            for new_disc in new_discs[color]:
                corresponding_disc = find_corresponding_disc(new_disc, prev_discs, threshold=frame.shape[1]*0.2, relative_to_ellipse=False)
                if corresponding_disc is not None: # It's a previously seen disc
                    new_disc['id'] = corresponding_disc['id'] # update its id in the new list
                    discs[color][discs[color].index(corresponding_disc)]['ellipse'] = new_disc['ellipse'] # update the ellipse in the old list
                    prev_discs.remove(corresponding_disc) # remove it so we don't match it again
                else: # It's a new disc
                    new_disc['id'] = disc_ids[color]
                    disc_ids[color] += 1
                    discs[color].append(new_disc)
            # Optionally, remove discs that have left the field of view
            discs[color] = [d for d in discs[color] if d['id'] in (d['id'] for d in new_discs[color])]

        # Draw the id on the discs
        for color in disc_colors.keys():
            for new_disc in discs[color]:
                center = new_disc['ellipse'][0]
                text_color = (0, 0, 0) if new_disc['ignore'] else (255, 255, 255)
                cv2.putText(frame, str(new_disc['id']), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        # Get the most recent disc that is not ignored for each color
        yellow_disc = max((d for d in discs['yellow'] if not d['ignore']), key=lambda d: d['id'], default=None)
        green_disc = max((d for d in discs['green'] if not d['ignore']), key=lambda d: d['id'], default=None)
        orange_disc = max((d for d in discs['orange'] if not d['ignore']), key=lambda d: d['id'], default=None)
        blue_disc = max((d for d in discs['blue'] if not d['ignore']), key=lambda d: d['id'], default=None)

        # Update state
        if yellow_disc or current_time < stop_until: # if a yellow disc was seen this iteration or the timer is still running
            state = 2
        elif orange_disc or (state == 3 and current_time < scan_until and not green_disc): # if an orange disc was seen this iteration or the timer is still running and no green disc was seen
            state = 3
        elif blue_disc or (state == 4 and current_time < scan_until and not green_disc): # if a blue disc was seen this iteration or the timer is still running and no green disc was seen
            state = 4
        elif green_disc:
            state = 1
        else:
            state = 0

        # Now act based on state.
        if state == 0:
            # Decelerate smoothly
            car.slow_down(2)
            car.spin_down(2)
            
            print("State 0: Awaiting disc detection")
        elif state == 1:
            # Regular movement logic.
            _, _, _, cam_dist, _, cam_yaw = estimate_circle_pose(green_disc['ellipse'], frame, camera_matrix=camera_matrix, real_diameter=0.1)
            car.angular_speed = -heading_pid(cam_yaw)
            yaw_threshold = 10.0
            factor = max(0, 1 - (abs(cam_yaw) / yaw_threshold))
            car.linear_speed = factor * (-distance_pid(cam_dist))

            print("State 1: Moving towards green disc")
        elif state == 2:
            # Decelerate smoothly
            car.slow_down(2)
            car.spin_down(2)

            # ignore the yellow disc
            if yellow_disc:
                yellow_disc['ignore'] = True
                stop_until = current_time + 5.0
            
            print(f"State 2: Stopped for {stop_until - current_time:.2f} s")
        elif state == 3:
            # spin clockwise
            car.linear_speed = 0.0
            car.angular_speed = -math.radians(30)

            # ignore the discs and set the timer
            if orange_disc:
                orange_disc['ignore'] = True
                if green_disc:
                    green_disc['ignore'] = True
                scan_until = current_time + 9.0
            
            print(f"State 3: Scanning clockwise for {scan_until - current_time:.2f} s")
        elif state == 4:
            # spin counter-clockwise
            car.linear_speed = 0.0
            car.angular_speed = math.radians(30)

            # ignore the disc and set the timer
            if blue_disc:
                blue_disc['ignore'] = True
                if green_disc:
                    green_disc['ignore'] = True
                scan_until = current_time + 9.0
            
            print(f"State 4: Scanning counter-clockwise for {scan_until - current_time:.2f} s")

        cv2.imshow('Vision Sensor Image', frame)
        cv2.setWindowProperty('Vision Sensor Image', cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

import math
from typing import Callable
import json
import random
from dataclasses import dataclass
import numpy as np
import cv2

class VideoProvider:
    def __init__(self, vid_src=None, mirror=True):
        self.vid_src = vid_src or cv2.VideoCapture(0)
        self.mirror = mirror
        self.got_frame = False

        self.orig_frame = None
        self.rgb_frame = None
        self.hsv_frame = None
        # Store an unmirrored copy for ArUco detection

    def get_frame(self):
        """ Capture a new frame. Store both the original (unmirrored)
            and, if mirror is True, the mirrored (RGB) frame and its HSV version.
        """
        self.got_frame, raw_frame = (self.vid_src.read() if self.vid_src is not None else (False, None))
        if self.got_frame:
            self.orig_frame = raw_frame.copy()
            self.rgb_frame = cv2.flip(self.orig_frame, 1) if self.mirror else self.orig_frame
            self.hsv_frame = cv2.cvtColor(self.rgb_frame, cv2.COLOR_BGR2HSV)
        return self.got_frame

    def get_masked_frame(self, lower_hsv=(15, 150, 150), upper_hsv=(35, 255, 255)):
        masked_frame = cv2.inRange(self.hsv_frame, lower_hsv, upper_hsv)
        masked_frame = cv2.GaussianBlur(masked_frame, (5, 5), 0)
        return masked_frame

    def normalize_pos(self, pos):
        return (pos[0] / self.rgb_frame.shape[1], pos[1] / self.rgb_frame.shape[0])

    def denormalize_pos(self, norm_pos):
        return (int(norm_pos[0] * self.rgb_frame.shape[1]), int(norm_pos[1] * self.rgb_frame.shape[0]))

    def __del__(self):
        if self.vid_src is not None and self.vid_src.isOpened():
            self.vid_src.release()

class ObjectTracker:
    def __init__(self, obj_id=None, obj_desc=None, obj_color=None):
        self.id = obj_id or random.randint(0, 1000)
        self.desc = obj_desc
        self.color = obj_color
        self.last_result = None

    def analyze_image(self, vid_analyzer: VideoProvider):
        raise NotImplementedError("Subclasses should implement find_norm_image_pos")

    def load_from_dict(self, entry):
        raise NotImplementedError("Subclasses should implement load_from_dict")

    @staticmethod
    def load_from_file(filename, existing_finders=None):
        """
        Load object definitions from a JSON file and update/create finder instances.
        By default, we assume entries describe a ColorFinder.
        """
        with open(filename, "r") as f:
            data = json.load(f)
        
        existing_finders = existing_finders or []
        existing_map = {finder.id: finder for finder in existing_finders if finder.id is not None}
        
        updated_finders = []
        for entry in data:
            entry_id = entry.get("id")
            if entry_id in existing_map:
                finder = existing_map.pop(entry_id)
                finder.load_from_dict(entry)
            else:
                # Create a ColorFinder by default
                finder = ColorTracker()
                finder.load_from_dict(entry)
            updated_finders.append(finder)
        
        return updated_finders

class ColorTracker(ObjectTracker):
    def __init__(self, lower_hsv=None, upper_hsv=None, min_area=100, obj_id=None, obj_desc=None, obj_color=None):
        super().__init__(obj_id, obj_desc, obj_color)
        self.lower_hsv = lower_hsv or (15, 150, 150)
        self.upper_hsv = upper_hsv or (35, 255, 255)
        self.min_area = min_area
        if obj_color is None:
            self.color = self.get_rgb_color()

    def get_rgb_color(self):
        lower_rgb = cv2.cvtColor(np.uint8([[self.lower_hsv]]), cv2.COLOR_HSV2RGB)[0][0]
        upper_rgb = cv2.cvtColor(np.uint8([[self.upper_hsv]]), cv2.COLOR_HSV2RGB)[0][0]
        average_rgb = (lower_rgb + upper_rgb) // 2
        return tuple(average_rgb)

    @staticmethod
    def get_objects_with_bounds(masked_frame, min_area=100):
        contours, _ = cv2.findContours(masked_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours based on area.
        filtered = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
        # Sort contours by area in descending order.
        filtered.sort(key=cv2.contourArea, reverse=True)
        objects = []
        h, w = masked_frame.shape[:2]
        for contour in filtered:
            # Get an axis-aligned bounding rectangle.
            x, y, rect_w, rect_h = cv2.boundingRect(contour)
            # Compute center.
            cx = x + rect_w / 2
            cy = y + rect_h / 2
            norm_center = (cx / w, cy / h)
            # Define the four straight corners.
            norm_bounds = [
                (x / w, y / h),                           # top-left
                ((x + rect_w) / w, y / h),                # top-right
                ((x + rect_w) / w, (y + rect_h) / h),     # bottom-right
                (x / w, (y + rect_h) / h)                 # bottom-left
            ]
            objects.append((norm_center, norm_bounds))
        return objects

    def analyze_image(self, vid_analyzer: 'VideoProvider'):
        res = ObjectTrackingResult()
        if vid_analyzer.got_frame:
            masked_frame = vid_analyzer.get_masked_frame(lower_hsv=self.lower_hsv, upper_hsv=self.upper_hsv)
            # cv2.imshow("Masked Frame", masked_frame)
            objects = self.get_objects_with_bounds(masked_frame, self.min_area)
            if objects:
                norm_center, norm_bounds = objects[0]
                res.norm_image_pos = norm_center
                res.norm_image_bounds = norm_bounds
                img_pos = vid_analyzer.denormalize_pos(norm_center)
                # Instead of drawing immediately, set up a frame_drawer function.
                res.frame_drawer = lambda frame: (
                    cv2.circle(frame, img_pos, 10, (0, 255, 0), -1),
                    cv2.polylines(frame,
                                  [np.int32([vid_analyzer.denormalize_pos(pt) for pt in norm_bounds])],
                                  True, (0, 255, 255), 2),
                    cv2.putText(frame, self.desc if self.desc else "Object",
                                (img_pos[0]-50, img_pos[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 0, 0), 2)
                )
            res.rgb_frame = vid_analyzer.rgb_frame.copy()
        return res

    def load_from_dict(self, entry):
        self.lower_hsv = tuple(entry.get("lower_hsv"))
        self.upper_hsv = tuple(entry.get("upper_hsv"))
        self.id = entry.get("id")
        self.desc = entry.get("desc")
        self.min_area = entry.get("min_area", 100)
        self.color = entry.get("color", None)

class ArUcoTracker(ObjectTracker):
    def __init__(self, marker_id=None, obj_desc=None, obj_color=None):
        # Do not use obj_id for detection; use marker_id instead.
        super().__init__(obj_id=None, obj_desc=obj_desc, obj_color=obj_color)
        self.marker_id = marker_id  # Dedicated field for ArUco detection
        self.ref_size = 0.061

    def analyze_image(self, vid_analyzer: VideoProvider):
        res = ArucoTrackingResult()
        if vid_analyzer.orig_frame is None:
            return res

        # Detect marker
        gray = cv2.cvtColor(vid_analyzer.orig_frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            parameters = cv2.aruco.DetectorParameters_create()
        else:
            parameters = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            for marker_corners, detected_id in zip(corners, ids.flatten()):
                if detected_id == self.marker_id:
                    # Print apparent size of the marker
                    # h, w, _ = vid_analyzer.orig_frame.shape
                    # corners_array = marker_corners.reshape((4, 2))
                    # distances = []
                    # for i in range(4):
                    #     j = (i + 1) % 4  # wrap-around index
                    #     dx = corners_array[j][0] - corners_array[i][0]
                    #     dy = corners_array[j][1] - corners_array[i][1]
                    #     distances.append(math.hypot(dx, dy))
                    # apparent_size = max(distances)
                    # normalized_size = apparent_size / w
                    # print(f"[DEBUG] ArUco marker {detected_id} normalized apparent size: {normalized_size:.3f}")

                    # Process corners and compute center
                    norm_bounds = self._get_normalized_corners(marker_corners, vid_analyzer)
                    norm_center = self._compute_center(norm_bounds)
                    res.norm_image_pos = norm_center
                    res.norm_image_bounds = norm_bounds

                    # Estimate full pose and compute Euler angles (yaw, pitch, roll).
                    pose = self._estimate_pose(marker_corners, vid_analyzer)
                    res.euler_angles = pose if pose is not None else (0, 0, 0)
                    res.distance = pose[3] if pose is not None else 0.0

                    # Build frame-drawer function.
                    res.frame_drawer = lambda frame: self._draw_frame(
                        frame, norm_bounds, norm_center, detected_id, pose, vid_analyzer
                    )
                    break
        return res

    def _get_normalized_corners(self, marker_corners, vid_analyzer: VideoProvider):
        """Extracts and normalizes the corners from the marker detection."""
        h, w, _ = vid_analyzer.orig_frame.shape
        # Use native order (expected: [top-left, top-right, bottom-right, bottom-left])
        corners_array = marker_corners.reshape((4, 2))
        norm_bounds = [(pt[0] / w, pt[1] / h) for pt in corners_array]
        if vid_analyzer.mirror:
            norm_bounds = [(1 - nx, ny) for (nx, ny) in norm_bounds]
        return norm_bounds

    def _compute_center(self, norm_bounds):
        """Computes the normalized center given normalized corner points."""
        xs, ys = zip(*norm_bounds)
        return (sum(xs) / len(norm_bounds), sum(ys) / len(norm_bounds))

    def _estimate_pose(self, marker_corners, vid_analyzer: VideoProvider):
        """Estimates pose and extracts Euler angles (yaw, pitch, roll) and distance.
        The returned tuple is (yaw, pitch, roll, distance).
        """
        h, w, _ = vid_analyzer.orig_frame.shape
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
            estimated_distance = self.ref_size / normalized_apparent_size
            
            return (euler_deg[1], new_pitch, euler_deg[2], estimated_distance)
        return None

    def _draw_frame(self, frame, norm_bounds, norm_center, detected_id, pose, vid_analyzer: VideoProvider):
        """Draws L-shaped markers at each corner and overlays center and pose info with distance."""
        for i, pt in enumerate(norm_bounds):
            pos = vid_analyzer.denormalize_pos(pt)
            draw_corner_L(frame, pos, i, mirror=vid_analyzer.mirror, arm=10, color=(0,255,255), thickness=2)
        cv2.circle(frame, vid_analyzer.denormalize_pos(norm_center), 10, (0,255,0), -1)
        cv2.putText(frame, f"ID: {detected_id}",
                    (vid_analyzer.denormalize_pos(norm_center)[0]-50,
                    vid_analyzer.denormalize_pos(norm_center)[1]-10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        # Pad angles to 3 digits with no decimals using FONT_HERSHEY_PLAIN (monospaced style)
        yaw   = f"{int(round(pose[0])):03d}"
        pitch = f"{int(round(pose[1])):03d}"
        roll  = f"{int(round(pose[2])):03d}"
        cv2.putText(frame, f"Yaw: {yaw}  Pitch: {pitch}  Roll: {roll}",
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.putText(frame, f"Distance: {pose[3]:.2f} m",
                    (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    def load_from_dict(self, entry):
        self.marker_id = entry.get("marker_id")
        self.desc = entry.get("desc")

@dataclass
class ObjectTrackingResult:
    # frame_drawer is a callable that takes a frame (np.ndarray) and draws overlays on it.
    frame_drawer: Callable[[np.ndarray], None] = lambda frame: None
    norm_image_pos: tuple = None
    norm_image_bounds: list = None

@dataclass
class ArucoTrackingResult(ObjectTrackingResult):
    # Euler angles: (yaw, pitch, roll)
    euler_angles: tuple = None
    distance: float = None  # Distance to the marker in meters (if applicable)

# Helpers
def order_points(pts):
    """
    Orders four points in the order: top-left, top-right, bottom-right, bottom-left.
    pts should be a numpy array of shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def draw_corner_L(frame, pos, index, mirror=True, arm=10, color=(0,255,255), thickness=2):
    """
    Draws an "L" shape at the given position.
    The behavior depends on the corner index:
      - index 0 (top-left): draw right and down.
      - index 1 (top-right): draw left and down.
      - index 2 (bottom-right): draw left and up.
      - index 3 (bottom-left): draw right and up.
    """
    x, y = pos
    if index == 0 and not mirror or index == 1 and mirror:
        # Top-left: draw right and down.
        cv2.line(frame, (x, y), (x+arm, y), color, thickness)
        cv2.line(frame, (x, y), (x, y+arm), color, thickness)
    elif index == 1 and not mirror or index == 0 and mirror:
        # Top-right: draw left and down.
        cv2.line(frame, (x, y), (x-arm, y), color, thickness)
        cv2.line(frame, (x, y), (x, y+arm), color, thickness)
    elif index == 2 and not mirror or index == 3 and mirror:
        # Bottom-right: draw left and up.
        cv2.line(frame, (x, y), (x-arm, y), color, thickness)
        cv2.line(frame, (x, y), (x, y-arm), color, thickness)
    elif index == 3 and not mirror or index == 2 and mirror:
        # Bottom-left: draw right and up.
        cv2.line(frame, (x, y), (x+arm, y), color, thickness)
        cv2.line(frame, (x, y), (x, y-arm), color, thickness)

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

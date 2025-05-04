import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import json
import numpy as np
import time
import math
import cv2
import keybrd
import video_replicator

class VideoPlayer:
    def __init__(self, frame_source):
        self.frame_source = frame_source
        self.frame_count = 0
        self._frame_idx = 0.0
        self.fps = 30  # Default FPS
        self._get_frame = None
        self.last_time = None
        self.dt = 0.0
        self.setup_video_source()

    def get_frame(self, idx=None):
        if idx is None:
            idx = self.frame_idx
        return self._get_frame(idx)

    def step(self, step_size=1):
        self._frame_idx += step_size
        self._frame_idx = self._frame_idx % self.frame_count
    
    def time_step(self):
        self.dt = time.time() - self.last_time if self.last_time is not None else 0.0
        self.last_time = time.time()
        return self.dt

    def move(self, speed=1):
        self._frame_idx += speed * self.dt * self.fps
        self._frame_idx = self._frame_idx % self.frame_count

    @property
    def frame_idx(self):
        return int(self._frame_idx)

    def setup_video_source(self):
        # If frame_source is a folder, load images
        if os.path.isdir(self.frame_source):
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = sorted([
                os.path.join(self.frame_source, f) 
                for f in os.listdir(self.frame_source) 
                if f.lower().endswith(image_extensions)
            ])
            self.frame_count = len(image_files)
            print("Total frames (images):", self.frame_count)
            
            def get_frame(idx):
                idx = int(idx)
                if idx < 0 or idx >= len(image_files):
                    print("Index out of bounds:", idx)
                    return None
                frame = cv2.imread(image_files[idx])
                if frame is None:
                    print("Failed to load image", image_files[idx])
                return frame
            
            self._get_frame = get_frame
        else:
            # Assume frame_source is a video file.
            cap = cv2.VideoCapture(self.frame_source)
            if not cap.isOpened():
                print("Error opening video file:", self.frame_source)
                exit(1)
            
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total frames:", self.frame_count)
            
            def get_frame(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame", idx)
                    return None
                return frame
            
            self._get_frame = get_frame

# Helper functions
def draw_quad(frame, quad, drawOutline=True):
    # Ensure quad is a numpy array.
    quad_arr = np.array(quad) if not isinstance(quad, np.ndarray) else quad
    # Convert quad_arr to the proper type.
    quad_int = quad_arr.astype(np.int32)
    if drawOutline:
        cv2.polylines(frame, [quad_int], isClosed=True, color=(255, 255, 255), thickness=2)
    
    # Compute center of the quadrilateral.
    corners = quad_int.reshape(4, 2)
    center = np.mean(corners, axis=0).astype(np.int32)
    center = np.array(center, dtype=np.float32)
    
    # get distance between first two corners
    line_length = np.linalg.norm(corners[0] - corners[1]) * 0.25
    
    # For each corner, compute direction toward the center
    # then draw a line from the corner toward the center with the computed length.
    for idx, corner in enumerate(corners):
        corner_f = np.array(corner, dtype=np.float32)
        direction = center - corner_f
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm  # normalize
        
        endpoint = (corner_f + direction * line_length).astype(np.int32)
        # The first corner's line will be red; others will be orange.
        color = (0, 0, 255) if idx == 0 else (0, 165, 255)
        cv2.line(frame, tuple(corner), tuple(endpoint), color, 5)

# Video player functions
def setup_video_source(frame_source):
    global frame_count
    # If frame_source is a folder, load images
    if os.path.isdir(frame_source):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = sorted([
            os.path.join(frame_source, f) 
            for f in os.listdir(frame_source) 
            if f.lower().endswith(image_extensions)
        ])
        if not image_files:
            print("No images found in folder:", frame_source)
            exit(1)
        frame_count = len(image_files)
        print("Total frames (images):", frame_count)
        
        def get_frame(idx):
            idx = int(idx)
            if idx < 0 or idx >= len(image_files):
                print("Index out of bounds:", idx)
                return None
            frame = cv2.imread(image_files[idx])
            if frame is None:
                print("Failed to load image", image_files[idx])
            return frame
        
        return get_frame
    else:
        # Assume frame_source is a video file.
        cap = cv2.VideoCapture(frame_source)
        if not cap.isOpened():
            print("Error opening video file:", frame_source)
            exit(1)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames:", frame_count)
        
        def get_frame(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame", idx)
                return None
            return frame
        
        return get_frame

def get_file_data(file_path):
    # Get the file entry from the files dictionary, or add it if it doesn't exist.
    file_entry = files.get(file_path, None)

    last_modified = os.path.getmtime(file_path)
    if file_entry is None or file_entry['last_modified'] != last_modified:
        # Read the contents of the file assuming its json
        with open(file_path, 'r') as f:
            data = f.read()
        # Parse the json data
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {file_path}: {e}")
            return None

        files[file_path] = {
            'last_modified': last_modified,
            'data': data,
        }
    
    return files[file_path]['data']

def get_frame_data(file, fr_idx=None):
    data = get_file_data(file)
    if data is None:
        return None
    
    fr_idx = fr_idx or vp.frame_idx
    frame_key = str(int(fr_idx))
    if frame_key not in data:
        return None
    return data[frame_key]

# Drawing functions        
def show_ducks():
    print("ducks", end=',')
    ducks = get_frame_data('ducks.json')
    if ducks is None:
        return
    for point_id, position in ducks.items():
        x, y = position[0], position[1]
        cv2.putText(frame, str(point_id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def show_quads():
    print("quads", end=',')
    quads = get_frame_data('quads.json')
    if quads is None:
        return
    for quad in quads:
        draw_quad(frame, quad, drawOutline=True)

def show_tiles():
    print("tiles", end=',')
    tiles = get_frame_data('tiles.json')
    if tiles is None:
        return
    for tile in tiles:
        q = tile['shape']
        center = np.mean(q, axis=0).astype(np.int32)
        cv2.putText(frame, str(tile['id']), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def show_anchor():
    print("anchor", end=',')
    anchor_tile = get_frame_data('anchor_tiles.json')
    if anchor_tile is None:
        return
    q = anchor_tile['shape']
    center = np.mean(q, axis=0).astype(np.int32)
    draw_quad(frame, q)
    cv2.putText(frame, str(anchor_tile['id']), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

def sync_sim():
    import video_replicator
    anchor_tile = get_frame_data('anchor_tiles.json')
    ducks = get_frame_data('ducks.json') # Each duck is a key value pair of id and position
    ducks = [position for position in ducks.values()] # Convert ducks to a list of positions only
    if anchor_tile is None or ducks is None:
        return
    video_replicator.sync_to_video(anchor_tile, frame, ducks)
    print("sim_sync", end=',')

# Setup
os.chdir(os.path.dirname(os.path.abspath(__file__)))
re = keybrd.rising_edge # Function to check if a key is pressed once
pr = keybrd.is_pressed # Function to check if a key is pressed
files = {}  # Dictionary to store file data
frame = None  # Placeholder for the current frame

# Initialize the video player
vp = VideoPlayer(r"input.mp4")

last_time = time.time()
while True:
    # Get current frame
    vp.time_step()
    vp.move(1 if pr('d') else -1 if pr('a') else 0)  # Move forward/backward
    vp.move((1 if pr('e') else -1 if pr('q') else 0) * 10)  # Fast forward/backward
    vp.step(1 if re('w') else -1 if re('s') else 0)  # Step forward/backward
    frame = vp.get_frame()

    if frame is not None:
        print(f"Frame {vp.frame_idx}/{vp.frame_count} ", end='')
        
        # Layers
        if keybrd.is_toggled('1'):
            show_ducks()
        if keybrd.is_toggled('2'):
            show_quads()
        if keybrd.is_toggled('3'):
            show_tiles()
        if keybrd.is_toggled('4'):
            show_anchor()
        if keybrd.is_toggled('5'):
            sync_sim()
        cv2.imshow("Video", frame)
        print()
    
    if re('p'):
        # Save the current frame as an image.
        output_file = f"frame_{vp.frame_idx}.png"
        cv2.imwrite(output_file, frame)
        print(f"Saved frame {vp.frame_idx} as {output_file}")

    # Press 'Escape' to exit.
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

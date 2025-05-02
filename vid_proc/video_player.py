import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import json
import numpy as np
import time
import math
import cv2
import keybrd
from filterPoints import detect_quads

# Helper functions
def draw_quad(frame, quad, drawOutline=True):
    # Convert quad to the proper type.
    quad_int = quad.astype(np.int32)
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
    
    print(f"{file_path}", end=' ')
    return files[file_path]['data']

def get_frame_data(file, fr_idx=None):
    data = get_file_data(file)
    if data is None:
        return None
    
    fr_idx = fr_idx or frame_idx
    frame_key = str(int(fr_idx))
    if frame_key not in data:
        return None
    return data[frame_key]

# Drawing functions
def draw_corners():
    frame_data = get_frame_data('corners.json')
    if frame_data is None:
        return
    for corner in frame_data:
        x, y = corner[0], corner[1]
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

def draw_quads():
    quads = detect_quads(frame)
    for quad in quads:
        draw_quad(frame, quad, drawOutline=True)
        
def draw_points():
    frame_data = get_frame_data('ducks.json')
    if frame_data is None:
        return
    for point_id, position in frame_data.items():
        x, y = position[0], position[1]
        cv2.putText(frame, str(point_id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Setup variables
re = keybrd.rising_edge # Function to check if a key is pressed once
pr = keybrd.is_pressed # Function to check if a key is pressed
frame_count = 1  # This will be overwritten in setup_video_source
frame_idx = 0.0 # Don't ask why it's a float, it just is
fps = 30  # Default FPS
files = {}  # Dictionary to store file data
frame = None  # Placeholder for the current frame
get_frame = setup_video_source(r"input.mp4")

last_time = time.time()
while True:
    # Time step.
    dt = time.time() - last_time
    last_time = time.time()

    # Update frame index based on keyboard input.
    frame_idx += dt * fps if pr('d') else -dt * fps if pr('a') else 0       # Move forward/backward
    frame_idx += (dt * fps if pr('e') else -dt * fps if pr('q') else 0) * 10 # Fast forward/backward
    frame_idx += 1 if re('w') else -1 if re('s') else 0                     # Step forward/backward
    frame_idx = frame_idx % frame_count

    # Get the current frame by converting frame_idx to an int.
    frame = get_frame(int(frame_idx))
    if frame is not None:
        print(f"Frame {int(frame_idx)}/{frame_count} ", end='')
        
        # Layers
        if keybrd.is_toggled('1'):
            draw_corners()
        if keybrd.is_toggled('2'):
            draw_quads()
        if keybrd.is_toggled('3'):
            draw_points()
        cv2.imshow("Video", frame)
        print()
    
    # Press 'Escape' to exit.
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

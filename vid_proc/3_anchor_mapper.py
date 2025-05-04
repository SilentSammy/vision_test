import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import json
import numpy as np
import time
import math
import cv2
from pose_estimation import find_corresponding_point, rotate_quad
from itertools import count
import keybrd
import copy

os.chdir(os.path.dirname(os.path.abspath(__file__)))

tile_frames = json.load(open('tiles.json')) # It's a dictionary of frame_idx -> tiles, each tile is a dict with 'id' and 'shape' keys
mapping = json.load(open('anchor_map.json')) # It's a dictionary of frame_idx -> tile to anchor mapping, each mapping is a dict with 'tile_id' and 'anchor_id' keys

anchor_frames = {} # It's a dictionary of frame_idx -> anchor, which is a dict with 'id' and 'shape' keys
current_map = None
for frame_idx, tiles in tile_frames.items():
    # Check if the current frame has a mapping
    if frame_idx in mapping:
        current_map = mapping[frame_idx]
    
    if current_map is None:
        continue

    # Get mapping instructions
    tile_id = current_map['tile_id']
    anchor_id = current_map['anchor_id']
    rotate_count = current_map.get('rot_cnt', 0)

    # Find the corresponding tile in the current frame
    tile = next((t for t in tiles if t['id'] == tile_id), None)
    if tile is None:
        continue
    
    # Rotate quad if needed
    quad = tile['shape']
    quad = rotate_quad(quad, rotate_count).tolist()

    # Set this frame's anchor
    anchor = {
        'id': anchor_id,
        'shape': quad
    }
    anchor_frames[frame_idx] = anchor

# Export the anchors to a file
with open('anchor_tiles.json', 'w') as f:
    json.dump(anchor_frames, f, indent=4)
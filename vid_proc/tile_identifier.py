import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import json
import numpy as np
import time
import math
import cv2
from pose_estimation import find_corresponding_point
from itertools import count
import keybrd
import copy

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def assign_tracked_ids(new_objs, objs, id_gen, pos_get=None, threshold_px=50, persist=False):
    pos_get = pos_get or (lambda obj: obj['shape'][0])

    # Iterate over the new objects and assign existing or new IDs
    prev_objs = objs.copy()
    for new_obj in new_objs:
        # Find the corresponding object in the previous list
        new_pos = pos_get(new_obj)
        old_poss = [pos_get(o) for o in prev_objs]
        corresponding_point = find_corresponding_point(new_pos, old_poss, threshold=threshold_px)
        if corresponding_point is not None: # It's a previously seen object
            corresponding_obj = prev_objs[old_poss.index(corresponding_point)]
            new_obj['id'] = corresponding_obj['id']
            corresponding_obj['shape'] = new_obj['shape']
            prev_objs.remove(corresponding_obj) # remove it so we don't match it again
        else: # It's a new object
            new_obj['id'] = id_gen()
            objs.append(new_obj)
    # Optionally, remove objects that have left the field of view
    if not persist:
        objs = [o for o in objs if o['id'] in (o['id'] for o in new_objs)]
    return objs

quad_frames = json.load(open('quads.json')) # It's a dictionary of frame_idx -> quads, each quad is a list of 4 points
tile_frames = {} # It's a dictionary of frame_idx -> tiles
id_gen = count(0)
tiles = [] # List of tiles, each tile is a dict with 'id' and 'shape' keys
for frame_idx, quads in quad_frames.items():
    # Each quad is a list of 4 points
    new_tiles = [{'id': None, 'shape': q} for q in quads]
    print(len(new_tiles), "tiles found")

    # Assign IDs to the tiles
    tiles = assign_tracked_ids(new_tiles, tiles, lambda: next(id_gen))

    tile_frames[frame_idx] = copy.deepcopy(tiles)

# Save the tiles to a file
with open('tiles.json', 'w') as f:
    json.dump(tile_frames, f, indent=4)
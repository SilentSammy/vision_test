import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import math
import itertools
import json
from scipy.spatial import Delaunay, ConvexHull
import numpy as np
from pose_estimation import order_points

# Load corners from corners.json
input_path = 'corners.json'
with open(input_path, 'r') as f:
    corners = json.load(f)

def find_quads(points, angle_tol=30, aspect_tol=(0.5, 2.0)):
    """
    points: list of [x,y] or (N×2) array
    angle_tol: max deviation (in °) from 90° allowed at each corner
    aspect_tol: (min, max) width/height ratio of the quad's bounding box
    returns: list of quads; each is a 4×2 array of points in CCW order
    """
    pts = np.asarray(points)
    tri = Delaunay(pts)
    quads = []

    for t_idx, simplex in enumerate(tri.simplices):
        for nbr in tri.neighbors[t_idx]:
            # only consider each adjacent pair once
            if nbr <= t_idx or nbr == -1:
                continue

            # merge the two triangles’ vertices
            quad_idx = np.unique(np.concatenate([simplex, tri.simplices[nbr]]))
            if quad_idx.size != 4:
                print(f"Filtered out: Not 4 unique vertices, got {quad_idx.size}")
                continue

            quad = pts[quad_idx]
            # quick convexity check
            hull = ConvexHull(quad)
            if hull.vertices.size != 4:
                print("Filtered out: Not a convex quadrilateral")
                continue

            # order points CCW around centroid
            center = quad.mean(axis=0)
            angs = np.arctan2(quad[:,1] - center[1], quad[:,0] - center[0])
            quad_ccw = quad[np.argsort(angs)]

            # check near-right angles
            if not _check_angles(quad_ccw, angle_tol):
                print(f"Filtered out: Angles not within tolerance of {angle_tol}°")
                continue

            # check bounding-box aspect ratio
            minx, miny = quad_ccw.min(axis=0)
            maxx, maxy = quad_ccw.max(axis=0)
            ar = (maxx - minx) / (maxy - miny + 1e-8)
            if not (aspect_tol[0] <= ar <= aspect_tol[1]):
                print(f"Filtered out: Aspect ratio {ar:.2f} not in range {aspect_tol}")
                continue

            quads.append(quad_ccw)

    return quads

def _check_angles(quad, tol):
    """Return True if all 4 interior angles of CCW‐ordered quad are within tol of 90°."""
    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    for i in range(4):
        prev_pt = quad[(i - 1) % 4]
        curr_pt = quad[i]
        next_pt = quad[(i + 1) % 4]
        if abs(angle(prev_pt, curr_pt, next_pt) - 90) > tol:
            return False

    return True

def convert_to_serializable(data):
    """
    Recursively converts NumPy arrays in the data structure to lists
    to make it JSON serializable.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    else:
        return data

# Group points into quads
quads = {}
for frame, points in corners.items():
    quads[frame] = find_quads(points)

# Order points in each quad
for frame, quad_list in quads.items():
    for i, quad in enumerate(quad_list):
        ordered_quad = order_points(quad)
        quads[frame][i] = ordered_quad

# Convert quads to a JSON-serializable format
serializable_quads = convert_to_serializable(quads)

# Export to quads.json
output_path = 'quads.json'
with open(output_path, 'w') as f:
    json.dump(serializable_quads, f, indent=4)
print(f"Quad data saved to {output_path}")
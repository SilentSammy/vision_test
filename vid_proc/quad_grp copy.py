import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import math
import itertools
import json
from scipy.spatial import Delaunay, ConvexHull
import numpy as np
from pose_estimation import order_points
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# Load corners from corners.json
input_path = 'corners.json'
with open(input_path, 'r') as f:
    corners = json.load(f)

def find_quads(points, eps_ratio=0.4):
    """
    points:   list of [x,y] or (N,2) array of your corner coordinates
    eps_ratio:  fraction of the median spacing to use as DBSCAN eps

    returns: list of quads; each quad is a 4×2 array
             in the order:
               [ (r,c), (r,c+1), (r+1,c+1), (r+1,c) ]
    """
    pts = np.asarray(points, dtype=float)

    # 1) find the two main axes of the grid
    pca = PCA(n_components=2).fit(pts)
    axis_u, axis_v = pca.components_

    # 2) project onto each axis
    proj_u = pts.dot(axis_u)
    proj_v = pts.dot(axis_v)

    # estimate a good eps by the median gap between sorted projections
    def estimate_eps(proj):
        diffs = np.diff(np.sort(proj))
        return np.median(diffs) * eps_ratio

    eps_u = estimate_eps(proj_u)
    eps_v = estimate_eps(proj_v)

    # cluster each 1D projection into lines
    clu_u = DBSCAN(eps=eps_u, min_samples=1).fit_predict(proj_u.reshape(-1, 1))
    clu_v = DBSCAN(eps=eps_v, min_samples=1).fit_predict(proj_v.reshape(-1, 1))

    # remap arbitrary labels  →  0…K−1 sorted by line position
    def remap(labels, proj):
        unique = sorted(set(labels), key=lambda lab: np.mean(proj[labels == lab]))
        return {lab: idx for idx, lab in enumerate(unique)}

    map_u = remap(clu_u, proj_u)
    map_v = remap(clu_v, proj_v)
    cols = np.array([map_u[l] for l in clu_u])
    rows = np.array([map_v[l] for l in clu_v])

    # 3) pick one point per (row,col) cell (discard duplicates, if any)
    cell = {}
    for pt, r, c in zip(pts, rows, cols):
        cell[(r, c)] = pt

    # assemble quads for every 1×1 block
    max_r, max_c = rows.max(), cols.max()
    quads = []
    for r in range(max_r):
        for c in range(max_c):
            corners = [(r, c), (r, c+1), (r+1, c+1), (r+1, c)]
            if all(k in cell for k in corners):
                quad = np.vstack([cell[k] for k in corners])
                quads.append(quad)

    return quads

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
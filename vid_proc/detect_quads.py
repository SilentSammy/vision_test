import cv2
import numpy as np
import json
from sklearn.cluster import KMeans, DBSCAN
import math
import os

VIDEO_PATH    = 'input.mp4'
START_FRAME   = 0
FRAME_STEP    = 1
OUTPUT_JSON   = 'quads.json'

CANNY1        = 50
CANNY2        = 150
HOUGH_THRESH  = 80
MIN_LINE_LEN  = 30
MAX_LINE_GAP  = 5
DBSCAN_DIST   = 40

# --- OTHERS ---
def filter_points(pts):
    if not pts: return []
    arr = np.array(pts)
    lbl = DBSCAN(eps=DBSCAN_DIST, min_samples=1).fit_predict(arr)
    uniq=[]
    for l in np.unique(lbl):
        grp=arr[lbl==l]; c=grp.mean(axis=0)
        uniq.append((float(c[0]), float(c[1])))
    uniq = sorted(uniq, key=lambda p:(p[1],p[0]))
    out=[]
    for x,y in uniq:
        if all(abs(x-x2)>=DBSCAN_DIST or abs(y-y2)>=DBSCAN_DIST for x2,y2 in out):
            out.append((x,y))
    return out

# --- HELPERS ---
def segment_angle(seg):
    """Return the angle (radians) of the segment (between −π and π)."""
    x1, y1, x2, y2 = seg
    return math.atan2(y2 - y1, x2 - x1)

def merge_two_segments(seg1, seg2):
    """
    Merge two segments by projecting all endpoints onto the line of seg1
    (assuming segments are collinear enough) and taking the min/max projections.
    Returns a new segment defined by these two extreme points.
    """
    x1, y1, x2, y2 = seg1
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return seg1
    # Use seg1's first endpoint as reference.
    ux, uy = dx / norm, dy / norm
    ref = (x1, y1)
    
    def proj(pt):
        return (pt[0] - ref[0]) * ux + (pt[1] - ref[1]) * uy

    pts = [(seg1[0], seg1[1]), (seg1[2], seg1[3]),
           (seg2[0], seg2[1]), (seg2[2], seg2[3])]
    projections = [proj(pt) for pt in pts]
    min_proj = min(projections)
    max_proj = max(projections)
    new_pt1 = (ref[0] + min_proj * ux, ref[1] + min_proj * uy)
    new_pt2 = (ref[0] + max_proj * ux, ref[1] + max_proj * uy)
    return (new_pt1[0], new_pt1[1], new_pt2[0], new_pt2[1])

def point_line_distance(pt, seg):
    """
    Compute the perpendicular distance from a point pt to the infinite line
    defined by segment seg.
    """
    x1, y1, x2, y2 = seg
    x0, y0 = pt
    den = math.hypot(x2 - x1, y2 - y1)
    if den < 1e-6:
        return float('inf')
    num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    return num / den

def intersect(l1, l2):
    x1,y1,x2,y2 = l1; x3,y3,x4,y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)

# --- STEPS ---
def detect_segments(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    segs = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                           minLineLength=MIN_LINE_LEN,
                           maxLineGap=MAX_LINE_GAP)
    return [] if segs is None else [tuple(s) for s in segs[:,0]]

def merge_segments(segs):
    """
    Merge segments that are nearly parallel and nearly collinear.
    
    Two segments are merged if:
     - Their angles differ by less than ANGLE_THRESH (in radians), and
     - At least one endpoint of one segment is within DIST_THRESH pixels 
       of the infinite line defined by the other.
       
    Returns a list of merged segments.
    """
    ANGLE_THRESH = math.radians(5)  # 5 degrees threshold
    DIST_THRESH = 10  # pixel distance threshold
    merged = []
    
    for seg in segs:
        merged_flag = False
        for i in range(len(merged)):
            m = merged[i]
            # Check angle difference.
            angle_diff = abs(segment_angle(seg) - segment_angle(m))
            if angle_diff > ANGLE_THRESH:
                continue
            # Check if at least one endpoint of seg is close to m's line.
            if (point_line_distance((seg[0], seg[1]), m) < DIST_THRESH or 
                point_line_distance((seg[2], seg[3]), m) < DIST_THRESH):
                # Merge the two segments.
                new_seg = merge_two_segments(m, seg)
                merged[i] = new_seg
                merged_flag = True
                break
        if not merged_flag:
            merged.append(seg)
    return merged

def extend_segments(segs, frame_shape, thres=0.5):
    """
    Given a list of segments (each a tuple (x1, y1, x2, y2)) and the frame shape,
    extend each segment (if its original length is at least 0.5 * min(frame_width, frame_height))
    so that it reaches the borders of the frame. Segments shorter than this threshold are removed.

    Parameters:
      segs: list of tuples (x1, y1, x2, y2)
      frame_shape: tuple (height, width, channels)

    Returns:
      A list of extended segments (each a tuple (x1, y1, x2, y2)).
    """
    height, width = frame_shape[:2]
    threshold = thres * min(width, height)
    extended = []
    
    for seg in segs:
        x1, y1, x2, y2 = seg
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < threshold:
            # Skip segments that are too short.
            continue
        
        # Parameterize line: P(t) = (x1, y1) + t*(dx, dy)
        intersections = []
        
        # Left border: x = 0
        if dx != 0:
            t = (0 - x1) / dx
            y_int = y1 + t * dy
            if 0 <= y_int <= height:
                intersections.append((0, y_int, t))
        # Right border: x = width
        if dx != 0:
            t = (width - x1) / dx
            y_int = y1 + t * dy
            if 0 <= y_int <= height:
                intersections.append((width, y_int, t))
        # Top border: y = 0
        if dy != 0:
            t = (0 - y1) / dy
            x_int = x1 + t * dx
            if 0 <= x_int <= width:
                intersections.append((x_int, 0, t))
        # Bottom border: y = height
        if dy != 0:
            t = (height - y1) / dy
            x_int = x1 + t * dx
            if 0 <= x_int <= width:
                intersections.append((x_int, height, t))
        
        if len(intersections) < 2:
            # cannot extend if fewer than two valid intersections
            continue
        
        # Choose the two extreme intersection points based on t values.
        intersections.sort(key=lambda x: x[2])
        pt_start = intersections[0][:2]
        pt_end = intersections[-1][:2]
        
        extended.append((pt_start[0], pt_start[1], pt_end[0], pt_end[1]))
    
    return extended

def cluster_lines_hv(segs):
    dirs, lines = [], []
    for x1,y1,x2,y2 in segs:
        dx,dy = x2-x1, y2-y1
        n = np.hypot(dx,dy)
        if n<1e-6: continue
        dirs.append([dx/n, dy/n]); lines.append((x1,y1,x2,y2))
    if not dirs: return [], []
    labels = KMeans(n_clusters=2, random_state=0).fit(dirs).labels_
    avg_dx = [np.mean([abs(dirs[i][0]) for i in range(len(dirs)) if labels[i]==c]) for c in (0,1)]
    hl = int(np.argmax(avg_dx))
    horiz = [lines[i] for i in range(len(lines)) if labels[i]==hl]
    vert  = [lines[i] for i in range(len(lines)) if labels[i]!=hl]

    # Compute mean directions for each cluster.
    mean_h = np.mean([dirs[i] for i in range(len(dirs)) if labels[i]==hl], axis=0)
    mean_v = np.mean([dirs[i] for i in range(len(dirs)) if labels[i]!=hl], axis=0)

    # Set an angle threshold (10 degrees)
    angle_thresh = math.radians(20)

    # Build dictionaries mapping original indices to the angle difference from the mean.
    angles_h_dict = {}
    angles_v_dict = {}
    for i, d in enumerate(dirs):
        if labels[i] == hl:
            diff = math.atan2(d[1], d[0]) - math.atan2(mean_h[1], mean_h[0])
            # Normalize to [-pi,pi]
            diff = (diff + math.pi) % (2*math.pi) - math.pi
            angles_h_dict[i] = diff
        else:
            diff = math.atan2(d[1], d[0]) - math.atan2(mean_v[1], mean_v[0])
            diff = (diff + math.pi) % (2*math.pi) - math.pi
            angles_v_dict[i] = diff

    # Now filter out segments whose angle difference exceeds the threshold.
    horiz = [lines[i] for i in range(len(lines)) 
            if labels[i]==hl and abs(angles_h_dict[i]) < angle_thresh]
    vert  = [lines[i] for i in range(len(lines)) 
            if labels[i]!=hl and abs(angles_v_dict[i]) < angle_thresh]

    return horiz, vert

def get_intersections(horiz, vert, frame_shape):
    h,w = frame_shape
    pts=[]
    intersections = []
    for hl in horiz:
        for vl in vert:
            p = intersect(hl,vl)
            if p and 0<=p[0]<w and 0<=p[1]<h:
                pts.append(p)
                intersections.append((hl,vl))
    return pts, intersections

def get_quads(pts, intersections):
    quads = []
    for i in range(len(pts)):
        # Get the first point, and its corresponding segments
        top_left = pts[i]
        hor = intersections[i][0]
        ver = intersections[i][1]

        # Get all points along hor
        hor_pts_idx = [i for i in range(len(pts)) if intersections[i][0] == hor]
        hor_pts = [pts[i] for i in hor_pts_idx]
        # Only keep the points that are to the right of top_left
        hor_pts_idx = [i for i in hor_pts_idx if pts[i][0] > top_left[0]]
        hor_pts = [pts[i] for i in hor_pts_idx]
        # Choose the closest point to top_left
        if not hor_pts:
            continue
        top_right_idx = hor_pts_idx[0]
        top_right = pts[top_right_idx]

        # Get all points along ver
        ver_pts_idx = [i for i in range(len(pts)) if intersections[i][1] == ver]
        ver_pts = [pts[i] for i in ver_pts_idx]
        # Only keep the points that are below top_left
        ver_pts_idx = [i for i in ver_pts_idx if pts[i][1] > top_left[1]]
        ver_pts = [pts[i] for i in ver_pts_idx]
        # Choose the closest point to top_left
        if not ver_pts:
            continue
        bottom_left_idx = ver_pts_idx[0]
        bottom_left = pts[bottom_left_idx]

        # Get the vertical segment of top_right
        bottom_right_vert = intersections[top_right_idx][1]
        # Get the horizontal segment of bottom_left
        bottom_right_hor = intersections[bottom_left_idx][0]
        # Get the point of intersection of bottom_right_vert and bottom_right_hor
        bottom_right = intersect(bottom_right_vert, bottom_right_hor)
        if not bottom_right:
            continue
        quad = np.array([top_left, top_right, bottom_right, bottom_left], dtype=int)
        quad = quad.tolist()
        quads.append(quad)
    return quads

# --- Full detection for one frame ---
def detect_quads(frame):
    segs = detect_segments(frame)
    segs = merge_segments(segs)
    segs = extend_segments(segs, frame.shape, 0.25)
    segs = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in segs]

    # Cluster lines into horizontal and vertical groups
    horiz, vert = cluster_lines_hv(segs)
    horiz = sorted(horiz, key=lambda s: (s[1], s[0]))   # Sort horiz from top to bottom
    vert = sorted(vert, key=lambda s: (s[0], s[1]))     # Sort vert from left to right

    # Get intersections of horizontal and vertical lines
    pts, intersections = get_intersections(horiz, vert, frame.shape[:2])
    
    # Group points into quads
    quads = get_quads(pts, intersections)

    return quads

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = START_FRAME
    frame_data = {}
    while frame_idx < 100:
        # Grab the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break

        # Get the quads
        quads = detect_quads(frame)
        frame_data[frame_idx] = quads

        # Next frame
        frame_idx += 1
        frame_idx = frame_idx

    # Save JSON
    with open(OUTPUT_JSON,'w') as f:
        json.dump(frame_data, f, indent=2)
    print(f'Coordinates saved in JSON: {OUTPUT_JSON}')

if __name__=='__main__':
    main()

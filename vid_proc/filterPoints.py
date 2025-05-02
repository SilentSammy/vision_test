import cv2
import numpy as np
import json
from sklearn.cluster import KMeans, DBSCAN

VIDEO_PATH    = './input.mp4'
START_FRAME   = 0
FRAME_STEP    = 1
OUTPUT_JSON   = './coordinates.json'

CANNY1        = 50
CANNY2        = 150
HOUGH_THRESH  = 80
MIN_LINE_LEN  = 30
MAX_LINE_GAP  = 5
DBSCAN_DIST   = 40

def intersect(l1, l2):
    x1,y1,x2,y2 = l1; x3,y3,x4,y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)

def detect_segments(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    segs = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                           minLineLength=MIN_LINE_LEN,
                           maxLineGap=MAX_LINE_GAP)
    return [] if segs is None else [tuple(s) for s in segs[:,0]]

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
    return horiz, vert

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

# --- Full detection for one frame ---
def detect_corners(frame):
    segs = detect_segments(frame)
    horiz, vert = cluster_lines_hv(segs)
    h,w = frame.shape[:2]
    pts=[]
    for hl in horiz:
        for vl in vert:
            p = intersect(hl,vl)
            if p and 0<=p[0]<w and 0<=p[1]<h:
                pts.append(p)
    return filter_points(pts)

# ————————————————————————————————
# helper: convert a segment (x1,y1,x2,y2) to (rho,theta)
def segment_to_rho_theta(seg):
    x1,y1,x2,y2 = seg
    dx, dy = x2-x1, y2-y1
    norm = np.hypot(dx,dy)
    if norm < 1e-6:
        return None
    # normal vector (nx,ny) such that nx*x + ny*y = rho
    nx, ny =  dy/norm, -dx/norm
    theta = np.arctan2(ny, nx)
    # force theta ∈ [0, π)
    if theta < 0:
        theta += np.pi
        rho = - (nx*x1 + ny*y1)
    else:
        rho =   nx*x1 + ny*y1
    return (rho, theta)

# ————————————————————————————————
# helper: cluster a list of (rho,theta) by rho → one line per cluster
def cluster_lines_by_rho(lines, eps):
    if not lines:
        return []
    rhos = np.array([l[0] for l in lines]).reshape(-1,1)
    thetas = np.array([l[1] for l in lines])
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(rhos)
    clustered = []
    for lab in np.unique(labels):
        idx = np.where(labels==lab)[0]
        clustered.append((
            float(rhos[idx].mean()), 
            float(thetas[idx].mean())
        ))
    return clustered

# ————————————————————————————————
# helper: intersect two lines in (rho,θ) form
def intersect_lines(l1, l2):
    rho1, th1 = l1
    rho2, th2 = l2
    A = np.array([[np.cos(th1), np.sin(th1)],
                  [np.cos(th2), np.sin(th2)]])
    b = np.array([rho1, rho2])
    return np.linalg.solve(A, b)

# ————————————————————————————————
def detect_quads(frame,
                 canny1=50, canny2=150,
                 hough_thresh=80,
                 min_line_len=30, max_line_gap=5,
                 dbscan_dist=30):
    # 1) find segments & split into horiz/vert
    segs = detect_segments(frame)
    horiz_segs, vert_segs = cluster_lines_hv(segs)

    # 2) convert to infinite lines
    lines_h = [segment_to_rho_theta(s) for s in horiz_segs]
    lines_v = [segment_to_rho_theta(s) for s in vert_segs]
    lines_h = [l for l in lines_h if l is not None]
    lines_v = [l for l in lines_v if l is not None]

    # 3) cluster by rho
    lines_h = cluster_lines_by_rho(lines_h, eps=dbscan_dist)
    lines_v = cluster_lines_by_rho(lines_v, eps=dbscan_dist)

    # 4) sort: horizontals by y = rho·sinθ, verticals by x = rho·cosθ
    lines_h.sort(key=lambda lt: lt[0]*np.sin(lt[1]))
    lines_v.sort(key=lambda lt: lt[0]*np.cos(lt[1]))

    # 5) intersect adjacent pairs → quads
    quads = []
    for i in range(len(lines_h)-1):
        for j in range(len(lines_v)-1):
            p1 = intersect_lines(lines_h[i],   lines_v[j])
            p2 = intersect_lines(lines_h[i],   lines_v[j+1])
            p3 = intersect_lines(lines_h[i+1], lines_v[j+1])
            p4 = intersect_lines(lines_h[i+1], lines_v[j])
            quad = np.vstack([p1,p2,p3,p4])
            quads.append(quad)

    return quads

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = START_FRAME
    coords = {}
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break
        pts = detect_corners(frame)
        coords[frame_idx] = [(float(x), float(y)) for x,y in pts]
        # display
        disp = frame.copy()
        for x,y in pts:
            cv2.circle(disp,(int(x),int(y)),5,(0,255,0),-1)
        cv2.putText(disp, f'Frame {frame_idx}', (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.imshow('Corner Detection', disp)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        frame_idx += int(cap.get(cv2.CAP_PROP_POS_MSEC) or 1)
        frame_idx = frame_idx
    cap.release()
    cv2.destroyAllWindows()
    # Save JSON
    with open(OUTPUT_JSON,'w') as f:
        json.dump(coords, f, indent=2)
    print(f'Coordinates saved in JSON: {OUTPUT_JSON}')

if __name__=='__main__':
    main()

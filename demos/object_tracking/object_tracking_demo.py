import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import cv2
from pose_estimation import find_ellipses, find_corresponding_point
from sim_tools import sim, get_image
from itertools import count

demo_cam = sim.getObject('/Demo/visionSensor')

color = ((45, 100, 100), (75, 255, 255))
next_id = count(0)
objs = []

def assign_tracked_ids(new_objs, objs, id_gen, pos_get=None, threshold_px=100, persist=False):
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

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        frame = get_image(demo_cam)

        # Find objects in the image (each object is dict of an id, and a shape in OpenCV format)
        new_objs = [{'id': None, 'shape': e} for e in find_ellipses(frame, lower_hsv=color[0], upper_hsv=color[1])]
        print(len(new_objs), "objects found")
        
        # Assign IDs to the objects
        objs = assign_tracked_ids(new_objs, objs, lambda: next(next_id), persist=True)
        
        # Draw the id on the objects
        for obj in objs:
            center = obj['shape'][0]
            cv2.putText(frame, str(obj['id']), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        cv2.namedWindow("Vision Sensor", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Vision Sensor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

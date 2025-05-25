import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import cv2
from pose_estimation import find_ellipses, find_corresponding_point
from sim_tools import sim, get_image
from itertools import count
import time

def assign_tracked_ids(new_objs, tracked_objs, id_gen, get_id, set_id, get_pos, upd_obj, threshold_px=100, persist=False):
    """
    Updates a collection of tracked objects by matching them with a new set of detected objects.
    This function is agnostic to the underlying object structure and works with any type, as long
    as the appropriate accessor, mutator, and updater callables are provided.

    For each object in new_objs, the function:
      - Extracts a representative position using get_pos.
      - Compares this position against each object in tracked_objs (via get_pos) using the 
        find_corresponding_point utility and a threshold (threshold_px) to determine if the 
        object has been seen before.
      - If a match is found:
          • The existing object's ID (obtained via get_id) is assigned to the new object using set_id.
          • The existing object's data is updated with that from the new object using upd_obj.
          • The matched object is then removed from further consideration.
      - If no match is found:
          • A new unique ID is generated (using id_gen), assigned to the new object using set_id,
            and the new object is added to tracked_objs.
    
    Optionally, if persist is False, objects in tracked_objs that are not present in new_objs are 
    considered "lost" and removed from tracked_objs; regardless, they are returned in a separate lost_objs list.

    Parameters:
      new_objs (list): Collection of newly detected objects.
      tracked_objs (list): Collection of objects currently being tracked.
      id_gen (callable): Function to generate new unique IDs.
      get_id (callable): Function to extract an object's unique identifier.
      set_id (callable): Function to set an object's unique identifier.
      get_pos (callable): Function that returns a representative position (e.g., a point) for matching.
      upd_obj (callable): Function that updates an existing object's data with that of a new detection.
      threshold_px (int, optional): Maximum pixel distance to consider two objects as matching (default is 100).
      persist (bool, optional): If False, objects not matched in new_objs will be removed from tracked_objs (default is False).

    Returns:
      tuple: (tracked_objs, lost_objs)
             tracked_objs: The updated collection of tracked objects.
             lost_objs: Objects that were not matched in new_objs (i.e. "lost" objects).
    """
    # Iterate over the new objects and assign existing or new IDs
    prev_objs = tracked_objs.copy()
    for new_obj in new_objs:
        # Attempt to find a corresponding point in the previous list
        new_pos = get_pos(new_obj)
        old_poss = [get_pos(o) for o in prev_objs]
        corresponding_point = find_corresponding_point(new_pos, old_poss, threshold=threshold_px)

        # It's a previously seen object
        if corresponding_point is not None:
            # Use the corresponding point's index to find the corresponding object
            old_idx = old_poss.index(corresponding_point)
            corresponding_obj = prev_objs[old_idx]

            # Assign the ID from the old object to the new object
            set_id(new_obj, get_id(corresponding_obj))

            # Update the old object with the new object's data
            upd_obj(corresponding_obj, new_obj)

            # Remove the matched object from the previous list so it won't be matched again
            del prev_objs[old_idx]
        else:  # It's a new object
            # Assign a new ID to the new object
            set_id(new_obj, id_gen())
            # Add the new object to the list of tracked objects
            tracked_objs.append(new_obj)
    
    # Optionally, remove objects that have left the field of view
    lost_objs = [o for o in tracked_objs if get_id(o) not in (get_id(o) for o in new_objs)]
    if not persist:
        tracked_objs = [o for o in tracked_objs if o not in lost_objs]
    return tracked_objs, lost_objs

def clear_lost_objects(tracked_objs, lost_objs, lost_timeout, is_lost, get_lost_time, set_lost_time, refind, get_id=None): # get_lost_time, set_lost_time, refind_obj
    # Remove the lost key from objects that are no longer lost
    refound_objs = [o for o in tracked_objs if o not in lost_objs and is_lost(o)]
    for obj in refound_objs:
        if get_id is not None:
            print(f"Object {get_id(obj)} refound after being lost for {time.time() - get_lost_time(obj)} seconds")
        refind(obj)

    # For each lost object, set a lost_time key if it doesn't have one
    newly_lost_objs = [o for o in lost_objs if not is_lost(o)]
    for obj in newly_lost_objs:
        set_lost_time(obj, time.time())

    # For each lost object, if it has a lost_time key, check if it's been lost for more than n seconds
    for obj in lost_objs:
        if is_lost(obj):
            if time.time() - get_lost_time(obj) > lost_timeout:
                tracked_objs.remove(obj)
                if get_id is not None:
                    print(f"Object {get_id(obj)} removed after being lost for {lost_timeout} seconds")
    
demo_cam = sim.getObject('/Demo/visionSensor')
color = ((45, 100, 100), (75, 255, 255))
next_id = count(0)
tracked_objs = []
lost_timeout = 2

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        frame = get_image(demo_cam)

        # Find objects in the image (each object is dict of an id, and a shape in OpenCV format)
        new_objs = [{'id': None, 'shape': e} for e in find_ellipses(frame, lower_hsv=color[0], upper_hsv=color[1])]
        print(len(new_objs), "objects found")
        
        # Assign IDs to the objects
        tracked_objs, lost_objs = assign_tracked_ids(
            new_objs=new_objs,
            tracked_objs=tracked_objs,
            id_gen=lambda: next(next_id),
            get_id=lambda obj: obj['id'],
            set_id=lambda obj, id: obj.update({'id': id}),
            get_pos=lambda obj: obj['shape'][0],
            upd_obj=lambda old_obj, new_obj: old_obj.update(new_obj),
            threshold_px=150,
            persist=True
        )
        
        # Clear lost objects
        clear_lost_objects(
            tracked_objs=tracked_objs,
            lost_objs=lost_objs,
            lost_timeout=lost_timeout,
            is_lost=lambda obj: 'lost_time' in obj,
            get_id=lambda obj: obj['id'],
            get_lost_time=lambda obj: obj['lost_time'],
            set_lost_time=lambda obj, t: obj.update({'lost_time': t}),
            refind=lambda obj: obj.pop('lost_time', None)
        )

        # Draw the id on the objects
        for obj in tracked_objs:
            center = obj['shape'][0]
            cv2.putText(frame, str(obj['id']), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        cv2.namedWindow("Vision Sensor", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Vision Sensor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    sim.stopSimulation()

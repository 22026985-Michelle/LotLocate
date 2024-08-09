import numpy as np
from scipy.spatial import distance
from typing import List, Tuple
class Entry:
    def __init__(self):
        self.objects = {}
        self.obj_id = 0

    def update(self, bounding_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int, int]]:
        tracked_objects = []
        for box in bounding_boxes:
            x, y, w, h = box
            centroid = np.array([(x + w) / 2, (y + h) / 2])

            min_dist = float('inf')
            closest_obj_id = None
            for obj_id, obj_centroid in self.objects.items():
                dist = distance.euclidean(centroid, obj_centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_obj_id = obj_id

            if min_dist < 10 and closest_obj_id is not None:
                self.objects[closest_obj_id] = centroid
                tracked_objects.append((x, y, w, h, closest_obj_id))
            else:
                self.objects[self.obj_id] = centroid
                tracked_objects.append((x, y, w, h, self.obj_id))
                self.obj_id += 1

        return tracked_objects
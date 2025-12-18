import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox, max_history=5):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.original_boxes = deque(maxlen=max_history)
        self.original_boxes.append(bbox)
        self.confidences = deque(maxlen=30)

    def update(self, bbox, confidence=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.original_boxes.append(bbox)
        if confidence is not None:
            self.confidences.append(confidence)

    def predict(self):
        if ((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    def get_smoothed_box(self, window_size=5):
        if len(self.original_boxes) == 0:
            return self.get_state()[0]
        recent = list(self.original_boxes)[-window_size:]
        return np.mean(recent, axis=0)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w*h
        r = w/float(h)
        return np.array([x, y, s, r]).reshape((4,1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2]*x[3])
        h = x[2]/w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2-xx1)
    h = np.maximum(0., yy2-yy1)
    wh = w*h
    return wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
                 (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    unmatched_dets = [d for d in range(len(detections)) if d not in row_ind]
    unmatched_trks = [t for t in range(len(trackers)) if t not in col_ind]
    matches = []
    for d, t in zip(row_ind, col_ind):
        if iou_matrix[d,t] < iou_threshold:
            unmatched_dets.append(d)
            unmatched_trks.append(t)
        else:
            matches.append(np.array([d,t]).reshape(1,2))
    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_dets), np.array(unmatched_trks)

class SimpleSORT:
    def __init__(self, max_age=30, iou_threshold=0.3, min_box_area=1000, max_box_area=20000):
        self.trackers = []
        self.frame_count = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.max_track_id_ever = 0
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area

    def update(self, detections, confidences):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers),4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trks[t,:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.delete(trks, to_del, axis=0)
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trks, self.iou_threshold)

        # Update matched
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]], confidences[m[0]])

        # Create new trackers for unmatched
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            trk.confidences.append(confidences[i])
            self.trackers.append(trk)
            self.max_track_id_ever = max(self.max_track_id_ever, trk.id)

        results = []
        for trk in reversed(self.trackers):
            box = trk.get_smoothed_box()
            w,h = box[2]-box[0], box[3]-box[1]
            area = w*h

            # --- APPLY SIZE FILTER ---
            if area < self.min_box_area or area > self.max_box_area:
                continue

            # Remove old trackers
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
            else:
                avg_conf = np.mean(list(trk.confidences)) if trk.confidences else 0
                results.append((trk.id+1, box, avg_conf, len(trk.original_boxes)))

        return results

    def get_statistics(self):
        return {
            'active_tracks': len(self.trackers),
            'max_track_id': self.max_track_id_ever,
            'total_unique_tracks': KalmanBoxTracker.count
        }

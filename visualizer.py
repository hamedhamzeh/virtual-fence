import cv2
import numpy as np

class TrackVisualizer:
    @staticmethod
    def get_color_for_id(track_id):
        np.random.seed(int(track_id))
        hue = int(track_id * 17) % 180
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, color))

    @staticmethod
    def draw_tracked_box(frame, box, track_id, confidence, line_width=2, show_label=True):
        x1, y1, x2, y2 = box.astype(int)
        color = TrackVisualizer.get_color_for_id(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
        if show_label:
            label = f"ID:{track_id} {confidence:.2f}"
            font_scale = 0.4
            font_thickness = 1
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            cv2.rectangle(frame, (x1, y1 - label_height - baseline - 5),
                          (x1 + label_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

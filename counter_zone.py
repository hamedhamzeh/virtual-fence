import cv2

class ZoneCounter:
    def __init__(
        self,
        frame_width,
        frame_height,
        margin_top,
        margin_left,
        margin_right,
        margin_bottom,
        fps,
        apply_grace_after_sec=1.0,
        grace_frames=5
    ):
        """
        Margins are given in PIXELS from each side of the frame.
        grace_frames: ignore new IDs appearing inside zone for N frames
        """

        self.zone = (
            margin_left,
            margin_top,
            frame_width - margin_right,
            frame_height - margin_bottom
        )

        self.count = 0
        self.counted_ids = set()
        self.prev_inside = {}
        self.first_seen_frame = {}

        self.grace_frames = grace_frames
        self.frame_idx = 0
        self.grace_start_frame = int(apply_grace_after_sec * fps)

    def step(self, frame, tracked_objects):
        """
        SINGLE FUNCTION to call from inference.
        - Updates counter
        - Draws zone
        - Draws count
        """
        self.frame_idx += 1

        for track_id, box, _, _ in tracked_objects:
            self._update_track(track_id, box)

        self._draw_zone(frame)
        self._draw_counter(frame)

    def _update_track(self, track_id, box):
        if track_id not in self.first_seen_frame:
            self.first_seen_frame[track_id] = self.frame_idx

        if track_id in self.counted_ids:
            return

        cx, cy = self._centroid(box)
        inside = self._inside_zone(cx, cy)
        was_inside = self.prev_inside.get(track_id, False)
        self.prev_inside[track_id] = inside

        age = self.frame_idx - self.first_seen_frame[track_id]

        # ENTRY EVENT: outside → inside
        if inside and not was_inside:
            # Phase A: first second → count EVERYTHING
            if self.frame_idx < self.grace_start_frame:
                self.count += 1
                self.counted_ids.add(track_id)
                return

            # Phase B: apply grace filter
            if age <= self.grace_frames:
                return  # ignore ID-switch

            self.count += 1
            self.counted_ids.add(track_id)

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _inside_zone(self, x, y):
        x1, y1, x2, y2 = self.zone
        return x1 <= x <= x2 and y1 <= y <= y2

    # ------------------------
    def _draw_zone(self, frame):
        x1, y1, x2, y2 = map(int, self.zone)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 250), 3)

    def _draw_counter(self, frame):
        cv2.putText(
            frame,
            f"Count: {self.count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

"""
Detection Filtering Module
Contains filters for post-processing detections and tracked objects.
"""

import numpy as np


class DetectionFilter:
    """
    Filter for post-processing detections based on various criteria.
    """
    
    def __init__(
        self, 
        min_box_area=0,
        min_aspect_ratio=1.2,
        min_track_duration=10,
        max_box_area=None
    ):
        """
        Initialize detection filter with configurable parameters.
        
        Args:
            min_box_area (int): Minimum bounding box area in pixels to consider valid
            min_aspect_ratio (float): Minimum height/width ratio (persons are taller than wide)
            min_track_duration (int): Minimum frames a track must exist before being shown (anti-flicker)
            max_box_area (int, optional): Maximum bounding box area in pixels (to filter large blobs)
        """
        self.min_box_area = min_box_area
        self.min_aspect_ratio = min_aspect_ratio
        self.min_track_duration = min_track_duration
        self.max_box_area = max_box_area
    
    def filter_by_size(self, x1, y1, x2, y2):
        """
        Filter detection based on bounding box size.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            bool: True if detection passes size filter, False otherwise
        """
        w = x2 - x1
        h = y2 - y1
        area = w * h
        
        if area < float(self.min_box_area or 0):
            return False
        
        # Check maximum area if specified
        if self.max_box_area is not None and area > self.max_box_area:
            return False
        
        return True
    
    def filter_by_aspect_ratio(self, x1, y1, x2, y2):
        """
        Filter detection based on aspect ratio.
        Pedestrians are typically taller than wide (ratio > 1.0).
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            bool: True if detection passes aspect ratio filter, False otherwise
        """
        w = x2 - x1
        h = y2 - y1
        
        # Avoid division by zero
        if w == 0:
            return False
        
        aspect_ratio = h / w
        
        return aspect_ratio >= self.min_aspect_ratio
    
    def filter_by_track_duration(self, track_history_length):
        """
        Filter detection based on track duration (anti-flicker).
        Only show tracks that have persisted for minimum number of frames.
        
        Args:
            track_history_length (int): Number of frames the track has existed
            
        Returns:
            bool: True if detection passes duration filter, False otherwise
        """
        return track_history_length >= self.min_track_duration
    
    def apply_all_filters(self, box, track_history_length):
        """
        Apply all filters to a detection.
        
        Args:
            box (np.ndarray): Bounding box coordinates [x1, y1, x2, y2]
            track_history_length (int): Number of frames the track has existed
            
        Returns:
            bool: True if detection passes all filters, False otherwise
        """
        x1, y1, x2, y2 = box
        
        # Apply size filter
        if not self.filter_by_size(x1, y1, x2, y2):
            return False
        
        # Apply aspect ratio filter
        if not self.filter_by_aspect_ratio(x1, y1, x2, y2):
            return False
        
        # Apply track duration filter
        if not self.filter_by_track_duration(track_history_length):
            return False
        
        return True


def create_default_filter():
    """
    Create a filter with default parameters optimized for pedestrian detection.
    
    Returns:
        DetectionFilter: Configured filter instance
    """
    return DetectionFilter(
        min_box_area=1000,      # Ignore very small blobs
        min_aspect_ratio=1.2,    # Person must be taller than wide
        min_track_duration=15    # Wait 15 frames (~0.5s @ 30fps) before showing new person
    )

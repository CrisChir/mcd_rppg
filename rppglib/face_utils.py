"""
Face detection and processing utilities for rPPG.

This module provides functions for face detection, landmark extraction,
face cropping, and alignment.

Author: CrisChir
Date: September 2025
License: MIT
"""

import numpy as np
import cv2
from scipy.spatial import ConvexHull
from typing import Tuple, Optional, List

# Number of facial landmarks (68 for 2D face alignment)
NUM_LANDMARKS = 68


class FaceAlignmentExtractor:
    """
    Face detector using face_alignment library.
    
    This class wraps the face_alignment library to provide a consistent
    interface for face detection and landmark extraction.
    
    Args:
        device: Device to use for face detection ('cuda:0' or 'cpu')
    """
    
    def __init__(self, device: str = 'cuda:0'):
        """Initialize face alignment detector."""
        try:
            import face_alignment
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, 
                device=device
            )
        except ImportError:
            raise ImportError(
                "face_alignment library not found. "
                "Install with: pip install face_alignment"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize face_alignment: {e}")
        
        self.prev_preds = np.zeros((NUM_LANDMARKS, 2), dtype='int16')
        self.device = device
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect face and extract landmarks from a single frame.
        
        Args:
            frame: Input image as numpy array (H, W, 3)
        
        Returns:
            Landmarks as numpy array of shape (68, 2) with (x, y) coordinates.
            Returns previous landmarks if no face is detected.
        """
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Detect landmarks
        preds = self.fa.get_landmarks(frame)
        
        if preds is None:
            # Return previous landmarks if no face detected
            return self.prev_preds.copy()
        
        # Get first face (assuming single face per frame)
        landmarks = preds[0].astype('int16')
        
        # Update previous landmarks
        self.prev_preds = landmarks.copy()
        
        return landmarks


def detect_landmarks(video: np.ndarray, device: str = 'cuda:0') -> np.ndarray:
    """
    Detect facial landmarks for all frames in a video.
    
    Args:
        video: Input video as numpy array of shape (T, H, W, 3)
        device: Device to use for detection ('cuda:0' or 'cpu')
    
    Returns:
        Landmarks array of shape (T, 68, 2) containing (x, y) coordinates
    """
    detector = FaceAlignmentExtractor(device=device)
    
    landmarks = []
    for frame in video:
        lms = detector.process_frame(frame)
        landmarks.append(lms)
    
    return np.array(landmarks)


def find_bbox(landmarks: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find bounding box from landmarks.
    
    Args:
        landmarks: Landmarks array of shape (T, 68, 2) or (68, 2)
    
    Returns:
        Tuple of (x_min, x_max, y_min, y_max)
    """
    if landmarks.ndim == 3:
        # Multiple frames
        x_min = landmarks[:, :, 0].min()
        x_max = landmarks[:, :, 0].max()
        y_min = landmarks[:, :, 1].min()
        y_max = landmarks[:, :, 1].max()
    else:
        # Single frame
        x_min = landmarks[:, 0].min()
        x_max = landmarks[:, 0].max()
        y_min = landmarks[:, 1].min()
        y_max = landmarks[:, 1].max()
    
    return int(x_min), int(x_max), int(y_min), int(y_max)


def crop_video(video: np.ndarray, bbox: Tuple[int, int, int, int], 
               padding: int = 20) -> np.ndarray:
    """
    Crop video to face region defined by bounding box.
    
    Args:
        video: Input video array of shape (T, H, W, 3)
        bbox: Bounding box as (x_min, x_max, y_min, y_max)
        padding: Padding to add around the face (pixels)
    
    Returns:
        Cropped video array
    """
    x_min, x_max, y_min, y_max = bbox
    
    # Add padding
    x_min = max(0, x_min - padding)
    x_max = min(video.shape[2], x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(video.shape[1], y_max + padding)
    
    # Crop video
    cropped = video[:, y_min:y_max, x_min:x_max, :]
    
    return cropped


def crop_landmarks(landmarks: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Adjust landmarks coordinates after cropping.
    
    Args:
        landmarks: Landmarks array of shape (T, 68, 2)
        bbox: Bounding box as (x_min, x_max, y_min, y_max)
    
    Returns:
        Adjusted landmarks array
    """
    x_min, x_max, y_min, y_max = bbox
    
    # Create copy to avoid modifying original
    adjusted = landmarks.copy()
    
    # Adjust coordinates
    adjusted[:, :, 0] -= x_min
    adjusted[:, :, 1] -= y_min
    
    return adjusted


def get_convex_points(points: np.ndarray) -> np.ndarray:
    """
    Get convex hull points from a set of 2D points.
    
    Args:
        points: Input points array of shape (N, 2)
    
    Returns:
        Convex hull points as array of shape (M, 2)
    """
    try:
        hull = ConvexHull(points)
        convex_points = points[hull.vertices].astype('int32')
        return convex_points
    except Exception:
        # If convex hull fails, return all points
        return points.astype('int32')


def get_mask(frame: np.ndarray, poly_points: np.ndarray) -> np.ndarray:
    """
    Create a binary mask from polygon points.
    
    Args:
        frame: Input frame as numpy array (H, W, 3)
        poly_points: Polygon points as array of shape (N, 2)
    
    Returns:
        Binary mask as numpy array of shape (H, W)
    """
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype='uint8')
    
    # Reshape points for OpenCV
    polys = [poly_points.reshape(-1, 1, 2)]
    
    cv2.fillPoly(mask, polys, 255)
    mask = mask > 125
    
    return mask


def process_video(video: np.ndarray, min_face_size: int = 64, 
                  max_face_size: int = 512, device: str = 'cuda:0') -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a video by detecting faces, cropping to face region, and applying mask.
    
    This is the main preprocessing function that:
    1. Detects facial landmarks for each frame
    2. Finds bounding box from landmarks
    3. Crops video to face region
    4. Applies convex hull mask to remove background
    
    Args:
        video: Input video as numpy array of shape (T, H, W, 3)
        min_face_size: Minimum face size in pixels (for validation)
        max_face_size: Maximum face size in pixels (for validation)
        device: Device to use for face detection
    
    Returns:
        Tuple of (processed_video, landmarks):
        - processed_video: Video cropped to face region with mask applied
        - landmarks: Landmarks array of shape (T, 68, 2)
    
    Raises:
        AssertionError: If no face is detected in any frame
    """
    # Make a copy to avoid modifying original
    video = video.copy()
    
    # Detect landmarks
    landmarks = detect_landmarks(video, device=device)
    
    # Find bounding box
    bbox = find_bbox(landmarks)
    
    # Validate face size
    x_min, x_max, y_min, y_max = bbox
    face_width = x_max - x_min
    face_height = y_max - y_min
    
    if face_width < min_face_size or face_height < min_face_size:
        raise AssertionError(
            f"Face too small: {face_width}x{face_height} < {min_face_size}x{min_face_size}"
        )
    
    if face_width > max_face_size or face_height > max_face_size:
        raise AssertionError(
            f"Face too large: {face_width}x{face_height} > {max_face_size}x{max_face_size}"
        )
    
    # Crop video
    video = crop_video(video, bbox)
    
    # Adjust landmarks
    landmarks = crop_landmarks(landmarks, bbox)
    
    # Apply mask to each frame
    for i in range(video.shape[0]):
        points = landmarks[i]
        frame = video[i]
        convex = get_convex_points(points)
        mask = get_mask(frame, convex)
        video[i] *= mask[:, :, None]
    
    return video, landmarks


def extract_roi(video: np.ndarray, landmarks: np.ndarray, roi_name: str) -> np.ndarray:
    """
    Extract a specific Region of Interest (ROI) from video based on landmarks.
    
    Supported ROI names:
    - 'full_face': Entire face region
    - 'forehead': Forehead region
    - 'left_cheek': Left cheek region
    - 'right_cheek': Right cheek region
    - 'nose': Nose region
    - 'chin': Chin region
    - 'left_eye': Left eye region
    - 'right_eye': Right eye region
    - 'mouth': Mouth region
    
    Args:
        video: Input video array of shape (T, H, W, 3)
        landmarks: Landmarks array of shape (T, 68, 2)
        roi_name: Name of ROI to extract
    
    Returns:
        ROI video array of shape (T, H_roi, W_roi, 3)
    """
    # Define ROI landmark indices (68-point model)
    roi_indices = {
        'full_face': list(range(68)),
        'forehead': list(range(17, 27)),
        'left_cheek': list(range(1, 17)) + [27, 28, 29, 30],
        'right_cheek': list(range(17, 31)) + [31, 32, 33, 34, 35],
        'nose': list(range(27, 36)),
        'chin': list(range(6, 11)) + list(range(30, 36)),
        'left_eye': list(range(36, 42)),
        'right_eye': list(range(42, 48)),
        'mouth': list(range(48, 68))
    }
    
    if roi_name not in roi_indices:
        raise ValueError(f"Unknown ROI: {roi_name}. Available: {list(roi_indices.keys())}")
    
    indices = roi_indices[roi_name]
    
    # Extract ROI from each frame
    roi_frames = []
    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx]
        frame_landmarks = landmarks[frame_idx]
        
        # Get ROI landmarks
        roi_landmarks = frame_landmarks[indices]
        
        # Find bounding box of ROI
        x_min = roi_landmarks[:, 0].min()
        x_max = roi_landmarks[:, 0].max()
        y_min = roi_landmarks[:, 1].min()
        y_max = roi_landmarks[:, 1].max()
        
        # Add padding
        pad = 10
        x_min = max(0, x_min - pad)
        x_max = min(frame.shape[1], x_max + pad)
        y_min = max(0, y_min - pad)
        y_max = min(frame.shape[0], y_max + pad)
        
        # Extract ROI
        roi_frame = frame[y_min:y_max, x_min:x_max]
        roi_frames.append(roi_frame)
    
    return np.array(roi_frames)


def extract_multiple_rois(video: np.ndarray, landmarks: np.ndarray, 
                         roi_names: List[str]) -> dict:
    """
    Extract multiple ROIs from video.
    
    Args:
        video: Input video array
        landmarks: Landmarks array
        roi_names: List of ROI names to extract
    
    Returns:
        Dictionary mapping ROI names to ROI video arrays
    """
    rois = {}
    for roi_name in roi_names:
        rois[roi_name] = extract_roi(video, landmarks, roi_name)
    return rois

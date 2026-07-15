"""
Face detection and processing utilities for rPPG.

This module provides functions for face detection, landmark extraction,
face cropping, and alignment using MediaPipe's Face Landmark Detection task.

Note: This uses the new MediaPipe Tasks API (non-deprecated).

Author: CrisChir
Date: September 2025
License: MIT
"""

import numpy as np
import cv2
from scipy.spatial import ConvexHull
from typing import Tuple, Optional, List, Dict, Any
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe Face Landmark Detection configuration
# Using the new Tasks API (non-deprecated)
FACE_LANDMARKER_TASK_PATH = None  # Use default bundled model

# Number of facial landmarks (468 for MediaPipe face mesh)
NUM_LANDMARKS = 468


class MediaPipeFaceLandmarker:
    """
    Face landmark detector using MediaPipe's Face Landmark Detection task.
    
    This uses the new MediaPipe Tasks API which is the recommended approach
    (the old solutions API is deprecated).
    
    Args:
        base_options: Base options for the task
        running_mode: Running mode (IMAGE, VIDEO, or LIVE_STREAM)
        num_faces: Maximum number of faces to detect
        min_detection_confidence: Minimum confidence for face detection
        min_tracking_confidence: Minimum confidence for face tracking
    """
    
    def __init__(
        self,
        base_options: Optional[python.BaseOptions] = None,
        running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """Initialize MediaPipe face landmark detector."""
        if base_options is None:
            base_options = python.BaseOptions(
                model_asset_path=FACE_LANDMARKER_TASK_PATH
            )
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.running_mode = running_mode
        self.prev_landmarks = None
        self.frame_count = 0
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face landmarks in a single frame.
        
        Args:
            frame: Input image as numpy array (H, W, 3) in RGB format
        
        Returns:
            Landmarks as numpy array of shape (468, 2) with (x, y) coordinates,
            or None if no face is detected.
        """
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        try:
            # Detect landmarks
            if self.running_mode == vision.RunningMode.IMAGE:
                result = self.detector.detect(mp_image)
            elif self.running_mode == vision.RunningMode.VIDEO:
                result = self.detector.detect_for_video(mp_image, self.frame_count)
                self.frame_count += 1
            else:  # LIVE_STREAM
                result = self.detector.detect_for_video(mp_image, self.frame_count)
                self.frame_count += 1
            
            # Process result
            if result and result.face_landmarks:
                # Get first face (assuming single face per frame)
                face_landmarks = result.face_landmarks[0]
                
                # Convert to numpy array
                landmarks = np.array([
                    (lm.x * frame.shape[1], lm.y * frame.shape[0])
                    for lm in face_landmarks
                ], dtype=np.float32)
                
                # Store for next frame
                self.prev_landmarks = landmarks.copy()
                
                return landmarks
            else:
                # No face detected
                return None
                
        except Exception as e:
            print(f"Error in face detection: {e}")
            return None


def detect_landmarks(
    video: np.ndarray,
    running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
    num_faces: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> np.ndarray:
    """
    Detect facial landmarks for all frames in a video using MediaPipe.
    
    Args:
        video: Input video as numpy array of shape (T, H, W, 3)
        running_mode: MediaPipe running mode
        num_faces: Maximum number of faces to detect
        min_detection_confidence: Minimum confidence for detection
        min_tracking_confidence: Minimum confidence for tracking
    
    Returns:
        Landmarks array of shape (T, 468, 2) containing (x, y) coordinates
    
    Raises:
        RuntimeError: If no face is detected in the first frame
    """
    detector = MediaPipeFaceLandmarker(
        running_mode=running_mode,
        num_faces=num_faces,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    
    landmarks = []
    for frame in video:
        # Ensure frame is in RGB format (MediaPipe expects RGB)
        if frame.shape[2] == 3:  # RGB
            pass
        elif frame.shape[2] == 4:  # RGBA
            frame = frame[:, :, :3]
        
        lms = detector.detect(frame)
        
        if lms is None:
            # If no face detected, use previous landmarks or raise error
            if len(landmarks) > 0:
                # Use previous landmarks
                lms = landmarks[-1].copy()
            else:
                raise RuntimeError("No face detected in the first frame")
        
        landmarks.append(lms)
    
    return np.array(landmarks, dtype=np.float32)


def find_bbox(landmarks: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find bounding box from landmarks.
    
    Args:
        landmarks: Landmarks array of shape (T, 468, 2) or (468, 2)
    
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
        landmarks: Landmarks array of shape (T, 468, 2)
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


def process_video(
    video: np.ndarray,
    min_face_size: int = 64,
    max_face_size: int = 512,
    running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a video by detecting faces, cropping to face region, and applying mask.
    
    This is the main preprocessing function that:
    1. Detects facial landmarks for each frame using MediaPipe
    2. Finds bounding box from landmarks
    3. Crops video to face region
    4. Applies convex hull mask to remove background
    
    Args:
        video: Input video as numpy array of shape (T, H, W, 3)
        min_face_size: Minimum face size in pixels (for validation)
        max_face_size: Maximum face size in pixels (for validation)
        running_mode: MediaPipe running mode
        min_detection_confidence: Minimum confidence for face detection
        min_tracking_confidence: Minimum confidence for face tracking
    
    Returns:
        Tuple of (processed_video, landmarks):
        - processed_video: Video cropped to face region with mask applied
        - landmarks: Landmarks array of shape (T, 468, 2)
    
    Raises:
        RuntimeError: If no face is detected or face size is invalid
    """
    # Make a copy to avoid modifying original
    video = video.copy()
    
    # Detect landmarks using MediaPipe
    landmarks = detect_landmarks(
        video,
        running_mode=running_mode,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    
    # Find bounding box
    bbox = find_bbox(landmarks)
    
    # Validate face size
    x_min, x_max, y_min, y_max = bbox
    face_width = x_max - x_min
    face_height = y_max - y_min
    
    if face_width < min_face_size or face_height < min_face_size:
        raise RuntimeError(
            f"Face too small: {face_width}x{face_height} < {min_face_size}x{min_face_size}"
        )
    
    if face_width > max_face_size or face_height > max_face_size:
        raise RuntimeError(
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
    
    Supported ROI names for MediaPipe 468-point model:
    - 'full_face': Entire face region
    - 'forehead': Forehead region (landmarks 103-105, 332-334, 6-10)
    - 'left_cheek': Left cheek region
    - 'right_cheek': Right cheek region
    - 'nose': Nose region
    - 'chin': Chin region
    - 'left_eye': Left eye region
    - 'right_eye': Right eye region
    - 'mouth': Mouth region
    - 'left_eyebrow': Left eyebrow
    - 'right_eyebrow': Right eyebrow
    - 'left_iris': Left iris
    - 'right_iris': Right iris
    
    Args:
        video: Input video array of shape (T, H, W, 3)
        landmarks: Landmarks array of shape (T, 468, 2)
        roi_name: Name of ROI to extract
    
    Returns:
        ROI video array of shape (T, H_roi, W_roi, 3)
    """
    # Define ROI landmark indices for MediaPipe 468-point model
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    roi_indices = {
        'full_face': list(range(468)),
        'forehead': [103, 104, 105, 332, 333, 334, 6, 7, 8, 9, 10],
        'left_cheek': list(range(0, 100)) + list(range(200, 300)),
        'right_cheek': list(range(100, 200)) + list(range(300, 400)),
        'nose': list(range(1, 20)) + list(range(195, 220)),
        'chin': list(range(150, 170)) + list(range(370, 390)),
        'left_eye': list(range(22, 32)) + list(range(220, 230)) + list(range(33, 42)) + list(range(243, 252)),
        'right_eye': list(range(42, 52)) + list(range(252, 262)) + list(range(263, 272)),
        'mouth': list(range(60, 80)) + list(range(290, 310)) + list(range(310, 320)),
        'left_eyebrow': list(range(43, 66)),
        'right_eyebrow': list(range(66, 103)),
        'left_iris': [468, 469, 470, 471, 472],  # Iris landmarks
        'right_iris': [473, 474, 475, 476, 477],
        # Additional ROIs for rPPG (forehead is most important)
        'forehead_center': [10, 151, 9, 8, 7, 6, 103, 104, 105, 67, 109, 10],
        'forehead_left': [230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220],
        'forehead_right': [430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440],
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


def extract_multiple_rois(
    video: np.ndarray, 
    landmarks: np.ndarray, 
    roi_names: List[str]
) -> Dict[str, np.ndarray]:
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


def get_face_mesh_landmarks_info() -> Dict[str, Any]:
    """
    Get information about MediaPipe face mesh landmarks.
    
    Returns:
        Dictionary with landmark indices for different facial regions
    """
    return {
        'total_landmarks': 468,
        'face_oval': list(range(0, 468)),
        'lips': list(range(60, 80)) + list(range(290, 310)),
        'left_eye': list(range(22, 32)) + list(range(220, 230)),
        'right_eye': list(range(42, 52)) + list(range(252, 262)),
        'left_eyebrow': list(range(43, 66)),
        'right_eyebrow': list(range(66, 103)),
        'nose': list(range(1, 20)) + list(range(195, 220)),
        'forehead': [103, 104, 105, 332, 333, 334, 6, 7, 8, 9, 10],
        'chin': list(range(150, 170)) + list(range(370, 390)),
        'iris': list(range(468, 478)),
        'reference': 'https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png'
    }

"""
Data utilities for rPPG processing.

This module provides functions for loading and processing video and signal data.
Compatibility: Works with MediaPipe and OpenCV.

Author: CrisChir
Date: September 2025
License: MIT
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union


def load_video(
    video_path: str, 
    start_frame: int = 0, 
    end_frame: Optional[int] = None,
    target_fps: Optional[float] = None
) -> np.ndarray:
    """
    Load a video file and return frames as a numpy array.
    
    Args:
        video_path: Path to the video file
        start_frame: Starting frame index (default: 0)
        end_frame: Ending frame index (default: None, load all frames)
        target_fps: Target FPS for resampling (default: None, use original)
    
    Returns:
        numpy array of shape (T, H, W, 3) containing RGB frames
    
    Raises:
        FileNotFoundError: If video file does not exist
        ValueError: If video cannot be loaded
    """
    if not cv2.isVideoAvailable():
        raise ImportError("OpenCV video module not available. Install opencv-python-headless.")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if start_frame >= total_frames:
        raise ValueError(f"start_frame ({start_frame}) exceeds total frames ({total_frames})")
    
    if end_frame is not None and end_frame > total_frames:
        end_frame = total_frames
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read frames
    frames = []
    frame_idx = start_frame
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Stop if we've reached end_frame
        if end_frame is not None and frame_idx >= end_frame:
            break
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_idx += 1
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")
    
    return np.array(frames, dtype=np.uint8)


def load_ppg(ppg_path: str) -> np.ndarray:
    """
    Load a PPG signal from a numpy file.
    
    Args:
        ppg_path: Path to the PPG signal file (.npy)
    
    Returns:
        numpy array of shape (T,) containing PPG signal
    
    Raises:
        FileNotFoundError: If PPG file does not exist
    """
    ppg = np.load(ppg_path)
    
    # Ensure 1D array
    if ppg.ndim > 1:
        ppg = ppg.flatten()
    
    return ppg.astype(np.float32)


def load_ecg(ecg_path: str) -> np.ndarray:
    """
    Load an ECG signal from a numpy file.
    
    Args:
        ecg_path: Path to the ECG signal file (.npy)
    
    Returns:
        numpy array of shape (T,) containing ECG signal
    """
    ecg = np.load(ecg_path)
    
    # Ensure 1D array
    if ecg.ndim > 1:
        ecg = ecg.flatten()
    
    return ecg.astype(np.float32)


def load_metadata(metadata_path: str) -> dict:
    """
    Load metadata from a CSV or JSON file.
    
    Args:
        metadata_path: Path to metadata file
    
    Returns:
        Dictionary containing metadata
    """
    import pandas as pd
    import json
    
    if metadata_path.endswith('.csv'):
        df = pd.read_csv(metadata_path)
        return df.to_dict('records')
    elif metadata_path.endswith('.json'):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path}")


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary containing video properties:
        - path: Video file path
        - total_frames: Total number of frames
        - width: Frame width in pixels
        - height: Frame height in pixels
        - fps: Frames per second
        - duration: Duration in seconds
        - fourcc: Video codec
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Calculate duration
    if fps > 0:
        duration = total_frames / fps
    else:
        duration = 0.0
    
    cap.release()
    
    return {
        'path': video_path,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'fps': fps,
        'duration': duration,
        'fourcc': chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + \
                 chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
    }


def resample_signal(signal: np.ndarray, original_rate: float, target_rate: float) -> np.ndarray:
    """
    Resample a signal to a target sampling rate.
    
    Args:
        signal: Input signal array
        original_rate: Original sampling rate (Hz)
        target_rate: Target sampling rate (Hz)
    
    Returns:
        Resampled signal array
    """
    from scipy.signal import resample
    
    if original_rate == target_rate:
        return signal.copy()
    
    num_samples = int(len(signal) * target_rate / original_rate)
    return resample(signal, num_samples)


def normalize_video(video: np.ndarray) -> np.ndarray:
    """
    Normalize video frames to [0, 1] range.
    
    Args:
        video: Input video array of shape (T, H, W, 3)
    
    Returns:
        Normalized video array of shape (T, H, W, 3) with float32 dtype
    """
    video = video.astype(np.float32)
    video = video / 255.0
    return video


def standardize_video(video: np.ndarray) -> np.ndarray:
    """
    Standardize video frames (zero mean, unit variance).
    
    Args:
        video: Input video array of shape (T, H, W, 3)
    
    Returns:
        Standardized video array
    """
    video = video.astype(np.float32)
    mean = video.mean()
    std = video.std()
    
    if std < 1e-8:
        std = 1.0
    
    video = (video - mean) / std
    return video


def video_to_grayscale(video: np.ndarray) -> np.ndarray:
    """
    Convert RGB video to grayscale.
    
    Args:
        video: Input video array of shape (T, H, W, 3)
    
    Returns:
        Grayscale video array of shape (T, H, W)
    """
    grayscale = []
    for frame in video:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        grayscale.append(gray)
    return np.array(grayscale, dtype=np.uint8)


def resize_video(video: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize all frames in a video to a target size.
    
    Args:
        video: Input video array of shape (T, H, W, 3)
        target_size: Target size as (width, height)
    
    Returns:
        Resized video array of shape (T, target_height, target_width, 3)
    """
    target_width, target_height = target_size
    resized = []
    for frame in video:
        resized_frame = cv2.resize(frame, (target_width, target_height))
        resized.append(resized_frame)
    return np.array(resized, dtype=np.uint8)

#!/usr/bin/env python3
"""
MCD-rPPG Dataset Preprocessing Script

A consolidated script for preprocessing the MCD-rPPG dataset using MediaPipe.
This script handles the actual dataset structure with .PW files, ppg_sync, and meta files.

Usage:
    python preprocess_dataset.py --dataset_path /path/to/mcd_rppg --output_path ./preprocessed_data --limit 10

Author: CrisChir
Date: September 2025
License: MIT
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import json
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# MediaPipe imports (non-deprecated Tasks API)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PreprocessingConfig:
    """Configuration class for preprocessing parameters."""
    
    def __init__(self):
        # Input/Output paths
        self.dataset_path = None
        self.output_path = "./preprocessed_data"
        
        # Processing parameters
        self.window_size = 256
        self.stride = 64
        self.target_size = (128, 128)
        self.padding = 20
        
        # Face detection parameters (MediaPipe)
        self.num_faces = 1
        self.running_mode = vision.RunningMode.VIDEO
        
        # Quality filters
        self.min_face_size = 64
        self.max_face_size = 512
        
        # Signal processing
        self.ppg_low_freq = 0.75
        self.ppg_high_freq = 4.0
        self.frame_rate = 30.0
        self.ppg_rate = 100.0
        
        # Dataset splits
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        self.random_state = 42


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess MCD-rPPG dataset with actual file structure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset root (contains video/, ppg/, ppg_sync/, meta/, eeg/)')
    parser.add_argument('--output_path', type=str, default='./preprocessed_data',
                        help='Path to save preprocessed data')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process (for testing)')
    parser.add_argument('--window_size', type=int, default=256,
                        help='Frames per sample')
    parser.add_argument('--stride', type=int, default=64,
                        help='Step between samples')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose logging')
    
    return parser.parse_args()


def parse_pw_file(pw_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse .PW file format: {ppg_value}   {timestamp}
    Returns: (ppg_values, timestamps)
    """
    ppg_values = []
    timestamps = []
    
    with open(pw_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    ppg_values.append(float(parts[0]))
                    timestamp_str = ' '.join(parts[1:])
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                    timestamps.append(timestamp)
                except Exception as e:
                    if args.verbose:
                        print(f"Warning: Could not parse line: {line} - {e}")
    
    return np.array(ppg_values), np.array(timestamps)


def parse_ppg_sync_file(sync_path: str) -> Dict[int, float]:
    """
    Parse ppg_sync .txt file format: {frame_number} {sync_value}
    Returns: dict mapping frame_number to sync_value
    """
    sync_data = {}
    
    with open(sync_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    frame_num = int(parts[0])
                    sync_value = float(parts[1])
                    sync_data[frame_num] = sync_value
                except Exception as e:
                    if args.verbose:
                        print(f"Warning: Could not parse line: {line} - {e}")
    
    return sync_data


def parse_meta_file(meta_path: str) -> Dict[int, datetime]:
    """
    Parse meta .txt file format: {frame_number}  {timestamp}
    Returns: dict mapping frame_number to timestamp
    """
    meta_data = {}
    
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    frame_num = int(parts[0])
                    timestamp_str = ' '.join(parts[1:])
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                    meta_data[frame_num] = timestamp
                except Exception as e:
                    if args.verbose:
                        print(f"Warning: Could not parse line: {line} - {e}")
    
    return meta_data


def extract_subject_id(filename: str) -> str:
    """Extract subject ID from filename."""
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    parts = name.split('_')
    if len(parts) >= 1:
        return parts[0]
    return 'unknown'


def extract_camera_name(filename: str) -> str:
    """Extract camera name from filename."""
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[1]
    return 'unknown'


def extract_condition(filename: str) -> str:
    """Extract condition from filename."""
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    parts = name.split('_')
    if len(parts) >= 3:
        return parts[-1]
    return 'unknown'


def find_matching_files(subject_id: str, camera: str, condition: str, dataset_path: str) -> Dict[str, str]:
    """
    Find all matching files for a given subject, camera, and condition.
    Returns: dict with paths to video, ppg, ppg_sync, meta, eeg files
    """
    files = {
        'video': None,
        'ppg': None,
        'ppg_sync': None,
        'meta': None,
        'eeg': None
    }
    
    # Find video file
    video_dir = os.path.join(dataset_path, 'video')
    video_pattern = f"{subject_id}_{camera}_{condition}.avi"
    video_path = os.path.join(video_dir, video_pattern)
    if os.path.exists(video_path):
        files['video'] = video_path
    
    # Find PPG file (one PPG file per subject+condition, not per camera)
    ppg_dir = os.path.join(dataset_path, 'ppg')
    ppg_pattern = f"{subject_id}_{condition}.PW"
    ppg_path = os.path.join(ppg_dir, ppg_pattern)
    if os.path.exists(ppg_path):
        files['ppg'] = ppg_path
    
    # Find PPG sync file
    ppg_sync_dir = os.path.join(dataset_path, 'ppg_sync')
    ppg_sync_pattern = f"{subject_id}_{camera}_{condition}.txt"
    ppg_sync_path = os.path.join(ppg_sync_dir, ppg_sync_pattern)
    if os.path.exists(ppg_sync_path):
        files['ppg_sync'] = ppg_sync_path
    
    # Find meta file
    meta_dir = os.path.join(dataset_path, 'meta')
    meta_pattern = f"{subject_id}_{camera}_{condition}.txt"
    meta_path = os.path.join(meta_dir, meta_pattern)
    if os.path.exists(meta_path):
        files['meta'] = meta_path
    
    # Find EEG file
    eeg_dir = os.path.join(dataset_path, 'eeg')
    eeg_pattern = f"{subject_id}_{condition}.json"
    eeg_path = os.path.join(eeg_dir, eeg_pattern)
    if os.path.exists(eeg_path):
        files['eeg'] = eeg_path
    
    return files


def load_dataset_from_structure(dataset_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    Load dataset from the actual file structure.
    Returns: List of sample dictionaries
    """
    samples = []
    
    # Get all video files
    video_dir = os.path.join(dataset_path, 'video')
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return samples
    
    video_files = []
    for file in os.listdir(video_dir):
        if file.endswith('.avi') or file.endswith('.mp4'):
            video_files.append(os.path.join(video_dir, file))
    
    print(f"Found {len(video_files)} video files")
    
    if limit:
        video_files = video_files[:limit]
    
    # Create samples
    for video_file in video_files:
        subject_id = extract_subject_id(video_file)
        camera = extract_camera_name(video_file)
        condition = extract_condition(video_file)
        
        # Find matching files
        matching_files = find_matching_files(subject_id, camera, condition, dataset_path)
        
        sample = {
            'subject_id': subject_id,
            'camera': camera,
            'condition': condition,
            'video_file': video_file,
            'ppg_file': matching_files['ppg'],
            'ppg_sync_file': matching_files['ppg_sync'],
            'meta_file': matching_files['meta'],
            'eeg_file': matching_files['eeg']
        }
        samples.append(sample)
    
    return samples


def load_video(video_path: str, start_frame: int = 0, end_frame = None):
    """Load a video file and return frames as numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    frame_idx = start_frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if end_frame is not None and frame_idx >= end_frame:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_idx += 1
    
    cap.release()
    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")
    return np.array(frames, dtype=np.uint8)


def initialize_mediapipe(config: PreprocessingConfig) -> vision.FaceLandmarker:
    """Initialize MediaPipe Face Landmarker with correct API."""
    print("Initializing MediaPipe Face Landmarker...")
    
    # Create base options
    base_options = python.BaseOptions(model_asset_path=None)
    
    # Create face landmarker options - CORRECTED: no min_detection_confidence here
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=config.running_mode,
        num_faces=config.num_faces
    )
    
    # Create detector
    detector = vision.FaceLandmarker.create_from_options(options)
    
    print("✅ MediaPipe Face Landmarker initialized!")
    print(f"   Running mode: {config.running_mode}")
    print(f"   Num faces: {config.num_faces}")
    
    return detector


def detect_face_landmarks(frame, detector, frame_idx=0):
    """Detect face landmarks using MediaPipe."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    try:
        result = detector.detect_for_video(mp_image, frame_idx)
        if result and result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            landmarks = np.array([
                (lm.x * frame.shape[1], lm.y * frame.shape[0])
                for lm in face_landmarks
            ], dtype=np.float32)
            return landmarks
        return None
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None


def find_bbox(landmarks):
    """Find bounding box from landmarks."""
    x_min = landmarks[:, 0].min()
    x_max = landmarks[:, 0].max()
    y_min = landmarks[:, 1].min()
    y_max = landmarks[:, 1].max()
    return int(x_min), int(x_max), int(y_min), int(y_max)


def crop_face(frame, bbox, padding=20):
    """Crop face region from frame."""
    x_min, x_max, y_min, y_max = bbox
    x_min = max(0, x_min - padding)
    x_max = min(frame.shape[1], x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(frame.shape[0], y_max + padding)
    return frame[y_min:y_max, x_min:x_max]


def get_convex_hull_mask(frame, landmarks):
    """Create convex hull mask from landmarks."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(landmarks)
        convex_points = landmarks[hull.vertices].astype('int32')
    except:
        convex_points = landmarks.astype('int32')
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype='uint8')
    cv2.fillPoly(mask, [convex_points.reshape(-1, 1, 2)], 255)
    return mask > 125


def process_frame(frame, detector, frame_idx, prev_landmarks=None, config=None):
    """Process a single frame: detect face, crop, and apply mask."""
    if config is None:
        config = PreprocessingConfig()
    
    landmarks = detect_face_landmarks(frame, detector, frame_idx)
    
    if landmarks is None:
        if prev_landmarks is not None:
            landmarks = prev_landmarks
        else:
            return None, None
    
    bbox = find_bbox(landmarks)
    x_min, x_max, y_min, y_max = bbox
    face_width = x_max - x_min
    face_height = y_max - y_min
    
    if face_width < config.min_face_size or face_height < config.min_face_size:
        return None, landmarks
    if face_width > config.max_face_size or face_height > config.max_face_size:
        return None, landmarks
    
    face = crop_face(frame, bbox, padding=config.padding)
    mask = get_convex_hull_mask(face, landmarks - np.array([x_min, y_min]))
    face = face * mask[:, :, None]
    face = cv2.resize(face, config.target_size)
    
    return face, landmarks


def align_ppg_with_video(ppg_values, ppg_timestamps, meta_data, frame_rate=30.0):
    """
    Align PPG signal with video frames using timestamps from meta file.
    Returns: ppg_signal aligned with video frames
    """
    if len(meta_data) == 0 or len(ppg_timestamps) == 0:
        return np.zeros(len(meta_data))
    
    # Convert timestamps to seconds since first timestamp
    first_ppg_time = ppg_timestamps[0]
    first_meta_time = list(meta_data.values())[0]
    
    ppg_seconds = [(ts - first_ppg_time).total_seconds() for ts in ppg_timestamps]
    meta_seconds = [(ts - first_meta_time).total_seconds() for ts in meta_data.values()]
    
    # Interpolate PPG values at video frame timestamps
    interp_func = interp1d(ppg_seconds, ppg_values, kind='linear', fill_value='extrapolate')
    aligned_ppg = interp_func(meta_seconds)
    
    return aligned_ppg


def bandpass_filter(signal, low_freq, high_freq, fs, order=4):
    """Apply bandpass filter to signal."""
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def preprocess_ppg(ppg, config):
    """Preprocess PPG signal."""
    ppg_filtered = bandpass_filter(ppg, config.ppg_low_freq, config.ppg_high_freq, config.ppg_rate)
    return (ppg_filtered - ppg_filtered.mean()) / ppg_filtered.std()


def extract_chunks(video, ppg, window_size, stride, frame_rate=30.0, ppg_rate=100.0):
    """Extract chunks from video and PPG."""
    chunks = []
    ppg_chunks = []
    num_frames = video.shape[0]
    ppg_per_frame = ppg_rate / frame_rate
    
    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        video_chunk = video[start:end]
        ppg_start = int(start * ppg_per_frame)
        ppg_end = int(end * ppg_per_frame)
        ppg_chunk = ppg[ppg_start:ppg_end]
        if len(ppg_chunk) < window_size:
            continue
        if len(ppg_chunk) > window_size:
            ppg_chunk = ppg_chunk[:window_size]
        chunks.append(video_chunk)
        ppg_chunks.append(ppg_chunk)
    
    return np.array(chunks), np.array(ppg_chunks)


def process_sample(sample, detector, config):
    """Process a single sample (video + PPG)."""
    result = {
        'subject_id': sample.get('subject_id', 'unknown'),
        'camera': sample.get('camera', 'unknown'),
        'condition': sample.get('condition', 'unknown'),
        'video_file': sample.get('video_file', ''),
        'video_chunks': [],
        'ppg_chunks': [],
        'landmarks': [],
        'success': False,
        'error': None
    }
    
    try:
        # Load video
        video = load_video(sample['video_file'])
        result['original_frames'] = len(video)
        
        # Load and align PPG
        ppg = None
        if sample.get('ppg_file') and os.path.exists(sample['ppg_file']):
            ppg_values, ppg_timestamps = parse_pw_file(sample['ppg_file'])
            
            # Load meta data for alignment
            meta_data = {}
            if sample.get('meta_file') and os.path.exists(sample['meta_file']):
                meta_data = parse_meta_file(sample['meta_file'])
            
            # Align PPG with video
            if len(meta_data) > 0:
                ppg = align_ppg_with_video(ppg_values, ppg_timestamps, meta_data, config.frame_rate)
            else:
                # Fallback: use first N PPG values
                ppg = ppg_values[:len(video)]
            
            ppg = preprocess_ppg(ppg, config)
            result['original_ppg_length'] = len(ppg)
        
        # Process each frame
        processed_frames = []
        all_landmarks = []
        prev_landmarks = None
        
        for frame_idx, frame in enumerate(video):
            processed_frame, landmarks = process_frame(frame, detector, frame_idx, prev_landmarks, config)
            if processed_frame is None:
                continue
            processed_frames.append(processed_frame)
            all_landmarks.append(landmarks)
            prev_landmarks = landmarks.copy()
        
        if len(processed_frames) < config.window_size:
            result['error'] = f"Too few frames: {len(processed_frames)}"
            return result
        
        # Convert to arrays
        processed_video = np.array(processed_frames)
        landmarks_array = np.array(all_landmarks)
        
        # Extract chunks
        if ppg is not None and len(ppg) > 0:
            video_chunks, ppg_chunks = extract_chunks(
                processed_video, ppg, config.window_size, config.stride,
                config.frame_rate, config.ppg_rate
            )
        else:
            # Create dummy PPG if not available
            video_chunks, _ = extract_chunks(
                processed_video, np.zeros(len(processed_video)),
                config.window_size, config.stride
            )
            ppg_chunks = np.zeros((len(video_chunks), config.window_size))
        
        result['video_chunks'] = video_chunks
        result['ppg_chunks'] = ppg_chunks
        result['landmarks'] = landmarks_array
        result['processed_frames'] = len(processed_frames)
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
    
    return result


def save_preprocessed_data(processed_samples, output_path, config):
    """Save preprocessed data to disk."""
    os.makedirs(output_path, exist_ok=True)
    
    all_video_chunks = []
    all_ppg_chunks = []
    all_metadata = []
    all_landmarks = []
    
    for i, sample in enumerate(processed_samples):
        if not sample['success']:
            continue
        all_video_chunks.extend(sample['video_chunks'])
        all_ppg_chunks.extend(sample['ppg_chunks'])
        all_landmarks.extend(sample['landmarks'])
        
        for j in range(len(sample['video_chunks'])):
            metadata = {
                'sample_idx': i,
                'chunk_idx': j,
                'subject_id': sample['subject_id'],
                'camera': sample['camera'],
                'condition': sample['condition'],
                'original_frames': sample['original_frames'],
                'processed_frames': sample['processed_frames'],
                'video_file': sample['video_file']
            }
            all_metadata.append(metadata)
    
    if all_video_chunks:
        np.save(os.path.join(output_path, 'video_chunks.npy'), np.array(all_video_chunks))
        print(f"Saved video chunks: shape {np.array(all_video_chunks).shape}")
    
    if all_ppg_chunks:
        np.save(os.path.join(output_path, 'ppg_chunks.npy'), np.array(all_ppg_chunks))
        print(f"Saved PPG chunks: shape {np.array(all_ppg_chunks).shape}")
    
    if all_landmarks:
        np.save(os.path.join(output_path, 'landmarks.npy'), np.array(all_landmarks))
        print(f"Saved landmarks: shape {np.array(all_landmarks).shape}")
    
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)
    print(f"Saved metadata: {len(metadata_df)} entries")
    
    # Create splits by subject
    subject_ids = metadata_df['subject_id'].unique()
    train_subjects, test_subjects = train_test_split(
        subject_ids, test_size=config.test_ratio + config.val_ratio, random_state=config.random_state
    )
    val_subjects, test_subjects = train_test_split(
        test_subjects, test_size=config.test_ratio / (config.test_ratio + config.val_ratio), random_state=config.random_state
    )
    
    for split_name, subjects in [('train', train_subjects), ('val', val_subjects), ('test', test_subjects)]:
        with open(os.path.join(output_path, f'{split_name}_subjects.txt'), 'w') as f:
            for subject in subjects:
                f.write(f'{subject}\n')
        print(f"Saved {split_name} split: {len(subjects)} subjects")
    
    return {
        'video_chunks': os.path.join(output_path, 'video_chunks.npy'),
        'ppg_chunks': os.path.join(output_path, 'ppg_chunks.npy'),
        'landmarks': os.path.join(output_path, 'landmarks.npy'),
        'metadata': os.path.join(output_path, 'metadata.csv'),
        'train': os.path.join(output_path, 'train_subjects.txt'),
        'val': os.path.join(output_path, 'val_subjects.txt'),
        'test': os.path.join(output_path, 'test_subjects.txt')
    }


def main():
    """Main preprocessing function."""
    global args
    args = parse_args()
    
    config = PreprocessingConfig()
    config.dataset_path = args.dataset_path
    config.output_path = args.output_path
    config.window_size = args.window_size
    config.stride = args.stride
    
    print("=" * 80)
    print("MCD-rPPG Dataset Preprocessing")
    print("=" * 80)
    print(f"Dataset path: {config.dataset_path}")
    print(f"Output path: {config.output_path}")
    print(f"Window size: {config.window_size}")
    print(f"Stride: {config.stride}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    samples = load_dataset_from_structure(config.dataset_path, limit=args.limit)
    print(f"Loaded {len(samples)} samples")
    
    if not samples:
        print("No samples found. Exiting.")
        return
    
    # Display sample information
    print("\nSample information:")
    for i, sample in enumerate(samples[:3]):
        print(f"  {i+1}. Subject: {sample['subject_id']}, Camera: {sample['camera']}, Condition: {sample['condition']}")
        print(f"     Video: {os.path.exists(sample['video_file'])}")
        print(f"     PPG: {sample['ppg_file']} ({os.path.exists(sample['ppg_file']) if sample['ppg_file'] else 'None'})")
        print(f"     PPG Sync: {sample['ppg_sync_file']} ({os.path.exists(sample['ppg_sync_file']) if sample['ppg_sync_file'] else 'None'})")
        print(f"     Meta: {sample['meta_file']} ({os.path.exists(sample['meta_file']) if sample['meta_file'] else 'None'})")
        print()
    
    # Initialize MediaPipe
    detector = initialize_mediapipe(config)
    
    # Process samples
    print(f"Processing {len(samples)} samples...")
    processed_samples = []
    
    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        result = process_sample(sample, detector, config)
        processed_samples.append(result)
        
        if result['success']:
            print(f"  ✅ Sample {i+1}: {result['subject_id']}_{result['camera']}_{result['condition']} - {len(result['video_chunks'])} chunks")
        else:
            print(f"  ❌ Sample {i+1}: Error - {result['error']}")
    
    success_count = sum(1 for r in processed_samples if r['success'])
    failure_count = len(processed_samples) - success_count
    
    print(f"\n✅ Processing complete! Success: {success_count}/{len(processed_samples)}, Failed: {failure_count}")
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    saved_files = save_preprocessed_data(processed_samples, config.output_path, config)
    print("\n✅ All data saved successfully!")
    
    for key, path in saved_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  {key}: {path} ({size:.2f} MB)")


if __name__ == '__main__':
    main()

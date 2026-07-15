#!/usr/bin/env python3
"""
MCD-rPPG Dataset Preprocessing Script

A consolidated script for preprocessing the MCD-rPPG dataset using MediaPipe.
This script handles the actual dataset structure with .PW files, ppg_sync, and meta files.

IMPORTANT: Uses SIMPLE RATIO-BASED SYNCHRONIZATION to match original preprocessing scripts.
- PPG at 100 Hz, Video at 30 FPS
- Ratio: 100/30 = 3.333... PPG samples per video frame
- Direct mapping: ppg_chunk = ppg[start_frame * 3.333 : end_frame * 3.333]
- NO complex timestamp parsing (matches original scripts)

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
        self.window_size = 256      # Frames per sample (MUST match original)
        self.stride = 64            # Step between samples (MUST match original)
        self.target_size = (128, 128)  # Target face size
        self.padding = 20           # Padding around face
        
        # Face detection parameters (MediaPipe)
        self.num_faces = 1
        self.running_mode = vision.RunningMode.VIDEO
        
        # Quality filters
        self.min_face_size = 64
        self.max_face_size = 512
        
        # Signal processing
        self.ppg_low_freq = 0.75    # Hz
        self.ppg_high_freq = 4.0    # Hz
        self.frame_rate = 30.0     # FPS (MUST match original)
        self.ppg_rate = 100.0      # Hz (MUST match original)
        
        # Dataset splits
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        self.random_state = 42


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess MCD-rPPG dataset with SIMPLE RATIO-BASED SYNCHRONIZATION',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset root (contains video/, ppg/, ppg_sync/, meta/, eeg/)')
    parser.add_argument('--output_path', type=str, default='./preprocessed_data',
                        help='Path to save preprocessed data')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process (for testing)')
    parser.add_argument('--window_size', type=int, default=256,
                        help='Frames per sample (MUST match original scripts)')
    parser.add_argument('--stride', type=int, default=64,
                        help='Step between samples (MUST match original scripts)')
    parser.add_argument('--frame_rate', type=float, default=30.0,
                        help='Video frame rate (MUST match original)')
    parser.add_argument('--ppg_rate', type=float, default=100.0,
                        help='PPG sampling rate (MUST match original)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose logging')
    
    return parser.parse_args()


def parse_pw_file_simple(pw_path: str) -> np.ndarray:
    """
    Parse .PW file - SIMPLE VERSION.
    
    Since we're using ratio-based synchronization, we just need the PPG values.
    We don't need to parse timestamps.
    
    Args:
        pw_path: Path to .PW file
    
    Returns:
        ppg_values: Array of PPG values
    """
    ppg_values = []
    with open(pw_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    ppg_values.append(float(parts[0]))
                except Exception as e:
                    if args.verbose:
                        print(f"Warning: Could not parse line: {line[:50]}... - {e}")
    
    return np.array(ppg_values)


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
    
    base_options = python.BaseOptions(model_asset_path=None)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=config.running_mode,
        num_faces=config.num_faces
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    print("MediaPipe Face Landmarker initialized")
    print(f"  Running mode: {config.running_mode}")
    print(f"  Num faces: {config.num_faces}")
    
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
    ppg_filtered = bandpass_filter(
        ppg,
        config.ppg_low_freq,
        config.ppg_high_freq,
        config.ppg_rate
    )
    return (ppg_filtered - ppg_filtered.mean()) / ppg_filtered.std()


def extract_chunks_simple(video, ppg, window_size, stride, video_fps=30.0, ppg_fps=100.0):
    """
    Extract chunks using SIMPLE RATIO-BASED SYNCHRONIZATION.
    
    This matches the ORIGINAL preprocessing scripts approach:
    - PPG at 100 Hz, Video at 30 FPS
    - Ratio: 100/30 = 3.333... PPG samples per video frame
    - Direct mapping without complex timestamp parsing
    
    Args:
        video: Video array of shape (T_video, H, W, 3)
        ppg: PPG array of shape (T_ppg,)
        window_size: Number of frames per chunk
        stride: Step between chunks
        video_fps: Video frame rate
        ppg_fps: PPG sampling rate
    
    Returns:
        video_chunks: Array of shape (N, window_size, H, W, 3)
        ppg_chunks: Array of shape (N, window_size)
    """
    chunks = []
    ppg_chunks = []
    
    # Calculate PPG samples per video frame (THIS IS THE KEY)
    ppg_per_video_frame = ppg_fps / video_fps  # 100/30 = 3.333...
    
    num_frames = video.shape[0]
    
    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        
        # Extract video chunk
        video_chunk = video[start:end]
        
        # Extract corresponding PPG chunk using SIMPLE RATIO
        ppg_start = int(start * ppg_per_video_frame)
        ppg_end = int(end * ppg_per_video_frame)
        ppg_chunk = ppg[ppg_start:ppg_end]
        
        # Ensure PPG chunk has correct length
        if len(ppg_chunk) == window_size:
            chunks.append(video_chunk)
            ppg_chunks.append(ppg_chunk)
    
    return np.array(chunks), np.array(ppg_chunks)


def process_sample(sample, detector, config):
    """Process a single sample (video + PPG) using SIMPLE RATIO-BASED synchronization."""
    result = {
        'subject_id': sample.get('subject_id', 'unknown'),
        'camera': sample.get('camera', 'unknown'),
        'condition': sample.get('condition', 'unknown'),
        'video_file': sample.get('video_file', ''),
        'video_chunks': [],
        'ppg_chunks': [],
        'landmarks': [],
        'success': False,
        'error': None,
        'alignment_info': {}
    }
    
    try:
        # Load video
        video = load_video(sample['video_file'])
        result['original_frames'] = len(video)
        
        # Load PPG (simple version - just values, no timestamps)
        ppg = None
        if sample.get('ppg_file') and os.path.exists(sample['ppg_file']):
            ppg = parse_pw_file_simple(sample['ppg_file'])
            
            # Verify PPG/Video ratio matches expected
            expected_ratio = config.ppg_rate / config.frame_rate
            actual_ratio = len(ppg) / len(video)
            result['alignment_info']['expected_ratio'] = expected_ratio
            result['alignment_info']['actual_ratio'] = actual_ratio
            result['alignment_info']['ratio_match'] = abs(actual_ratio - expected_ratio) < 0.01
            
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
        
        # Extract chunks using SIMPLE RATIO-BASED approach
        if ppg is not None and len(ppg) > 0:
            video_chunks, ppg_chunks = extract_chunks_simple(
                processed_video, ppg, config.window_size, config.stride,
                config.frame_rate, config.ppg_rate
            )
        else:
            # Create dummy PPG if not available
            video_chunks, _ = extract_chunks_simple(
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
            # Add alignment info
            if 'alignment_info' in sample:
                metadata.update({
                    'expected_ratio': sample['alignment_info'].get('expected_ratio'),
                    'actual_ratio': sample['alignment_info'].get('actual_ratio'),
                    'ratio_match': sample['alignment_info'].get('ratio_match')
                })
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
    config.frame_rate = args.frame_rate
    config.ppg_rate = args.ppg_rate
    
    print("=" * 80)
    print("MCD-rPPG Dataset Preprocessing")
    print("Using SIMPLE RATIO-BASED SYNCHRONIZATION (matches original scripts)")
    print("=" * 80)
    print(f"Dataset path: {config.dataset_path}")
    print(f"Output path: {config.output_path}")
    print(f"Window size: {config.window_size}")
    print(f"Stride: {config.stride}")
    print(f"Frame rate: {config.frame_rate} FPS")
    print(f"PPG rate: {config.ppg_rate} Hz")
    print(f"PPG/Video ratio: {config.ppg_rate/config.frame_rate:.4f}")
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
            print(f"  Sample {i+1}: {result['subject_id']}_{result['camera']}_{result['condition']} - {len(result['video_chunks'])} chunks, ratio_match={result['alignment_info'].get('ratio_match', False)}")
        else:
            print(f"  Sample {i+1}: Error - {result['error']}")
    
    success_count = sum(1 for r in processed_samples if r['success'])
    failure_count = len(processed_samples) - success_count
    
    print(f"\n Processing complete! Success: {success_count}/{len(processed_samples)}, Failed: {failure_count}")
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    saved_files = save_preprocessed_data(processed_samples, config.output_path, config)
    print("\n All data saved successfully!")
    
    for key, path in saved_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  {key}: {path} ({size:.2f} MB)")
    
    # Print alignment summary
    print(f"\n Alignment Summary:")
    ratio_matches = sum(1 for r in processed_samples if r['success'] and r['alignment_info'].get('ratio_match', False))
    print(f"  Videos with correct PPG/Video ratio: {ratio_matches}/{success_count}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
MCD-rPPG Preprocessing Test Script

This script tests the preprocessing pipeline on 2 random videos from the dataset.
It visualizes and verifies:
1. Original video frame
2. Preprocessed face (cropped and resized)
3. Facial landmarks overlay
4. PPG signal synchronized with video chunks
5. Vital signs from database
6. Video and PPG lengths
7. Alignment verification

Usage:
    python test_preprocessing.py --db_path /path/to/db.csv --dataset_root /path/to/dataset --output_path ./test_output

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
import random
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test preprocessing on 2 random videos from MCD-rPPG dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to database CSV file (db.csv)')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root path to dataset directory')
    parser.add_argument('--output_path', type=str, default='./test_output',
                        help='Path to save test output')
    parser.add_argument('--num_tests', type=int, default=2,
                        help='Number of random videos to test')
    parser.add_argument('--window_size', type=int, default=256,
                        help='Window size for chunks')
    parser.add_argument('--stride', type=int, default=64,
                        help='Stride for chunks')
    parser.add_argument('--frame_rate', type=float, default=30.0,
                        help='Video frame rate')
    parser.add_argument('--ppg_rate', type=float, default=100.0,
                        help='PPG sampling rate')
    parser.add_argument('--target_size', type=tuple, default=(128, 128),
                        help='Target face size')
    
    return parser.parse_args()


def initialize_mediapipe():
    """Initialize MediaPipe Face Landmarker."""
    print("Initializing MediaPipe Face Landmarker...")
    base_options = python.BaseOptions(model_asset_path=None)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe initialized")
    return detector


def parse_pw_file(pw_path):
    """Parse .PW file: {ppg_value}   {timestamp}"""
    ppg_values, timestamps = [], []
    with open(pw_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    ppg_values.append(float(parts[0]))
                    timestamps.append(datetime.strptime(' '.join(parts[1:]), '%Y-%m-%d %H:%M:%S.%f'))
                except Exception as e:
                    print(f"Warning: Could not parse line: {line[:50]}...")
    return np.array(ppg_values), np.array(timestamps)


def parse_meta_file(meta_path):
    """Parse meta file: {frame_number}  {timestamp}"""
    meta_data = {}
    with open(meta_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    meta_data[int(parts[0])] = datetime.strptime(' '.join(parts[1:]), '%Y-%m-%d %H:%M:%S.%f')
                except Exception as e:
                    print(f"Warning: Could not parse line: {line[:50]}...")
    return meta_data


def parse_ppg_sync_file(sync_path):
    """Parse PPG sync file: {frame_number} {sync_value}"""
    sync_data = {}
    with open(sync_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    sync_data[int(parts[0])] = float(parts[1])
                except Exception as e:
                    print(f"Warning: Could not parse line: {line[:50]}...")
    return sync_data


def align_ppg_with_video(ppg_values, ppg_timestamps, meta_data, frame_rate=30.0):
    """
    Align PPG signal with video frames using timestamp interpolation.
    
    This is the CORE alignment function that ensures PPG and video are synchronized.
    
    Args:
        ppg_values: Array of PPG values from .PW file
        ppg_timestamps: Array of timestamps for each PPG value
        meta_data: Dict mapping frame_number to timestamp
        frame_rate: Video frame rate (FPS)
    
    Returns:
        aligned_ppg: Array of PPG values aligned with video frames
    """
    if len(meta_data) == 0 or len(ppg_timestamps) == 0:
        return np.zeros(len(meta_data))
    
    # Convert all timestamps to seconds relative to first timestamp
    first_ppg = ppg_timestamps[0]
    first_meta = list(meta_data.values())[0]
    
    # PPG timestamps in seconds
    ppg_seconds = [(ts - first_ppg).total_seconds() for ts in ppg_timestamps]
    
    # Sort meta data by frame number and convert to seconds
    sorted_frames = sorted(meta_data.keys())
    meta_seconds = [(meta_data[f] - first_meta).total_seconds() for f in sorted_frames]
    
    # Interpolate PPG values at video frame timestamps
    interp_func = interp1d(ppg_seconds, ppg_values, kind='linear', fill_value='extrapolate')
    aligned_ppg = interp_func(meta_seconds)
    
    return aligned_ppg


def preprocess_ppg(ppg, low_freq=0.75, high_freq=4.0, fs=100.0):
    """Apply bandpass filter and normalize PPG signal."""
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, ppg)
    return (filtered - filtered.mean()) / filtered.std()


def load_video(video_path):
    """Load video file and return frames as numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Could not open: {video_path}')
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return np.array(frames)


def detect_face(frame, detector, frame_idx):
    """Detect face landmarks using MediaPipe."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, frame_idx)
    if result and result.face_landmarks:
        lm = result.face_landmarks[0]
        h, w = frame.shape[:2]
        return np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
    return None


def process_frame(frame, detector, frame_idx, prev_landmarks=None, min_size=64):
    """Process single frame: detect face, crop, resize."""
    landmarks = detect_face(frame, detector, frame_idx)
    if landmarks is None:
        return None, prev_landmarks if prev_landmarks is not None else None
    
    x_min, x_max = landmarks[:, 0].min(), landmarks[:, 0].max()
    y_min, y_max = landmarks[:, 1].min(), landmarks[:, 1].max()
    w, h = x_max - x_min, y_max - y_min
    
    if w < min_size or h < min_size:
        return None, landmarks
    
    pad = 20
    x_min = max(0, int(x_min) - pad)
    x_max = min(frame.shape[1], int(x_max) + pad)
    y_min = max(0, int(y_min) - pad)
    y_max = min(frame.shape[0], int(y_max) + pad)
    
    face = frame[y_min:y_max, x_min:x_max]
    return cv2.resize(face, (128, 128)), landmarks


def extract_chunks(video, ppg, window_size, stride, video_fps=30.0, ppg_fps=100.0):
    """Extract chunks from video and PPG."""
    chunks, ppg_chunks = [], []
    num_frames = video.shape[0]
    ppg_per_frame = ppg_fps / video_fps
    
    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        video_chunk = video[start:end]
        ppg_start = int(start * ppg_per_frame)
        ppg_end = int(end * ppg_per_frame)
        ppg_chunk = ppg[ppg_start:ppg_end]
        if len(ppg_chunk) == window_size:
            chunks.append(video_chunk)
            ppg_chunks.append(ppg_chunk)
    
    return np.array(chunks), np.array(ppg_chunks)


def test_single_video(row, detector, args):
    """
    Test preprocessing on a single video with full visualization.
    
    This function:
    1. Loads all files for the given row
    2. Processes the video
    3. Aligns PPG with video
    4. Creates visualizations
    5. Returns test results
    
    Args:
        row: Database row
        detector: MediaPipe detector
        args: Command line arguments
    
    Returns:
        dict: Test results with visualizations
    """
    results = {
        'patient_id': row['patient_id'],
        'step': row['step'],
        'camera': row['camera'],
        'view': row['view'],
        'success': False,
        'error': None,
        'video_info': {},
        'ppg_info': {},
        'alignment_info': {},
        'vitals': {}
    }
    
    try:
        # Get file paths
        video_path = os.path.join(args.dataset_root, row['video'])
        ppg_path = os.path.join(args.dataset_root, row['ppg'])
        meta_path = os.path.join(args.dataset_root, row['meta'])
        ppg_sync_path = os.path.join(args.dataset_root, row['ppg_sync'])
        
        # Verify files exist
        missing = [f for f in [video_path, ppg_path, meta_path] if not os.path.exists(f)]
        if missing:
            results['error'] = f'Missing files: {missing}'
            return results
        
        print(f"\n{'='*80}")
        print(f"Testing: Patient {row['patient_id']}, Step: {row['step']}, Camera: {row['camera']}, View: {row['view']}")
        print(f"{'='*80}")
        
        # Load video
        print("Loading video...")
        video = load_video(video_path)
        results['video_info']['original_frames'] = len(video)
        results['video_info']['shape'] = video.shape
        print(f"  Video: {video.shape[0]} frames, {video.shape[1]}x{video.shape[2]}, {video.shape[3]} channels")
        
        # Load PPG
        print("Loading PPG...")
        ppg_values, ppg_timestamps = parse_pw_file(ppg_path)
        results['ppg_info']['original_length'] = len(ppg_values)
        results['ppg_info']['duration_seconds'] = (ppg_timestamps[-1] - ppg_timestamps[0]).total_seconds()
        print(f"  PPG: {len(ppg_values)} values, duration: {results['ppg_info']['duration_seconds']:.2f}s")
        
        # Load meta
        print("Loading meta...")
        meta_data = parse_meta_file(meta_path)
        results['video_info']['meta_frames'] = len(meta_data)
        if len(meta_data) > 0:
            results['video_info']['meta_duration_seconds'] = (list(meta_data.values())[-1] - list(meta_data.values())[0]).total_seconds()
            print(f"  Meta: {len(meta_data)} frames, duration: {results['video_info']['meta_duration_seconds']:.2f}s")
        
        # Load PPG sync
        if os.path.exists(ppg_sync_path):
            ppg_sync_data = parse_ppg_sync_file(ppg_sync_path)
            results['alignment_info']['ppg_sync_frames'] = len(ppg_sync_data)
            print(f"  PPG Sync: {len(ppg_sync_data)} entries")
        
        # Align PPG with video
        print("Aligning PPG with video...")
        aligned_ppg = align_ppg_with_video(ppg_values, ppg_timestamps, meta_data, args.frame_rate)
        results['alignment_info']['aligned_ppg_length'] = len(aligned_ppg)
        print(f"  Aligned PPG: {len(aligned_ppg)} values")
        
        # Check alignment
        video_duration = len(video) / args.frame_rate
        ppg_duration = results['ppg_info']['duration_seconds']
        meta_duration = results['video_info'].get('meta_duration_seconds', 0)
        
        results['alignment_info']['video_duration'] = video_duration
        results['alignment_info']['ppg_duration'] = ppg_duration
        results['alignment_info']['meta_duration'] = meta_duration
        results['alignment_info']['duration_match'] = abs(video_duration - meta_duration) < 0.1
        
        print(f"\n  Duration check:")
        print(f"    Video: {video_duration:.2f}s")
        print(f"    PPG: {ppg_duration:.2f}s")
        print(f"    Meta: {meta_duration:.2f}s")
        print(f"    Meta matches video: {results['alignment_info']['duration_match']}")
        
        # Preprocess PPG
        processed_ppg = preprocess_ppg(aligned_ppg, args.ppg_rate)
        results['ppg_info']['processed_length'] = len(processed_ppg)
        
        # Process video frames
        print("Processing video frames...")
        processed_frames = []
        all_landmarks = []
        prev_landmarks = None
        
        for fi, frame in enumerate(video):
            pf, lms = process_frame(frame, detector, fi, prev_landmarks, args.min_face_size if hasattr(args, 'min_face_size') else 64)
            if pf is not None:
                processed_frames.append(pf)
                all_landmarks.append(lms)
                prev_landmarks = lms
        
        results['video_info']['processed_frames'] = len(processed_frames)
        print(f"  Processed frames: {len(processed_frames)}/{len(video)}")
        
        if len(processed_frames) == 0:
            results['error'] = 'No frames processed'
            return results
        
        # Extract chunks
        video_chunks, ppg_chunks = extract_chunks(
            np.array(processed_frames), processed_ppg,
            args.window_size, args.stride,
            args.frame_rate, args.ppg_rate
        )
        results['video_info']['num_chunks'] = len(video_chunks)
        results['ppg_info']['num_chunks'] = len(ppg_chunks)
        print(f"  Chunks extracted: {len(video_chunks)}")
        
        # Save vitals
        vitals_cols = ['weight', 'height', 'bmi', 'age', 'sex', 'upper_ap', 'lower_ap', 
                      'saturation', 'temperature', 'pulse', 'stress']
        for col in vitals_cols:
            if col in row:
                results['vitals'][col] = row[col]
        
        # Create visualizations
        print("\n  Creating visualizations...")
        
        # Create output directory for this test
        test_dir = os.path.join(args.output_path, f"test_{row['patient_id']}_{row['camera']}_{row['step']}")
        os.makedirs(test_dir, exist_ok=True)
        
        # Visualization 1: Original frame with landmarks
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original frame
        axes[0].imshow(video[0])
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        # Processed face
        axes[1].imshow(processed_frames[0])
        axes[1].set_title('Preprocessed Face')
        axes[1].axis('off')
        
        # Landmarks on original
        frame_with_lms = video[0].copy()
        lms = all_landmarks[0]
        for i, (x, y) in enumerate(lms):
            if i % 10 == 0:  # Plot every 10th landmark for clarity
                cv2.circle(frame_with_lms, (int(x), int(y)), 2, (0, 255, 0), -1)
        axes[2].imshow(frame_with_lms)
        axes[2].set_title('Landmarks Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f"Patient {row['patient_id']}, Camera: {row['camera']}, Step: {row['step']}")
        plt.tight_layout()
        plt.savefig(os.path.join(test_dir, 'frames_comparison.png'), dpi=100, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {test_dir}/frames_comparison.png")
        
        # Visualization 2: PPG signal with chunk
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Full PPG signal
        axes[0].plot(processed_ppg, label='Processed PPG', color='blue')
        axes[0].set_title('Full PPG Signal')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        axes[0].legend()
        
        # First PPG chunk
        if len(ppg_chunks) > 0:
            axes[1].plot(ppg_chunks[0], label='First PPG Chunk', color='red')
            axes[1].set_title(f'First PPG Chunk (Window: {args.window_size} samples)')
            axes[1].set_xlabel('Sample')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True)
            axes[1].legend()
        
        plt.suptitle(f"PPG Signal - Patient {row['patient_id']}")
        plt.tight_layout()
        plt.savefig(os.path.join(test_dir, 'ppg_signal.png'), dpi=100, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {test_dir}/ppg_signal.png")
        
        # Visualization 3: Video chunk frames
        if len(video_chunks) > 0:
            fig, axes = plt.subplots(1, min(5, video_chunks[0].shape[0]), figsize=(20, 4))
            for i, ax in enumerate(axes):
                ax.imshow(video_chunks[0][i])
                ax.set_title(f'Frame {i}')
                ax.axis('off')
            plt.suptitle(f'First Video Chunk - Patient {row["patient_id"]}')
            plt.tight_layout()
            plt.savefig(os.path.join(test_dir, 'video_chunk.png'), dpi=100, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {test_dir}/video_chunk.png")
        
        # Save test results
        results['success'] = True
        results['test_dir'] = test_dir
        
        # Print summary
        print(f"\n  Test Summary:")
        print(f"    Video: {results['video_info']['original_frames']} frames -> {results['video_info']['processed_frames']} processed")
        print(f"    PPG: {results['ppg_info']['original_length']} values -> {results['ppg_info']['aligned_ppg_length']} aligned")
        print(f"    Chunks: {results['video_info']['num_chunks']} video, {results['ppg_info']['num_chunks']} PPG")
        print(f"    Vitals: {len(results['vitals'])} parameters")
        print(f"    Visualizations saved to: {test_dir}")
        
        # Verify alignment matches original preprocessing
        print(f"\n  Alignment Verification:")
        print(f"    Video duration: {video_duration:.2f}s")
        print(f"    PPG duration: {ppg_duration:.2f}s")
        print(f"    Meta duration: {meta_duration:.2f}s")
        print(f"    Duration match (video vs meta): {results['alignment_info']['duration_match']}")
        print(f"    PPG/Video ratio: {len(ppg_values)/len(video):.2f} (should be ~{args.ppg_rate/args.frame_rate:.2f})")
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        import traceback
        traceback.print_exc()
        return results


def print_test_summary(results):
    """Print a summary of test results."""
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for i, result in enumerate(results):
        print(f"\nTest {i+1}:")
        print(f"  Patient: {result.get('patient_id', 'N/A')}")
        print(f"  Step: {result.get('step', 'N/A')}")
        print(f"  Camera: {result.get('camera', 'N/A')}")
        print(f"  View: {result.get('view', 'N/A')}")
        print(f"  Success: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"  Video: {result['video_info'].get('original_frames', 0)} frames")
            print(f"    -> Processed: {result['video_info'].get('processed_frames', 0)} frames")
            print(f"    -> Chunks: {result['video_info'].get('num_chunks', 0)}")
            print(f"  PPG: {result['ppg_info'].get('original_length', 0)} values")
            print(f"    -> Aligned: {result['alignment_info'].get('aligned_ppg_length', 0)} values")
            print(f"    -> Chunks: {result['ppg_info'].get('num_chunks', 0)}")
            print(f"  Alignment:")
            print(f"    Video duration: {result['alignment_info'].get('video_duration', 0):.2f}s")
            print(f"    PPG duration: {result['alignment_info'].get('ppg_duration', 0):.2f}s")
            print(f"    Meta duration: {result['alignment_info'].get('meta_duration', 0):.2f}s")
            print(f"    Duration match: {result['alignment_info'].get('duration_match', False)}")
            print(f"  Vitals: {len(result.get('vitals', {}))} parameters")
            print(f"  Output: {result.get('test_dir', 'N/A')}")
        else:
            print(f"  Error: {result.get('error', 'Unknown')}")


def main():
    """Main test function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load database
    print("Loading database...")
    db = pd.read_csv(args.db_path)
    print(f"Loaded {len(db)} rows from {args.db_path}")
    
    # Initialize MediaPipe
    detector = initialize_mediapipe()
    
    # Select random videos for testing
    print(f"\nSelecting {args.num_tests} random videos for testing...")
    test_rows = db.sample(n=min(args.num_tests, len(db)))
    print(f"Testing videos:")
    for _, row in test_rows.iterrows():
        print(f"  - Patient {row['patient_id']}, Camera: {row['camera']}, Step: {row['step']}")
    
    # Test each video
    all_results = []
    for _, row in test_rows.iterrows():
        results = test_single_video(row, detector, args)
        all_results.append(results)
    
    # Print summary
    print_test_summary(all_results)
    
    # Check if all tests passed
    all_success = all(r.get('success', False) for r in all_results)
    if all_success:
        print(f"\n✅ All {len(all_results)} tests passed!")
    else:
        print(f"\n⚠️ {sum(1 for r in all_results if r.get('success'))}/{len(all_results)} tests passed")
    
    # Verify alignment consistency
    print(f"\n{'='*80}")
    print("ALIGNMENT VERIFICATION")
    print(f"{'='*80}")
    
    for i, result in enumerate(all_results):
        if result.get('success'):
            print(f"\nTest {i+1}:")
            print(f"  Video frames: {result['video_info'].get('original_frames', 0)}")
            print(f"  Meta frames: {result['video_info'].get('meta_frames', 0)}")
            print(f"  Video duration: {result['alignment_info'].get('video_duration', 0):.2f}s")
            print(f"  Meta duration: {result['alignment_info'].get('meta_duration', 0):.2f}s")
            print(f"  Duration match: {result['alignment_info'].get('duration_match', False)}")
            
            # Check PPG/Video ratio
            video_frames = result['video_info'].get('original_frames', 0)
            ppg_values = result['ppg_info'].get('original_length', 0)
            if video_frames > 0:
                ratio = ppg_values / video_frames
                expected_ratio = args.ppg_rate / args.frame_rate
                print(f"  PPG/Video ratio: {ratio:.2f} (expected: {expected_ratio:.2f})")
                print(f"  Ratio match: {abs(ratio - expected_ratio) < 0.1}")
    
    print(f"\n{'='*80}")
    print("Test output saved to:", args.output_path)
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

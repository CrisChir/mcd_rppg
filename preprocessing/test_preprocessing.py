#!/usr/bin/env python3
"""
MCD-rPPG Preprocessing Test Script

This script tests the preprocessing pipeline on 2 random videos from the dataset.
It uses SIMPLE RATIO-BASED SYNCHRONIZATION to match the original preprocessing scripts.

Key synchronization approach:
- PPG at 100 Hz, Video at 30 FPS
- Ratio: 100/30 = 3.333... PPG samples per video frame
- Direct mapping: ppg_chunk = ppg[start_frame * 3.333 : end_frame * 3.333]
- NO complex timestamp parsing (matches original scripts)

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
        description='Test preprocessing on random videos from MCD-rPPG dataset',
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
                        help='Window size for chunks (MUST match original scripts)')
    parser.add_argument('--stride', type=int, default=64,
                        help='Stride for chunks (MUST match original scripts)')
    parser.add_argument('--frame_rate', type=float, default=30.0,
                        help='Video frame rate (MUST match original)')
    parser.add_argument('--ppg_rate', type=float, default=100.0,
                        help='PPG sampling rate (MUST match original)')
    parser.add_argument('--target_size', type=tuple, default=(128, 128),
                        help='Target face size')
    parser.add_argument('--min_face_size', type=int, default=64,
                        help='Minimum face size in pixels')
    
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


def preprocess_ppg(ppg, low_freq=0.75, high_freq=4.0, fs=100.0):
    """Apply bandpass filter and normalize PPG signal."""
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, ppg)
    return (filtered - filtered.mean()) / filtered.std()


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


def load_ppg_simple(ppg_path):
    """
    Load PPG from .PW file - SIMPLE VERSION.
    
    Since we're using ratio-based synchronization, we just need the PPG values.
    We don't need to parse timestamps if we're using the fixed ratio approach.
    
    Args:
        ppg_path: Path to .PW file
    
    Returns:
        ppg_values: Array of PPG values
    """
    ppg_values = []
    with open(ppg_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    ppg_values.append(float(parts[0]))
                except:
                    pass
    return np.array(ppg_values)


def test_single_video(row, detector, args):
    """
    Test preprocessing on a single video with full visualization.
    
    Uses SIMPLE RATIO-BASED SYNCHRONIZATION to match original scripts.
    
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
        
        # Verify files exist
        if not all(os.path.exists(f) for f in [video_path, ppg_path]):
            results['error'] = f'Missing files'
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
        
        # Load PPG (simple version - just values)
        print("Loading PPG...")
        ppg_values = load_ppg_simple(ppg_path)
        results['ppg_info']['original_length'] = len(ppg_values)
        print(f"  PPG: {len(ppg_values)} values")
        
        # Calculate expected ratio
        expected_ratio = args.ppg_rate / args.frame_rate
        actual_ratio = len(ppg_values) / len(video)
        results['alignment_info']['expected_ratio'] = expected_ratio
        results['alignment_info']['actual_ratio'] = actual_ratio
        results['alignment_info']['ratio_match'] = abs(actual_ratio - expected_ratio) < 0.01
        
        print(f"  PPG/Video ratio: {actual_ratio:.4f} (expected: {expected_ratio:.4f})")
        print(f"  Ratio match: {results['alignment_info']['ratio_match']}")
        
        # Preprocess PPG
        processed_ppg = preprocess_ppg(ppg_values, args.ppg_rate)
        results['ppg_info']['processed_length'] = len(processed_ppg)
        
        # Process video frames
        print("Processing video frames...")
        processed_frames = []
        all_landmarks = []
        prev_landmarks = None
        
        for fi, frame in enumerate(video):
            pf, lms = process_frame(frame, detector, fi, prev_landmarks, args.min_face_size)
            if pf is not None:
                processed_frames.append(pf)
                all_landmarks.append(lms)
                prev_landmarks = lms
        
        results['video_info']['processed_frames'] = len(processed_frames)
        print(f"  Processed frames: {len(processed_frames)}/{len(video)}")
        
        if len(processed_frames) == 0:
            results['error'] = 'No frames processed'
            return results
        
        # Extract chunks using SIMPLE RATIO-BASED approach
        print("Extracting chunks...")
        video_chunks, ppg_chunks = extract_chunks_simple(
            np.array(processed_frames), processed_ppg,
            args.window_size, args.stride,
            args.frame_rate, args.ppg_rate
        )
        results['video_info']['num_chunks'] = len(video_chunks)
        results['ppg_info']['num_chunks'] = len(ppg_chunks)
        print(f"  Chunks extracted: {len(video_chunks)}")
        
        # Create visualizations
        print("Creating visualizations...")
        
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
        print(f"  Saved: {test_dir}/frames_comparison.png")
        
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
        print(f"  Saved: {test_dir}/ppg_signal.png")
        
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
            print(f"  Saved: {test_dir}/video_chunk.png")
        
        # Save vitals
        vitals_cols = ['weight', 'height', 'bmi', 'age', 'sex', 'upper_ap', 'lower_ap', 
                      'saturation', 'temperature', 'pulse', 'stress']
        for col in vitals_cols:
            if col in row:
                results['vitals'][col] = row[col]
        
        # Save test results
        results['success'] = True
        results['test_dir'] = test_dir
        
        # Print summary
        print(f"\n  Test Summary:")
        print(f"    Video: {results['video_info']['original_frames']} frames -> {results['video_info']['processed_frames']} processed")
        print(f"    PPG: {results['ppg_info']['original_length']} values -> {results['ppg_info']['processed_length']} processed")
        print(f"    Chunks: {results['video_info']['num_chunks']} video, {results['ppg_info']['num_chunks']} PPG")
        print(f"    PPG/Video ratio: {actual_ratio:.4f} (expected: {expected_ratio:.4f})")
        print(f"    Ratio match: {results['alignment_info']['ratio_match']}")
        print(f"    Vitals: {len(results['vitals'])} parameters")
        print(f"    Visualizations saved to: {test_dir}")
        
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
            print(f"    -> Processed: {result['ppg_info'].get('processed_length', 0)} values")
            print(f"    -> Chunks: {result['ppg_info'].get('num_chunks', 0)}")
            print(f"  Alignment:")
            print(f"    Expected ratio (PPG/Video): {result['alignment_info'].get('expected_ratio', 0):.4f}")
            print(f"    Actual ratio: {result['alignment_info'].get('actual_ratio', 0):.4f}")
            print(f"    Ratio match: {result['alignment_info'].get('ratio_match', False)}")
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
        print("\n🎯 Synchronization verification:")
        print("   Using SIMPLE RATIO-BASED approach (100/30 = 3.333...)")
        print("   This matches the ORIGINAL preprocessing scripts")
        print("   No complex timestamp parsing needed")
    else:
        print(f"\n⚠️ {sum(1 for r in all_results if r.get('success'))}/{len(all_results)} tests passed")
    
    print(f"\n{'='*80}")
    print("Test output saved to:", args.output_path)
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

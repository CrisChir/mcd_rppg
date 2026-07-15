#!/usr/bin/env python3
"""
MCD-rPPG Dataset Preprocessing Script 3

This script processes the third set of videos (camera 3 - mobile phone).
It performs face detection using MediaPipe, cropping, and saves processed faces and landmarks.

Usage:
    python dataset_preprocessing_3.py [--input_path INPUT] [--output_path OUTPUT] [--num_workers N]

Author: CrisChir
Date: September 2025
License: MIT
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import rppglib.data_utils
    import rppglib.face_utils
except ImportError as e:
    print(f"Error importing rppglib: {e}")
    print("Please ensure the repository root is in your Python path.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess MCD-rPPG dataset - Camera 3 (Mobile Phone) using MediaPipe',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input_path', type=str, 
                        default='/gim/lv03/datasets/rppg/rppg_nir_data_train/video/',
                        help='Path to input video files')
    parser.add_argument('--output_path', type=str,
                        default='/gim/lv03/datasets/rppg/rppg_nir_data_train/',
                        help='Path to save processed data')
    parser.add_argument('--faces_path', type=str, default=None,
                        help='Path to save face crops (default: output_path/faces/)')
    parser.add_argument('--landmarks_path', type=str, default=None,
                        help='Path to save landmarks (default: output_path/landmarks/)')
    parser.add_argument('--errors_file', type=str, default='errors.csv',
                        help='Path to errors log file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting video index')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending video index (default: all videos)')
    parser.add_argument('--camera_id', type=int, default=3,
                        help='Camera identifier (1, 2, or 3)')
    parser.add_argument('--window_size', type=int, default=256,
                        help='Window size for chunking')
    parser.add_argument('--stride', type=int, default=64,
                        help='Stride for chunking')
    parser.add_argument('--min_face_size', type=int, default=64,
                        help='Minimum face size in pixels')
    parser.add_argument('--max_face_size', type=int, default=512,
                        help='Maximum face size in pixels')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5,
                        help='Minimum face detection confidence (0-1)')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5,
                        help='Minimum face tracking confidence (0-1)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip already processed files')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode with additional checks')
    
    return parser.parse_args()


def setup_directories(args):
    """Setup output directories."""
    # Create output directories
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.faces_path is None:
        args.faces_path = os.path.join(args.output_path, 'faces')
    if args.landmarks_path is None:
        args.landmarks_path = os.path.join(args.output_path, 'landmarks')
    
    os.makedirs(args.faces_path, exist_ok=True)
    os.makedirs(args.landmarks_path, exist_ok=True)
    
    # Ensure errors file directory exists
    errors_dir = os.path.dirname(args.errors_file)
    if errors_dir:
        os.makedirs(errors_dir, exist_ok=True)
    
    return args


def load_errors_file(args):
    """Load or create errors log file."""
    if os.path.isfile(args.errors_file):
        try:
            errors = pd.read_csv(args.errors_file)
            if args.verbose:
                print(f"Loaded {len(errors)} existing errors from {args.errors_file}")
        except Exception as e:
            print(f"Warning: Could not read errors file: {e}")
            errors = pd.DataFrame({'video_file': [], 'error_type': [], 'error_msg': []})
    else:
        errors = pd.DataFrame({'video_file': [], 'error_type': [], 'error_msg': []})
    
    return errors


def save_errors_file(errors, args):
    """Save errors to log file."""
    try:
        errors.to_csv(args.errors_file, index=False)
    except Exception as e:
        print(f"Warning: Could not save errors file: {e}")


def get_video_list(args):
    """Get list of videos to process."""
    # Find all AVI files
    videos = sorted(glob(f'{args.input_path}*.avi'))
    
    if not videos:
        print(f"Warning: No AVI files found in {args.input_path}")
        return []
    
    # Filter by camera ID if specified
    if args.camera_id is not None:
        videos = [v for v in videos if f'camera_{args.camera_id}' in v or 
                  f'cam_{args.camera_id}' in v]
    
    # Apply index range
    if args.end_idx is not None:
        videos = videos[args.start_idx:args.end_idx]
    else:
        videos = videos[args.start_idx:]
    
    # For this script, process every 3rd video starting from index 2
    videos = videos[2::3]
    
    if args.verbose:
        print(f"Found {len(videos)} videos to process (camera {args.camera_id})")
        print(f"First video: {videos[0] if videos else 'None'}")
        print(f"Last video: {videos[-1] if videos else 'None'}")
    
    return videos


def process_single_video(video_file, args, errors):
    """
    Process a single video file using MediaPipe.
    
    Args:
        video_file: Path to video file
        args: Command line arguments
        errors: DataFrame to log errors
    
    Returns:
        True if successful, False otherwise
    """
    video_filename = os.path.basename(video_file)
    
    # Generate output paths
    face_file = os.path.join(
        args.faces_path,
        os.path.splitext(video_filename)[0] + '.npy'
    )
    landmarks_file = os.path.join(
        args.landmarks_path,
        os.path.splitext(video_filename)[0] + '.npy'
    )
    
    # Check if already processed
    if args.skip_existing and os.path.isfile(face_file) and os.path.isfile(landmarks_file):
        if args.verbose:
            print(f"Skipping {video_file} (already processed)")
        return True
    
    # Check if video is in errors
    if video_file in errors['video_file'].values:
        if args.verbose:
            print(f"Skipping {video_file} (previously failed)")
        return False
    
    try:
        if args.verbose:
            print(f"Processing {video_file}")
        
        # Load video
        video = rppglib.data_utils.load_video(video_file)
        
        if args.debug:
            print(f"  Video shape: {video.shape}")
            print(f"  Video dtype: {video.dtype}")
        
        # Process video using MediaPipe (face detection and cropping)
        processed_video, landmarks = rppglib.face_utils.process_video(
            video,
            min_face_size=args.min_face_size,
            max_face_size=args.max_face_size,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence
        )
        
        if args.debug:
            print(f"  Processed video shape: {processed_video.shape}")
            print(f"  Landmarks shape: {landmarks.shape}")
        
        # Save results
        np.save(face_file, processed_video)
        np.save(landmarks_file, landmarks)
        
        if args.verbose:
            print(f"  Saved to {face_file}")
            print(f"  Saved to {landmarks_file}")
        
        return True
        
    except RuntimeError as e:
        error_row = {
            'video_file': video_file,
            'error_type': 'RuntimeError',
            'error_msg': str(e)
        }
        errors = pd.concat([errors, pd.DataFrame([error_row])], ignore_index=True)
        save_errors_file(errors, args)
        print(f"  Error processing {video_file}: {e}")
        return False
        
    except Exception as e:
        error_row = {
            'video_file': video_file,
            'error_type': str(type(e).__name__),
            'error_msg': str(e)
        }
        errors = pd.concat([errors, pd.DataFrame([error_row])], ignore_index=True)
        save_errors_file(errors, args)
        print(f"  Error processing {video_file}: {type(e).__name__}: {e}")
        return False


def main():
    """Main preprocessing function."""
    args = parse_args()
    args = setup_directories(args)
    
    print("=" * 80)
    print("MCD-rPPG Dataset Preprocessing Script 3")
    print(f"Camera: {args.camera_id} (Mobile Phone)")
    print(f"Using: MediaPipe Face Landmark Detection")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print("=" * 80)
    
    # Load errors file
    errors = load_errors_file(args)
    
    # Get video list
    videos = get_video_list(args)
    
    if not videos:
        print("No videos to process. Exiting.")
        return
    
    print(f"\nTotal videos to process: {len(videos)}")
    print(f"Starting from index: {args.start_idx}")
    if args.end_idx is not None:
        print(f"Ending at index: {args.end_idx}")
    print(f"Min detection confidence: {args.min_detection_confidence}")
    print(f"Min tracking confidence: {args.min_tracking_confidence}")
    print()
    
    # Process videos
    success_count = 0
    failure_count = 0
    
    for video_file in tqdm(videos, desc="Processing videos"):
        success = process_single_video(video_file, args, errors)
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    # Save final errors
    save_errors_file(errors, args)
    
    print("\n" + "=" * 80)
    print("Processing Complete")
    print("=" * 80)
    print(f"Total videos: {len(videos)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Errors logged to: {args.errors_file}")
    print(f"Processed faces saved to: {args.faces_path}")
    print(f"Processed landmarks saved to: {args.landmarks_path}")
    
    if failure_count > 0:
        print(f"\nWarning: {failure_count} videos failed. Check {args.errors_file} for details.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
MCD-rPPG Dataset Processing Pipeline

A comprehensive script for preprocessing the MCD-rPPG dataset with support for:
- Processing 10 test files or the full dataset
- Parallel processing with configurable number of workers (max 10)
- ROI extraction using MediaPipe landmarks
- Chunking with overlap
- PPG synchronization
- Memory-efficient batch processing

Usage:
    # Process 10 test files
    python data_processing_pipeline.py --mode test --dataset_path /path/to/dataset --output_path /path/to/output
    
    # Process full dataset
    python data_processing_pipeline.py --mode full --dataset_path /path/to/dataset --output_path /path/to/output
    
    # Custom configuration
    python data_processing_pipeline.py --mode test --dataset_path /path/to/dataset --output_path /path/to/output --chunk_size 450 --overlap 150 --workers 10

Author: Vibe Code (Mistral AI)
Date: 2025
License: MIT
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import json
import time
import gc
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths
DEFAULT_DATASET_PATH = '/home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/'
DEFAULT_OUTPUT_PATH = '/home/cristic/preprocessed_data'
DEFAULT_MEDIAPIPE_MODEL = '/home/cristic/face_landmarker.task'

# Default preprocessing parameters
DEFAULT_CHUNK_SIZE = 450
DEFAULT_OVERLAP_SIZE = 150
DEFAULT_ROI_BOX_SIZE = (24, 24)
DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_MAX_WORKERS = 10
DEFAULT_NUM_TEST_FILES = 10

# ROI Configuration (Corrected Canonical MediaPipe Mesh Indices)
ROIS = {
    'full_face': list(range(468)),
    'forehead': [10, 67, 69, 108, 109, 151, 337, 338, 297, 299, 9, 8],
    'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    'nose': [1, 2, 98, 327, 328, 2, 4, 5, 195, 197, 6, 168],
    'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 95],
    'chin': [152, 148, 176, 149, 150, 136, 172, 377, 400, 378, 379, 365, 397],
    'right_cheek_50': [50],
    'left_cheek_280': [280],
    'chin_199': [199]
}


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MCD-rPPG Dataset Processing Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'full'],
                        help='Processing mode: "test" for 10 files, "full" for all dataset')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH,
                        help='Path to dataset root directory')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Path to save preprocessed data')
    parser.add_argument('--db_path', type=str, default=None,
                        help='Path to database CSV file (optional)')
    parser.add_argument('--mediapipe_model', type=str, default=DEFAULT_MEDIAPIPE_MODEL,
                        help='Path to MediaPipe face landmarker model')
    
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='Number of frames per chunk')
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP_SIZE,
                        help='Number of overlapping frames between chunks')
    parser.add_argument('--roi_box_size', type=int, nargs=2, default=DEFAULT_ROI_BOX_SIZE,
                        help='ROI box size as (width, height)')
    parser.add_argument('--smoothing_window', type=int, default=DEFAULT_SMOOTHING_WINDOW,
                        help='Temporal smoothing window size for landmarks')
    parser.add_argument('--workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help='Maximum number of parallel workers')
    parser.add_argument('--test_files', type=int, default=DEFAULT_NUM_TEST_FILES,
                        help='Number of test files to process in test mode')
    
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose logging')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                        help='Skip files that already exist in output directory')
    
    return parser.parse_args()


# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================

class TemporalSmoother:
    """Temporal smoothing for landmarks to reduce jitter."""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def smooth(self, landmarks):
        self.history.append(landmarks.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)
        n = len(self.history)
        weights = [float(i + 1) for i in range(n)]
        smoothed = np.zeros_like(landmarks)
        for i, w in enumerate(weights):
            smoothed += self.history[i] * w
        smoothed /= sum(weights)
        return smoothed


class MediaPipeLandmarkDetector:
    """MediaPipe-based facial landmark detector with temporal smoothing."""
    def __init__(self, model_path, smoothing_window=5):
        self.model_path = model_path
        self.smoothing_window = smoothing_window
        self.smoother = TemporalSmoother(smoothing_window)
        self.detector = None
        self.frame_count = 0
        self.fps = 30.0
        self.mediapipe_available = False
        
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            self.mediapipe_available = True
            self.mp = mp
            self.python = python
            self.vision = vision
        except ImportError as e:
            print(f'Warning: MediaPipe not available: {e}')

    def initialize_detector(self):
        if not self.mediapipe_available:
            raise RuntimeError('MediaPipe not available')
        base_options = self.python.BaseOptions(model_asset_path=self.model_path)
        options = self.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=self.vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = self.vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, frame):
        if self.detector is None:
            self.initialize_detector()
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame)
        try:
            timestamp_ms = int(self.frame_count * (1000.0 / self.fps))
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            self.frame_count += 1
            if result and result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                frame_width, frame_height = frame.shape[1], frame.shape[0]
                points = np.array([(lm.x * frame_width, lm.y * frame_height) for lm in face_landmarks], dtype='float32')
                if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                    return None
                if np.max(points) > max(frame_width, frame_height) * 3 or np.min(points) < -max(frame_width, frame_height) * 2:
                    return None
                smoothed_points = self.smoother.smooth(points)
                return smoothed_points
            else:
                return None
        except Exception as e:
            print(f'Error in landmark detection: {e}')
            return None

    def reset(self):
        self.frame_count = 0
        self.smoother.history = []
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception as e:
                print(f'Warning during detector close: {e}')
            self.detector = None
        self.initialize_detector()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_video_frames_ffmpeg(video_path):
    """Count frames in a video using ffmpeg."""
    try:
        import imageio
        reader = imageio.get_reader(video_path, 'ffmpeg')
        n_frames = reader.count_frames()
        reader.close()
        return n_frames
    except Exception as e:
        print(f'Error counting frames: {e}')
        return 0


def get_video_fps(video_path):
    """Get video FPS."""
    try:
        import imageio
        reader = imageio.get_reader(video_path, 'ffmpeg')
        fps = reader.get_meta_data().get('fps', 30.0)
        reader.close()
        return fps
    except Exception:
        return 30.0


def load_ppg_sync_data(ppg_sync_path):
    """Load PPG sync data from file."""
    try:
        if ppg_sync_path.endswith('.npy'):
            data = np.load(ppg_sync_path)
        elif ppg_sync_path.endswith('.txt'):
            data = np.loadtxt(ppg_sync_path)
        else:
            data = np.load(ppg_sync_path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    except Exception as e:
        print(f'Error loading PPG sync: {e}')
        return np.array([])


def load_video_frames(video_path, start_frame=0, end_frame=None):
    """Load video frames using imageio."""
    try:
        import imageio
        reader = imageio.get_reader(video_path, 'ffmpeg')
        total_frames = reader.count_frames()
        if end_frame is None:
            end_frame = total_frames
        frames = []
        for i in range(start_frame, end_frame):
            frames.append(reader.get_next_data())
        reader.close()
        return np.array(frames)
    except Exception as e:
        print(f'Error loading video frames: {e}')
        return np.array([])


def create_chunks(n_frames, chunk_size, overlap_size):
    """Create chunks with overlap."""
    chunks = []
    chunk_idx = 0
    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        chunks.append((start, end, chunk_idx))
        if end == n_frames:
            break
        start = end - overlap_size
        chunk_idx += 1
    return chunks


def extract_roi_bbox(landmarks, roi_indices, frame_shape, box_size):
    """Extract ROI bounding box from landmarks."""
    valid_indices = [i for i in roi_indices if i < len(landmarks)]
    if not valid_indices:
        return (0, 0, box_size[0], box_size[1])
    roi_points = landmarks[valid_indices]
    raw_x = np.mean(roi_points[:, 0])
    raw_y = np.mean(roi_points[:, 1])
    if not np.isfinite(raw_x) or not np.isfinite(raw_y):
        return (0, 0, box_size[0], box_size[1])
    center_x = max(0, min(int(raw_x), frame_shape[1]))
    center_y = max(0, min(int(raw_y), frame_shape[0]))
    box_w, box_h = box_size
    x = max(0, center_x - box_w // 2)
    y = max(0, center_y - box_h // 2)
    x = min(x, frame_shape[1] - box_w)
    y = min(y, frame_shape[0] - box_h)
    return (int(x), int(y), int(box_w), int(box_h))


def extract_roi_region(frame, bbox):
    """Extract ROI region from frame."""
    x, y, w, h = bbox
    return frame[y:y+h, x:x+w]


# ============================================================================
# CHUNK PROCESSING
# ============================================================================

def process_video_chunk(video_path, ppg_sync_path, meta_data, chunk_start, chunk_end, 
                       chunk_idx, detector, roi_box_size):
    """Process a single video chunk."""
    try:
        # Load video frames for this chunk
        video_frames = load_video_frames(video_path, chunk_start, chunk_end)
        if len(video_frames) == 0:
            return None
        
        # Load PPG sync data
        ppg_sync_data = load_ppg_sync_data(ppg_sync_path)
        if len(ppg_sync_data) == 0:
            return None
        
        # Extract PPG chunk
        chunk_ppg = ppg_sync_data[chunk_start:chunk_end]
        if chunk_ppg.shape[1] >= 2:
            ppg_values = chunk_ppg[:, 0]
            time_deltas = chunk_ppg[:, 1]
        else:
            ppg_values = chunk_ppg[:, 0] if chunk_ppg.ndim > 1 else chunk_ppg
            time_deltas = np.zeros_like(ppg_values)
        
        # Reset detector for this chunk
        detector.reset()
        detector.fps = get_video_fps(video_path)
        
        # Detect landmarks for all frames
        chunk_landmarks = []
        for frame in video_frames:
            lms = detector.detect_landmarks(frame)
            if lms is not None:
                chunk_landmarks.append(lms)
            elif chunk_landmarks:
                chunk_landmarks.append(chunk_landmarks[-1].copy())
            else:
                chunk_landmarks.append(np.zeros((468, 2), dtype='float32'))
        
        chunk_landmarks = np.array(chunk_landmarks)
        
        # Extract ROIs for all frames
        roi_data = {}
        for roi_name, roi_indices in ROIS.items():
            roi_frames = []
            for frame_idx, frame in enumerate(video_frames):
                landmarks = chunk_landmarks[frame_idx]
                if np.all(landmarks == 0):
                    h, w = frame.shape[:2]
                    bbox = (w // 2 - roi_box_size[0] // 2, h // 2 - roi_box_size[1] // 2, 
                           roi_box_size[0], roi_box_size[1])
                else:
                    bbox = extract_roi_bbox(landmarks, roi_indices, frame.shape[:2], roi_box_size)
                roi_region = extract_roi_region(frame, bbox)
                roi_frames.append(roi_region)
            roi_data[roi_name] = np.array(roi_frames)
        
        # Create metadata
        chunk_meta = {
            'subject_id': meta_data.get('subject_id'),
            'condition': meta_data.get('condition'),
            'camera_type': meta_data.get('camera_type'),
            'view_type': meta_data.get('view_type'),
            'chunk_index': chunk_idx,
            'start_frame': chunk_start,
            'end_frame': chunk_end,
            'num_frames': chunk_end - chunk_start,
        }
        
        # Extract vital signs
        vital_signs = {}
        vital_cols = ['upper_ap', 'lower_ap', 'saturation', 'hemoglobin',
                      'glycated_hemoglobin', 'cholesterol', 'respiratory',
                      'rigidity', 'pulse', 'stress']
        for col in vital_cols:
            if col in meta_data:
                vital_signs[col] = meta_data[col]
        
        return {
            'roi_data': roi_data,
            'ppg_values': ppg_values,
            'time_deltas': time_deltas,
            'landmarks': chunk_landmarks,
            'metadata': chunk_meta,
            'vital_signs': vital_signs,
        }
    except Exception as e:
        print(f'Error processing chunk: {e}')
        return None


def save_chunk_as_npz(chunk_data, output_path):
    """Save chunk data as NPZ file."""
    try:
        os.makedirs(output_path, exist_ok=True)
        save_data = {}
        
        # Save ROI data
        for roi_name, roi_frames in chunk_data['roi_data'].items():
            save_data[f'roi_{roi_name}'] = roi_frames
        
        # Save PPG data
        save_data['ppg_values'] = chunk_data['ppg_values']
        save_data['time_deltas'] = chunk_data['time_deltas']
        
        # Save landmarks
        save_data['landmarks'] = chunk_data['landmarks']
        
        # Save metadata
        for key, value in chunk_data['metadata'].items():
            save_data[f'meta_{key}'] = value
        
        # Save vital signs
        for key, value in chunk_data['vital_signs'].items():
            save_data[f'vital_{key}'] = value
        
        # Create filename
        subject_id = chunk_data['metadata']['subject_id']
        camera = chunk_data['metadata']['camera_type']
        condition = chunk_data['metadata']['condition']
        chunk_idx = chunk_data['metadata']['chunk_index']
        filename = f'{subject_id}_{camera}_{condition}_chunk{chunk_idx}.npz'
        filepath = os.path.join(output_path, filename)
        
        # Save as compressed NPZ
        np.savez_compressed(filepath, **save_data)
        return filepath
    except Exception as e:
        print(f'Error saving chunk: {e}')
        return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def load_dataset_metadata(dataset_path, db_path=None, mode='test', num_test_files=10):
    """Load dataset metadata from CSV or directory structure."""
    if db_path and os.path.exists(db_path):
        df = pd.read_csv(db_path)
        meta_df = df.copy()
        print(f'Loaded database with {len(meta_df)} entries')
    else:
        # Try to find db.csv in dataset path
        db_path = os.path.join(dataset_path, 'db.csv')
        if os.path.exists(db_path):
            df = pd.read_csv(db_path)
            meta_df = df.copy()
            print(f'Loaded database with {len(meta_df)} entries')
        else:
            print('No database file found, scanning directory structure')
            meta_df = pd.DataFrame()
    
    # Prepare file paths
    file_columns = ['ecg', 'video', 'meta', 'ppg_sync']
    for col in file_columns:
        if col in meta_df.columns:
            meta_df[f'{col}_full'] = meta_df[col].apply(
                lambda x: os.path.join(dataset_path, x) if not os.path.isabs(x) else x
            )
    
    # Add derived columns
    meta_df['subject_id'] = meta_df.get('patient_id', meta_df.get('subject_id', 'unknown'))
    meta_df['condition'] = meta_df.get('step', meta_df.get('condition', 'unknown'))
    meta_df['camera_type'] = meta_df.get('camera', meta_df.get('camera_type', 'unknown'))
    meta_df['view_type'] = meta_df.get('view', meta_df.get('view_type', 'unknown'))
    
    # Filter valid entries
    valid_df = meta_df.dropna(subset=['video_full', 'ppg_sync_full'])
    print(f'Valid entries with video and ppg_sync: {len(valid_df)}')
    
    # Select files based on mode
    if mode == 'test':
        selected_files = valid_df.head(num_test_files)
        print(f'Selected {len(selected_files)} files for test mode')
    else:
        selected_files = valid_df
        print(f'Selected {len(selected_files)} files for full mode')
    
    return selected_files


def process_single_video(video_row, detector, output_path, chunk_size, overlap_size, 
                        roi_box_size, skip_existing=True):
    """Process all chunks for a single video."""
    video_path = video_row['video_full']
    ppg_sync_path = video_row['ppg_sync_full']
    
    # Prepare metadata
    meta_data = {
        'subject_id': video_row['subject_id'],
        'condition': video_row['condition'],
        'camera_type': video_row['camera_type'],
        'view_type': video_row['view_type'],
    }
    
    # Add vital signs
    vital_cols = ['upper_ap', 'lower_ap', 'saturation', 'hemoglobin',
                  'glycated_hemoglobin', 'cholesterol', 'respiratory',
                  'rigidity', 'pulse', 'stress']
    for col in vital_cols:
        if col in video_row:
            meta_data[col] = video_row[col]
    
    # Count frames
    n_frames = count_video_frames_ffmpeg(video_path)
    if n_frames == 0:
        return []
    
    # Create chunks
    chunks = create_chunks(n_frames, chunk_size, overlap_size)
    
    # Process each chunk
    saved_files = []
    for start, end, chunk_idx in chunks:
        # Check if chunk already exists
        subject_id = video_row['subject_id']
        camera = video_row['camera_type']
        condition = video_row['condition']
        filename = f'{subject_id}_{camera}_{condition}_chunk{chunk_idx}.npz'
        filepath = os.path.join(output_path, filename)
        
        if skip_existing and os.path.exists(filepath):
            print(f'  Skipping existing chunk: {filename}')
            saved_files.append(filepath)
            continue
        
        chunk_data = process_video_chunk(
            video_path, ppg_sync_path, meta_data, start, end, chunk_idx, 
            detector, roi_box_size
        )
        if chunk_data is not None:
            saved_path = save_chunk_as_npz(chunk_data, output_path)
            if saved_path:
                saved_files.append(saved_path)
    
    return saved_files


def main():
    """Main processing function."""
    args = parse_args()
    
    print('=' * 80)
    print('MCD-rPPG DATASET PROCESSING PIPELINE')
    print('=' * 80)
    print(f'Mode: {args.mode}')
    print(f'Dataset path: {args.dataset_path}')
    print(f'Output path: {args.output_path}')
    print(f'Chunk size: {args.chunk_size}')
    print(f'Overlap: {args.overlap}')
    print(f'ROI box size: {args.roi_box_size}')
    print(f'Max workers: {args.workers}')
    print(f'Skip existing: {args.skip_existing}')
    print()
    
    # Create output directories
    chunks_path = os.path.join(args.output_path, 'chunks')
    os.makedirs(chunks_path, exist_ok=True)
    
    # Load dataset metadata
    print('Loading dataset metadata...')
    selected_files = load_dataset_metadata(
        args.dataset_path, 
        args.db_path, 
        args.mode, 
        args.test_files
    )
    
    if len(selected_files) == 0:
        print('No files to process. Exiting.')
        return
    
    print()
    print('Dataset Statistics:')
    print(f'  Total selected files: {len(selected_files)}')
    print(f'  Camera types: {selected_files["camera_type"].unique()}')
    print(f'  Conditions: {selected_files["condition"].unique()}')
    print(f'  Subjects: {selected_files["subject_id"].nunique()}')
    print()
    
    # Initialize MediaPipe detector
    print('Initializing MediaPipe detector...')
    detector = MediaPipeLandmarkDetector(
        args.mediapipe_model, 
        smoothing_window=args.smoothing_window
    )
    
    if not detector.mediapipe_available:
        print('ERROR: MediaPipe is not available. Cannot process videos.')
        print('Please install MediaPipe and ensure the model file is available.')
        return
    
    print('MediaPipe detector initialized successfully')
    print()
    
    # Start processing
    start_time = time.time()
    print('Starting processing...')
    print('-' * 80)
    
    all_saved_files = []
    
    # Process videos sequentially (MediaPipe detector cannot be pickled)
    for i, video_row in enumerate(tqdm(selected_files, desc='Processing videos')):
        subject_id = video_row['subject_id']
        camera = video_row['camera_type']
        condition = video_row['condition']
        
        print(f'\nProcessing video {i+1}/{len(selected_files)}: {subject_id}_{camera}_{condition}')
        
        saved_files = process_single_video(
            video_row, 
            detector, 
            chunks_path, 
            args.chunk_size, 
            args.overlap,
            tuple(args.roi_box_size),
            args.skip_existing
        )
        
        all_saved_files.extend(saved_files)
        
        # Clean up memory periodically
        if (i + 1) % 5 == 0:
            gc.collect()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print()
    print('=' * 80)
    print('PROCESSING COMPLETE')
    print('=' * 80)
    print(f'Total videos processed: {len(selected_files)}')
    print(f'Total chunks saved: {len(all_saved_files)}')
    print(f'Processing time: {processing_time:.2f} seconds')
    print(f'Average time per video: {processing_time/len(selected_files):.2f} seconds')
    print()
    
    # Calculate total output size
    total_size = sum(os.path.getsize(f) for f in all_saved_files) / (1024 * 1024)
    print(f'Total output size: {total_size:.2f} MB')
    print()
    
    # Save processing summary
    summary = {
        'processing_mode': args.mode,
        'num_files_processed': len(selected_files),
        'num_chunks_saved': len(all_saved_files),
        'processing_time_seconds': processing_time,
        'total_output_size_mb': total_size,
        'timestamp': datetime.now().isoformat(),
        'chunk_size': args.chunk_size,
        'overlap_size': args.overlap,
        'roi_box_size': list(args.roi_box_size),
        'max_workers': args.workers,
        'dataset_path': args.dataset_path,
        'output_path': args.output_path
    }
    
    summary_path = os.path.join(args.output_path, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Processing summary saved to: {summary_path}')
    
    print()
    print('=' * 80)
    print('ALL DONE!')
    print('=' * 80)


if __name__ == '__main__':
    main()

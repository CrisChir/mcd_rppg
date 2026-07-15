# MCD-rPPG Preprocessing Pipeline

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green?logo=opencv)](https://opencv.org/)
[![Face Alignment](https://img.shields.io/badge/FaceAlignment-1.4.1-orange)](https://github.com/1adrianb/face-alignment)

This document provides comprehensive documentation for preprocessing the MCD-rPPG dataset, including setup instructions, preprocessing steps, scripts, troubleshooting, and performance optimization.

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Prerequisites](#-prerequisites)
3. [Dataset Download](#-dataset-download)
4. [Preprocessing Steps](#-preprocessing-steps)
5. [Preprocessing Scripts](#-preprocessing-scripts)
6. [Metadata and Synchronization](#-metadata-and-synchronization)
7. [Troubleshooting](#-troubleshooting)
8. [Performance Tips](#-performance-tips)
9. [Full Pipeline Example](#-full-pipeline-example)
10. [Preprocessed Dataset Output](#-preprocessed-dataset-output)

## 🎯 Overview

The preprocessing pipeline transforms raw MCD-rPPG dataset into a format suitable for training deep learning models. The pipeline includes:

- **Face Detection:** Detect and track faces across video frames
- **Face Alignment:** Extract facial landmarks for ROI selection
- **Chunking:** Split long videos into manageable segments
- **Filtering:** Remove low-quality frames and videos
- **Feature Extraction:** Extract facial regions and normalize data
- **Synchronization:** Align video frames with PPG/ECG signals

## 📦 Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | 4 cores | 16+ cores | Multi-core for parallel processing |
| **RAM** | 16 GB | 64+ GB | More RAM = larger batch sizes |
| **GPU** | None | NVIDIA GPU | Optional for face detection acceleration |
| **Storage** | 200 GB | 500+ GB SSD | Fast storage for video processing |
| **Disk Space** | 200 GB | 500+ GB | For raw + processed data |

### Software Dependencies

**Core Dependencies (in `requirements.txt`):**
```bash
face_alignment==1.4.1
matplotlib==3.10.7
numba==0.62.1
numpy==2.3.5
pandas==2.3.3
scikit_learn==1.7.2
scipy==1.16.3
timm==1.0.22
torch==2.6.0
torchaudio==2.6.0
tqdm==4.67.1
```

**Additional Preprocessing Dependencies:**
```bash
pip install opencv-python-headless datasets huggingface_hub scikit-image
```

### Storage Requirements

| Data Type | Size (per subject) | Total Size (600 subjects) |
|-----------|-------------------|---------------------------|
| Raw Videos | ~230 MB | ~135 GB |
| Processed Faces | ~50 MB | ~30 GB |
| Landmarks | ~5 MB | ~3 GB |
| PPG Signals | ~1 MB | ~600 MB |
| ECG Signals | ~1 MB | ~600 MB |
| **Total** | **~287 MB** | **~170 GB** |

**Note:** Processed data can be deleted after training to save space.

## 📥 Dataset Download

Three methods are available for downloading the MCD-rPPG dataset:

### Method 1: Hugging Face Hub (Recommended)

```python
from datasets import load_dataset
from huggingface_hub import login

# Login to Hugging Face (if private access required)
login(token="your_hf_token_here")

# Load the dataset
dataset = load_dataset("Bgeorge/mcd_rppg", split="train")

# Download to local directory
dataset.save_to_disk("mcd_rppg_raw")
```

**Pros:**
- Automatic caching
- Version control
- Partial downloads supported
- Built-in progress tracking

**Cons:**
- Requires internet connection
- May be slower for large datasets

### Method 2: Git LFS

```bash
# Install Git LFS
git lfs install

# Clone the dataset repository
git clone https://huggingface.co/datasets/Bgeorge/mcd_rppg
cd mcd_rppg

# Pull all files
git lfs pull
```

**Pros:**
- Version controlled
- Can resume interrupted downloads
- Good for incremental updates

**Cons:**
- Requires Git LFS setup
- Large repository size
- Slower for initial clone

### Method 3: Manual Download

1. Visit: https://huggingface.co/datasets/Bgeorge/mcd_rppg
2. Click "Download dataset" button
3. Extract the downloaded archive:
   ```bash
   # For .tar.gz files
   tar -xzvf mcd_rppg.tar.gz
   
   # For .zip files
   unzip mcd_rppg.zip
   ```
4. Verify file integrity:
   ```bash
   # Check file counts
   find mcd_rppg -name "*.avi" | wc -l  # Should be 3600
   find mcd_rppg -name "*.npy" | wc -l  # Should be 7200+ (PPG + ECG)
   ```

**Pros:**
- No dependencies required
- Can use download managers
- Good for offline environments

**Cons:**
- Manual process
- No version control
- Need to verify integrity

## 🔧 Preprocessing Steps

### Step 1: Face Detection

**Purpose:** Detect and track faces in video frames

**Algorithm:** Face Alignment library (68 landmarks)

**Process:**
1. Load video frame by frame
2. Detect face using face_alignment
3. Extract 68 facial landmarks
4. Handle face detection failures (skip or interpolate)

**Output:**
- `landmarks.npy`: Array of shape (T, 68, 2) containing (x, y) coordinates
- `detection_errors.csv`: Log of frames where face detection failed

### Step 2: Face Cropping and Alignment

**Purpose:** Extract and normalize facial regions

**Process:**
1. Find bounding box from landmarks
2. Crop video to face region
3. Align face using similarity transform
4. Normalize face size (optional)

**Output:**
- `faces.npy`: Array of shape (T, H, W, 3) containing cropped faces
- `bbox.npy`: Bounding box coordinates for each frame

### Step 3: Chunking

**Purpose:** Split long videos into training samples

**Parameters:**
- `window_size`: Number of frames per sample (default: 256)
- `stride`: Step size between consecutive samples (default: 64)
- `min_length`: Minimum chunk length to keep (default: 128)

**Process:**
1. Slide window across video with specified stride
2. Extract video chunks and corresponding PPG segments
3. Filter out chunks that are too short

**Output:**
- Multiple chunk files: `chunk_{video_id}_{start_frame}.npy`
- Corresponding PPG chunks: `ppg_chunk_{video_id}_{start_frame}.npy`

### Step 4: Quality Filtering

**Purpose:** Remove low-quality data

**Filters Applied:**

1. **Face Detection Confidence**
   - Remove frames with confidence < 0.5
   - Interpolate missing landmarks

2. **Motion Blur Detection**
   - Calculate frame-to-frame differences
   - Remove frames with excessive motion blur

3. **Illumination Quality**
   - Check for uniform illumination
   - Remove overly dark or bright frames

4. **Face Size**
   - Minimum face size: 64×64 pixels
   - Maximum face size: 512×512 pixels

5. **PPG Signal Quality**
   - Check for signal saturation
   - Remove segments with flat signals
   - Filter based on signal-to-noise ratio

**Output:**
- `filtered_videos/`: Clean video chunks
- `filtered_ppg/`: Corresponding clean PPG segments
- `quality_scores.csv`: Quality metrics for each chunk

### Step 5: Feature Extraction

**Purpose:** Extract features for model input

**Features Extracted:**

1. **Spatial Features**
   - Face ROI (Region of Interest)
   - Forehead region
   - Cheek regions (left and right)
   - Nose region
   - Chin region

2. **Temporal Features**
   - Frame differences
   - Optical flow
   - Motion compensation

3. **Normalization**
   - Per-frame normalization (0-1 range)
   - Per-video normalization (z-score)
   - Global normalization (dataset statistics)

**Output:**
- `features/`: Extracted features in various formats
- `normalization_params.json`: Normalization parameters

### Step 6: Synchronization Alignment

**Purpose:** Ensure precise alignment between video and physiological signals

**Process:**
1. Use metadata timestamps for initial alignment
2. Apply cross-correlation for fine alignment
3. Compensate for any drift during recording
4. Validate alignment quality

**Metadata Used:**
- `video_ppg_offset`: Time offset between video and PPG
- `ppg_ecg_offset`: Time offset between PPG and ECG
- `frame_rate`: Video frame rate
- `ppg_rate`: PPG sampling rate

**Output:**
- `aligned_data/`: Synchronized video-PPG pairs
- `alignment_quality.csv`: Alignment metrics

## 📜 Preprocessing Scripts

The repository includes three preprocessing scripts that can be run in parallel:

### Script 1: `dataset_preprocessing_1.py`

Processes videos from the first camera view (frontal webcam):

```bash
# Run on first third of videos
python dataset_preprocessing_1.py \
    --input_path /path/to/videos \
    --output_path /path/to/output \
    --camera_id 1 \
    --start_idx 0 \
    --end_idx 1200 \
    --num_workers 8
```

### Script 2: `dataset_preprocessing_2.py`

Processes videos from the second camera view (FullHD camcorder):

```bash
# Run on second third of videos
python dataset_preprocessing_2.py \
    --input_path /path/to/videos \
    --output_path /path/to/output \
    --camera_id 2 \
    --start_idx 1200 \
    --end_idx 2400 \
    --num_workers 8
```

### Script 3: `dataset_preprocessing_3.py`

Processes videos from the third camera view (mobile phone):

```bash
# Run on last third of videos
python dataset_preprocessing_3.py \
    --input_path /path/to/videos \
    --output_path /path/to/output \
    --camera_id 3 \
    --start_idx 2400 \
    --end_idx 3600 \
    --num_workers 8
```

### Common Script Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input_path` | str | Required | Path to raw video files |
| `--output_path` | str | Required | Path to save processed data |
| `--camera_id` | int | Required | Camera identifier (1, 2, or 3) |
| `--start_idx` | int | 0 | Starting video index |
| `--end_idx` | int | 3600 | Ending video index |
| `--num_workers` | int | 4 | Number of parallel workers |
| `--window_size` | int | 256 | Frames per training sample |
| `--stride` | int | 64 | Step between samples |
| `--min_face_size` | int | 64 | Minimum face size in pixels |
| `--max_face_size` | int | 512 | Maximum face size in pixels |
| `--skip_existing` | bool | True | Skip already processed files |
| `--verbose` | bool | False | Enable verbose logging |

### Running All Scripts in Parallel

```bash
# Method 1: Using GNU parallel
parallel -j 3 'python dataset_preprocessing_{}.py --input_path /data/videos --output_path /data/processed --num_workers 4' ::: 1 2 3

# Method 2: Using separate terminals
# Terminal 1
python dataset_preprocessing_1.py --input_path /data/videos --output_path /data/processed --num_workers 4

# Terminal 2
python dataset_preprocessing_2.py --input_path /data/videos --output_path /data/processed --num_workers 4

# Terminal 3
python dataset_preprocessing_3.py --input_path /data/videos --output_path /data/processed --num_workers 4

# Method 3: Using nohup for background execution
nohup python dataset_preprocessing_1.py --input_path /data/videos --output_path /data/processed --num_workers 4 > preprocess_1.log 2>&1 &
nohup python dataset_preprocessing_2.py --input_path /data/videos --output_path /data/processed --num_workers 4 > preprocess_2.log 2>&1 &
nohup python dataset_preprocessing_3.py --input_path /data/videos --output_path /data/processed --num_workers 4 > preprocess_3.log 2>&1 &
```

## 🔗 Metadata and Synchronization

### PPG and POG Sync Metadata

The dataset includes **PPG (Photoplethysmogram)** and **POG (Pulse Oxymeter Ground truth)** signals with synchronization metadata:

#### PPG Metadata Fields

| Field | Type | Description | Usage in Preprocessing |
|-------|------|-------------|------------------------|
| `ppg_file` | str | Path to PPG signal file | Load signal data |
| `ppg_sampling_rate` | int | Sampling rate (100 Hz) | Resample if needed |
| `ppg_start_time` | float | Start timestamp | Align with video |
| `ppg_end_time` | float | End timestamp | Validate duration |
| `ppg_quality` | float | Signal quality score (0-1) | Filter low-quality signals |

#### POG Sync Metadata

| Field | Type | Description | Usage in Preprocessing |
|-------|------|-------------|------------------------|
| `pog_sync` | bool | Whether POG is synchronized | Use for validation |
| `pog_file` | str | Path to POG signal file | Load reference signal |
| `pog_sampling_rate` | int | POG sampling rate (100 Hz) | Match with PPG |
| `ppg_pog_offset` | float | Time offset between PPG and POG | Fine alignment |
| `pog_quality` | float | POG signal quality (0-1) | Filter threshold |

#### Synchronization Process

```python
import numpy as np
from scipy.signal import correlate

def align_signals(video_frames, ppg_signal, metadata):
    """
    Align video frames with PPG signal using metadata and cross-correlation.
    
    Args:
        video_frames: Array of shape (T_video, H, W, 3)
        ppg_signal: Array of shape (T_ppg,)
        metadata: Dictionary containing sync metadata
    
    Returns:
        aligned_video: Aligned video frames
        aligned_ppg: Aligned PPG signal
        offset: Applied time offset in seconds
    """
    # Initial alignment from metadata
    video_rate = metadata['frame_rate']
    ppg_rate = metadata['ppg_rate']
    initial_offset = metadata['video_ppg_offset']
    
    # Convert offset to frame indices
    offset_frames = int(initial_offset * video_rate)
    offset_ppg_samples = int(initial_offset * ppg_rate)
    
    # Apply initial alignment
    if offset_frames > 0:
        video_frames = video_frames[offset_frames:]
        ppg_signal = ppg_signal[:-offset_ppg_samples]
    else:
        video_frames = video_frames[-offset_frames:]
        ppg_signal = ppg_signal[-offset_ppg_samples:]
    
    # Fine alignment using cross-correlation
    # (Implementation depends on your specific alignment strategy)
    
    return video_frames, ppg_signal, initial_offset
```

### Vitals Metadata Usage

Vitals metadata is used for:

1. **Validation:** Compare model predictions with ground truth vitals
2. **Normalization:** Scale features based on subject-specific vitals
3. **Filtering:** Remove samples with abnormal vitals
4. **Stratification:** Ensure balanced representation across vitals ranges

#### Example: Using Vitals for Filtering

```python
import pandas as pd

def filter_by_vitals(metadata_df, video_ids):
    """
    Filter videos based on vitals metadata.
    
    Args:
        metadata_df: DataFrame with vitals metadata
        video_ids: List of video identifiers
    
    Returns:
        valid_video_ids: List of video IDs that pass filters
    """
    valid_ids = []
    
    for video_id in video_ids:
        subject_id = extract_subject_id(video_id)
        condition = extract_condition(video_id)
        
        # Get subject vitals
        subject_vitals = metadata_df[metadata_df['subject_id'] == subject_id].iloc[0]
        
        # Apply filters
        if (subject_vitals['heart_rate_bpm'] < 40 or 
            subject_vitals['heart_rate_bpm'] > 200):
            continue  # Abnormal heart rate
        
        if subject_vitals['spo2_percent'] < 90:
            continue  # Low oxygen saturation
        
        if (subject_vitals['blood_pressure_sys'] < 90 or 
            subject_vitals['blood_pressure_sys'] > 180):
            continue  # Abnormal blood pressure
        
        valid_ids.append(video_id)
    
    return valid_ids
```

## ⚠️ Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

**Symptoms:**
- `MemoryError: Unable to allocate array`
- `Killed` (process terminated by system)
- Slow performance with high memory usage

**Solutions:**

```bash
# Reduce batch size
python dataset_preprocessing_1.py --num_workers 2 --batch_size 16

# Process fewer videos at once
python dataset_preprocessing_1.py --start_idx 0 --end_idx 100

# Use smaller window size
python dataset_preprocessing_1.py --window_size 128 --stride 32

# Enable memory-efficient mode
python dataset_preprocessing_1.py --memory_efficient True
```

**Advanced Solutions:**
- Use **memory-mapped arrays** (numpy.memmap)
- Process videos **one at a time** instead of batching
- Use **generators** instead of loading all data into memory
- **Upgrade RAM** or use a machine with more memory

#### 2. Slow Face Detection

**Symptoms:**
- Face detection takes > 1 second per frame
- Overall preprocessing is very slow
- CPU/GPU utilization is low

**Solutions:**

```bash
# Use GPU for face detection
python dataset_preprocessing_1.py --device cuda:0

# Reduce face detector accuracy for speed
python dataset_preprocessing_1.py --face_detector fast

# Use smaller input resolution
python dataset_preprocessing_1.py --input_resolution 640x480
```

**Advanced Solutions:**
- Use **MediaPipe** instead of face_alignment (faster but less accurate)
- **Batch face detection** across multiple frames
- Use **ONNX runtime** for accelerated inference
- **Pre-compute landmarks** and save to disk

#### 3. Face Detection Failures

**Symptoms:**
- Many frames with no face detected
- `AssertionError: No face detected`
- Low face detection confidence scores

**Solutions:**

```bash
# Lower detection confidence threshold
python dataset_preprocessing_1.py --min_confidence 0.3

# Enable face tracking to maintain detection
python dataset_preprocessing_1.py --enable_tracking True

# Use previous frame landmarks for interpolation
python dataset_preprocessing_1.py --interpolate_missing True
```

**Advanced Solutions:**
- **Manual inspection** of problematic videos
- **Skip videos** with persistent detection failures
- Use **ensemble of detectors** for robustness
- **Upscale low-resolution videos** before detection

#### 4. Synchronization Issues

**Symptoms:**
- PPG and video signals don't align
- Heart rate predictions don't match ground truth
- Large offsets between modalities

**Solutions:**

```python
# Verify and recalculate offsets
from rppglib.sync_utils import recalculate_offsets

offsets = recalculate_offsets(video_files, ppg_files, metadata)

# Use cross-correlation for fine alignment
from rppglib.sync_utils import align_with_correlation

aligned_video, aligned_ppg = align_with_correlation(video, ppg, max_offset=1.0)
```

**Advanced Solutions:**
- **Visual inspection** of alignment using plots
- **Manual adjustment** of offsets in metadata
- Use **ECG as reference** for more accurate alignment
- **Interpolate signals** to match sampling rates

#### 5. File Not Found Errors

**Symptoms:**
- `FileNotFoundError: [Errno 2] No such file or directory`
- Missing video or PPG files
- Inconsistent file naming

**Solutions:**

```bash
# Verify dataset integrity
python -c "
from datasets import load_dataset
ds = load_dataset('Bgeorge/mcd_rppg', split='train')
print(f'Total samples: {len(ds)}')
print(f'First sample keys: {ds[0].keys()}')
"

# Check file paths
ls -la /path/to/videos | head -20
ls -la /path/to/ppg | head -20

# Use absolute paths
python dataset_preprocessing_1.py --input_path /absolute/path/to/videos
```

**Advanced Solutions:**
- **Symlink missing files** to expected locations
- **Re-download** corrupted files
- **Verify checksums** of downloaded files
- **Update paths** in configuration files

#### 6. Dependency Conflicts

**Symptoms:**
- `ImportError: cannot import name X`
- `ModuleNotFoundError: No module named X`
- Version conflicts between packages

**Solutions:**

```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate

# Install exact versions
pip install face_alignment==1.4.1 opencv-python-headless==4.9.0.80

# Check installed versions
pip list | grep -E "(face|opencv|numpy|torch)"
```

**Advanced Solutions:**
- Use **requirements.txt** for consistent environment
- Use **conda** for better dependency management
- **Downgrade conflicting packages**
- **Check CUDA compatibility** for GPU packages

### Error Logging and Debugging

All preprocessing scripts log errors to `errors.csv`:

```csv
video_file,error_type,error_msg,timestamp
/path/to/video_001.avi,AssertionError,No face detected,2025-01-15 10:30:45
/path/to/video_002.avi,ValueError,Invalid frame shape,2025-01-15 10:31:22
```

**Debug Mode:**
```bash
python dataset_preprocessing_1.py --debug True --verbose True
```

This will:
- Save intermediate results
- Generate visualization of problematic frames
- Print detailed timing information
- Create stack traces for errors

## ⚡ Performance Tips

### Faster Preprocessing

#### 1. Parallel Processing

```bash
# Use all available CPU cores
python dataset_preprocessing_1.py --num_workers -1

# Distribute across multiple machines
# Use a job scheduler (SLURM, PBS, etc.)
```

#### 2. GPU Acceleration

```bash
# Use CUDA for face detection
python dataset_preprocessing_1.py --device cuda:0

# Use TensorRT for optimized inference
python dataset_preprocessing_1.py --engine tensorrt
```

#### 3. Memory Optimization

```bash
# Use memory-mapped arrays
python dataset_preprocessing_1.py --use_memmap True

# Process in smaller batches
python dataset_preprocessing_1.py --batch_size 8 --num_workers 2
```

#### 4. Caching

```bash
# Cache face detection results
python dataset_preprocessing_1.py --cache_dir /tmp/face_cache --use_cache True

# Cache intermediate results
python dataset_preprocessing_1.py --intermediate_cache True
```

### Reducing Storage Requirements

#### 1. Compression

```bash
# Compress processed data
python -c "
import numpy as np
import zarr

# Save as compressed numpy
np.savez_compressed('faces.npz', faces=faces, landmarks=landmarks)

# Use Zarr for chunked storage
zarr.save('faces.zarr', faces, compressor=zarr.Blosc(cname='zstd', clevel=5))
"
```

#### 2. Selective Processing

```bash
# Process only specific camera views
python dataset_preprocessing_1.py --camera_id 1 --only_camera True

# Process only high-quality videos
python dataset_preprocessing_1.py --min_quality 0.8

# Skip already processed files
python dataset_preprocessing_1.py --skip_existing True
```

#### 3. Cleanup

```bash
# Remove intermediate files after processing
python dataset_preprocessing_1.py --cleanup True

# Manual cleanup script
find /path/to/output -name "*.npy" -size +100M -delete
find /path/to/output -name "temp_*" -delete
```

### Optimizing Training

#### 1. Data Loading Optimization

```python
# Use PyTorch DataLoader with multiple workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=8, prefetch_factor=2)

# Use pinned memory for GPU training
dataloader = DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True)
```

#### 2. Mixed Precision Training

```python
# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 3. Gradient Accumulation

```python
# Accumulate gradients for larger effective batch size
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = model(batch['video'])
    loss = criterion(outputs, batch['ppg'])
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🚀 Full Pipeline Example

### End-to-End Python Code

```python
"""
Complete preprocessing pipeline example.
This script demonstrates the full preprocessing workflow from raw data to training-ready samples.
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import cv2
import face_alignment
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

# Configuration
CONFIG = {
    'dataset_path': 'mcd_rppg_raw',
    'output_path': 'mcd_rppg_processed',
    'window_size': 256,
    'stride': 64,
    'min_face_size': 64,
    'max_face_size': 512,
    'ppg_low_freq': 0.75,
    'ppg_high_freq': 4.0,
    'frame_rate': 30,
    'ppg_rate': 100,
    'num_workers': 4,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1
}

# Ensure output directory exists
os.makedirs(CONFIG['output_path'], exist_ok=True)

# ============================================================================
# Step 1: Load Dataset
# ============================================================================
print("Loading dataset...")
dataset = load_dataset("Bgeorge/mcd_rppg", split="train")
print(f"Loaded {len(dataset)} samples")

# ============================================================================
# Step 2: Initialize Face Detector
# ============================================================================
print("Initializing face detector...")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda:0')
print("Face detector initialized")

# ============================================================================
# Step 3: Preprocessing Functions
# ============================================================================

def detect_face(frame):
    """Detect face and extract landmarks."""
    preds = fa.get_landmarks(frame)
    if preds is None:
        return None
    return preds[0].astype('int16')

def crop_face(frame, landmarks):
    """Crop face region from frame."""
    x_min, x_max = landmarks[:, 0].min(), landmarks[:, 0].max()
    y_min, y_max = landmarks[:, 1].min(), landmarks[:, 1].max()
    
    # Add padding
    pad = 20
    x_min = max(0, x_min - pad)
    x_max = min(frame.shape[1], x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(frame.shape[0], y_max + pad)
    
    return frame[y_min:y_max, x_min:x_max]

def bandpass_filter(signal, low_freq, high_freq, fs, order=4):
    """Apply bandpass filter to signal."""
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

def preprocess_ppg(ppg_signal):
    """Preprocess PPG signal."""
    # Bandpass filter
    ppg_filtered = bandpass_filter(
        ppg_signal, 
        CONFIG['ppg_low_freq'], 
        CONFIG['ppg_high_freq'], 
        CONFIG['ppg_rate']
    )
    
    # Normalize
    ppg_normalized = (ppg_filtered - ppg_filtered.mean()) / ppg_filtered.std()
    
    return ppg_normalized

def extract_chunks(video, ppg, window_size, stride):
    """Extract chunks from video and PPG."""
    chunks = []
    ppg_chunks = []
    
    num_frames = video.shape[0]
    ppg_length = ppg.shape[0]
    
    # Calculate PPG samples per frame
    ppg_per_frame = CONFIG['ppg_rate'] / CONFIG['frame_rate']
    
    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        
        # Get video chunk
        video_chunk = video[start:end]
        
        # Get corresponding PPG chunk
        ppg_start = int(start * ppg_per_frame)
        ppg_end = int(end * ppg_per_frame)
        ppg_chunk = ppg[ppg_start:ppg_end]
        
        # Ensure PPG chunk has correct length
        if len(ppg_chunk) < window_size:
            continue
        if len(ppg_chunk) > window_size:
            ppg_chunk = ppg_chunk[:window_size]
        
        chunks.append(video_chunk)
        ppg_chunks.append(ppg_chunk)
    
    return np.array(chunks), np.array(ppg_chunks)

# ============================================================================
# Step 4: Process Dataset
# ============================================================================
print("Processing dataset...")

all_chunks = []
all_ppg_chunks = []
all_metadata = []

# Process first N samples for demonstration
N_SAMPLES = 10  # Set to len(dataset) for full processing

for i in tqdm(range(min(N_SAMPLES, len(dataset)))):
    sample = dataset[i]
    
    # Load video and PPG
    video = sample['video']
    ppg = sample['ppg']
    metadata = {
        'subject_id': sample.get('subject_id', f'subject_{i:03d}'),
        'camera_id': sample.get('camera_id', 1),
        'condition': sample.get('condition', 'rest'),
        'video_file': sample.get('video_file', f'video_{i:03d}.avi')
    }
    
    # Process each frame
    processed_frames = []
    landmarks_list = []
    
    for frame_idx, frame in enumerate(video):
        # Convert to numpy if needed
        if hasattr(frame, 'numpy'):
            frame = frame.numpy()
        
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Detect face
        landmarks = detect_face(frame)
        
        if landmarks is None:
            # Skip frame if no face detected
            continue
        
        # Crop face
        face = crop_face(frame, landmarks)
        
        # Resize to consistent size
        face = cv2.resize(face, (128, 128))
        
        processed_frames.append(face)
        landmarks_list.append(landmarks)
    
    if len(processed_frames) < CONFIG['window_size']:
        # Skip video if too short after filtering
        continue
    
    # Convert to numpy arrays
    processed_video = np.array(processed_frames)
    landmarks_array = np.array(landmarks_list)
    
    # Preprocess PPG
    ppg_processed = preprocess_ppg(ppg)
    
    # Extract chunks
    video_chunks, ppg_chunks = extract_chunks(
        processed_video, 
        ppg_processed,
        CONFIG['window_size'],
        CONFIG['stride']
    )
    
    # Add to collections
    all_chunks.extend(video_chunks)
    all_ppg_chunks.extend(ppg_chunks)
    
    # Add metadata for each chunk
    for j in range(len(video_chunks)):
        chunk_metadata = metadata.copy()
        chunk_metadata['chunk_idx'] = j
        chunk_metadata['start_frame'] = j * CONFIG['stride']
        all_metadata.append(chunk_metadata)

print(f"Processed {len(all_chunks)} chunks from {N_SAMPLES} samples")

# ============================================================================
# Step 5: Save Processed Data
# ============================================================================
print("Saving processed data...")

# Convert to numpy arrays
all_chunks = np.array(all_chunks)
all_ppg_chunks = np.array(all_ppg_chunks)

# Save chunks
np.save(os.path.join(CONFIG['output_path'], 'video_chunks.npy'), all_chunks)
np.save(os.path.join(CONFIG['output_path'], 'ppg_chunks.npy'), all_ppg_chunks)

# Save metadata
metadata_df = pd.DataFrame(all_metadata)
metadata_df.to_csv(os.path.join(CONFIG['output_path'], 'metadata.csv'), index=False)

print(f"Saved {len(all_chunks)} chunks to {CONFIG['output_path']}")

# ============================================================================
# Step 6: Create Train/Val/Test Splits
# ============================================================================
print("Creating train/val/test splits...")

# Split by subject to avoid data leakage
subject_ids = metadata_df['subject_id'].unique()
train_subjects, test_subjects = train_test_split(
    subject_ids, 
    test_size=CONFIG['test_ratio'] + CONFIG['val_ratio'],
    random_state=42
)
val_subjects, test_subjects = train_test_split(
    test_subjects,
    test_size=CONFIG['test_ratio'] / (CONFIG['test_ratio'] + CONFIG['val_ratio']),
    random_state=42
)

# Create split files
splits = {
    'train': train_subjects,
    'val': val_subjects,
    'test': test_subjects
}

for split_name, subject_list in splits.items():
    split_indices = metadata_df[metadata_df['subject_id'].isin(subject_list)].index
    with open(os.path.join(CONFIG['output_path'], f'{split_name}_indices.txt'), 'w') as f:
        for idx in split_indices:
            f.write(f'{idx}\n')

print(f"Created splits: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")

# ============================================================================
# Step 7: Verify Output
# ============================================================================
print("Verifying output...")

# Check saved files
video_chunks = np.load(os.path.join(CONFIG['output_path'], 'video_chunks.npy'))
ppg_chunks = np.load(os.path.join(CONFIG['output_path'], 'ppg_chunks.npy'))
metadata_df = pd.read_csv(os.path.join(CONFIG['output_path'], 'metadata.csv'))

print(f"Video chunks shape: {video_chunks.shape}")
print(f"PPG chunks shape: {ppg_chunks.shape}")
print(f"Metadata shape: {metadata_df.shape}")

# Verify a sample
sample_idx = 0
print(f"\nSample {sample_idx}:")
print(f"  Video chunk shape: {video_chunks[sample_idx].shape}")
print(f"  PPG chunk shape: {ppg_chunks[sample_idx].shape}")
print(f"  Metadata: {metadata_df.iloc[sample_idx].to_dict()}")

print("\n✅ Preprocessing pipeline completed successfully!")
```

### Running the Full Pipeline

```bash
# Save the script above as run_preprocessing.py
python run_preprocessing.py

# Or run with custom configuration
python run_preprocessing.py --output_path /path/to/output --num_samples 100
```

## 📦 Preprocessed Dataset Output

### Summary: Preprocessed Dataset Ready for Training

After running the preprocessing pipeline, you will have the following outputs:

```
mcd_rppg_processed/
├── video_chunks.npy           # Array of shape (N, T, H, W, 3)
├── ppg_chunks.npy             # Array of shape (N, T)
├── metadata.csv               # DataFrame with N rows, metadata for each chunk
├── train_indices.txt          # Indices for training split
├── val_indices.txt            # Indices for validation split
├── test_indices.txt           # Indices for test split
├── preprocessing_log.txt      # Log of preprocessing operations
└── errors.csv                 # List of errors encountered
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Chunks** | ~100,000 - 500,000 (depending on parameters) |
| **Chunk Duration** | 8.5 seconds (256 frames @ 30 FPS) |
| **Video Shape** | (256, 128, 128, 3) |
| **PPG Shape** | (256,) |
| **Total Size** | ~50-100 GB (compressed) |

### Example for Inference Preprocessing

For inference on new videos, use the following preprocessing:

```python
"""
Inference preprocessing example.
This demonstrates how to preprocess a new video for inference using a trained model.
"""

import cv2
import numpy as np
import face_alignment
from scipy.signal import butter, filtfilt

def preprocess_for_inference(video_path, window_size=256):
    """
    Preprocess a video for inference.
    
    Args:
        video_path: Path to video file
        window_size: Number of frames per sample
    
    Returns:
        chunks: List of preprocessed video chunks
    """
    # Initialize face detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda:0')
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    video = np.array(frames)
    
    # Process video
    processed_frames = []
    
    for frame in video:
        # Detect face
        preds = fa.get_landmarks(frame)
        if preds is None:
            continue
        
        landmarks = preds[0].astype('int16')
        
        # Crop face
        x_min, x_max = landmarks[:, 0].min(), landmarks[:, 0].max()
        y_min, y_max = landmarks[:, 1].min(), landmarks[:, 1].max()
        
        pad = 20
        x_min = max(0, x_min - pad)
        x_max = min(frame.shape[1], x_max + pad)
        y_min = max(0, y_min - pad)
        y_max = min(frame.shape[0], y_max + pad)
        
        face = frame[y_min:y_max, x_min:x_max]
        face = cv2.resize(face, (128, 128))
        
        processed_frames.append(face)
    
    # Create chunks
    chunks = []
    for start in range(0, len(processed_frames) - window_size + 1, window_size):
        end = start + window_size
        chunk = processed_frames[start:end]
        
        # Normalize chunk
        chunk = chunk.astype(np.float32) / 255.0
        chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-8)
        
        chunks.append(chunk)
    
    return np.array(chunks)

# Example usage
video_path = "test_video.avi"
chunks = preprocess_for_inference(video_path)

print(f"Processed {len(chunks)} chunks from {video_path}")
print(f"Chunk shape: {chunks[0].shape}")

# Now you can run inference:
# predictions = model.predict(chunks)
```

## 📚 Additional Resources

- **[DATASET.md](../DATASET.md)** - Complete dataset documentation
- **[README.md](../README.md)** - Main repository documentation
- **[rppglib/](../rppglib/)** - Core library with utilities

## 🤝 Support

For issues or questions related to preprocessing:

1. **GitHub Issues:** https://github.com/CrisChir/mcd_rppg/issues
2. **Create a new issue** with the `[preprocessing]` tag
3. **Include:**
   - Error messages
   - Python version
   - Package versions
   - Hardware specifications
   - Steps to reproduce

---

**Last Updated:** September 2025
**Maintainers:** CrisChir, Bgeorge
**License:** MIT

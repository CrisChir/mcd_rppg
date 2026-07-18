# MCD-rPPG Data Processing Pipeline

This directory contains the data processing pipeline for the MCD-rPPG dataset, designed to efficiently preprocess video files for remote photoplethysmography (rPPG) analysis.

## Overview

The processing pipeline provides two main interfaces:

1. **Jupyter Notebook**: `../EDA_Analysis/Data_Processing_Pipeline.ipynb` - Interactive interface for testing and visualization
2. **Python Script**: `data_processing_pipeline.py` - Command-line interface for batch processing

## Features

- **Dual Processing Modes**:
  - Test mode: Process 10 video files for quick validation
  - Full mode: Process the entire dataset

- **Efficient Processing**:
  - Chunk-based processing with configurable overlap
  - Memory-efficient batch processing
  - Temporal smoothing for stable landmark detection

- **Comprehensive ROI Extraction**:
  - 9 facial regions using MediaPipe landmarks
  - 24x24 pixel ROI boxes for each region
  - Support for custom ROI configurations

- **PPG Synchronization**:
  - Automatic alignment of video frames with PPG signals
  - Support for PPG sync files in various formats

- **Parallel Processing**:
  - Configurable number of workers (max 10)
  - Optimized for multi-core systems

## Quick Start

### Using the Jupyter Notebook

1. Open `../EDA_Analysis/Data_Processing_Pipeline.ipynb` in Jupyter
2. Update the configuration parameters:
   - `PROCESSING_MODE`: Set to `'test'` or `'full'`
   - `DATASET_PATH`: Path to your MCD-rPPG dataset
   - `OUTPUT_PATH`: Where to save processed data
   - `MAX_WORKERS`: Number of parallel workers (max 10)
3. Run all cells to start processing

### Using the Command-Line Script

```bash
# Process 10 test files
python data_processing_pipeline.py \
    --mode test \
    --dataset_path /path/to/mcd_rppg \
    --output_path /path/to/output \
    --workers 10

# Process full dataset
python data_processing_pipeline.py \
    --mode full \
    --dataset_path /path/to/mcd_rppg \
    --output_path /path/to/output \
    --workers 10

# Custom configuration
python data_processing_pipeline.py \
    --mode test \
    --dataset_path /path/to/mcd_rppg \
    --output_path /path/to/output \
    --chunk_size 450 \
    --overlap 150 \
    --roi_box_size 24 24 \
    --workers 10 \
    --skip_existing
```

## Configuration Options

### Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `test` | Processing mode: 'test' or 'full' |
| `--dataset_path` | `/home/cristic/data/...` | Path to dataset root |
| `--output_path` | `/home/cristic/preprocessed_data` | Output directory |
| `--db_path` | `None` | Path to database CSV (optional) |
| `--mediapipe_model` | `/home/cristic/face_landmarker.task` | MediaPipe model path |

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chunk_size` | 450 | Frames per chunk |
| `--overlap` | 150 | Overlapping frames between chunks |
| `--roi_box_size` | 24 24 | ROI box dimensions (width, height) |

### Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--smoothing_window` | 5 | Temporal smoothing window for landmarks |
| `--workers` | 10 | Maximum parallel workers |
| `--test_files` | 10 | Number of files in test mode |
| `--skip_existing` | False | Skip already processed files |
| `--verbose` | False | Enable verbose logging |

## ROI Configuration

The pipeline extracts the following facial regions:

- `full_face`: All 468 landmarks
- `forehead`: Forehead region
- `left_eye`: Left eye region
- `right_eye`: Right eye region
- `nose`: Nose region
- `mouth`: Mouth region
- `chin`: Chin region
- `right_cheek_50`: Specific landmark at index 50
- `left_cheek_280`: Specific landmark at index 280
- `chin_199`: Specific landmark at index 199

Each ROI is extracted as a 24x24 pixel box centered on the region's landmark centroid.

## Output Structure

Processed data is saved in the following structure:

```
output_path/
├── chunks/
│   ├── subject1_camera1_condition1_chunk0.npz
│   ├── subject1_camera1_condition1_chunk1.npz
│   └── ...
├── processing_summary.json
└── ...
```

Each NPZ file contains:

- `roi_*`: ROI data arrays for each facial region
- `ppg_values`: PPG signal values
- `time_deltas`: Time deltas between frames
- `landmarks`: Facial landmarks (468 points per frame)
- `meta_*`: Metadata (subject ID, camera type, condition, etc.)
- `vital_*`: Vital signs (if available)

## Requirements

- Python 3.8+
- MediaPipe with face landmarker model
- OpenCV
- NumPy
- Pandas
- scikit-image / imageio
- tqdm
- joblib (for parallel processing)

Install requirements:

```bash
pip install mediapipe opencv-python numpy pandas imageio tqdm joblib
```

## Performance Considerations

### Memory Usage

- Processing in chunks helps manage memory
- Each chunk contains 450 frames with 9 ROIs
- Typical memory usage: ~500-800 MB per video

### Processing Time

- Test mode (10 files): ~10-30 minutes
- Full dataset: Several hours (depending on hardware)
- MediaPipe face detection is the main bottleneck

### Optimization Tips

1. **Hardware**: Use a machine with a good GPU for MediaPipe acceleration
2. **Storage**: Ensure the dataset and model files are on fast SSDs
3. **Memory**: Close other applications to free up RAM
4. **Parallel Processing**: Use the maximum number of workers your CPU can handle (max 10)

## Example Usage

### Test Mode (Quick Validation)

```python
# In Jupyter notebook
PROCESSING_MODE = 'test'
NUM_TEST_FILES = 10
MAX_WORKERS = 10

# Run all cells
```

### Full Dataset Processing

```bash
# Command line
python data_processing_pipeline.py \
    --mode full \
    --dataset_path /home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/ \
    --output_path /home/cristic/preprocessed_data \
    --workers 10
```

## Loading Processed Data

```python
import numpy as np

# Load a single chunk
with np.load('chunks/subject1_camera1_condition1_chunk0.npz') as data:
    # Access ROI data
    roi_forehead = data['roi_forehead']  # Shape: (n_frames, 24, 24, 3)
    roi_left_eye = data['roi_left_eye']   # Shape: (n_frames, 24, 24, 3)
    
    # Access PPG data
    ppg_values = data['ppg_values']      # Shape: (n_frames,)
    time_deltas = data['time_deltas']    # Shape: (n_frames,)
    
    # Access landmarks
    landmarks = data['landmarks']       # Shape: (n_frames, 468, 2)
    
    # Access metadata
    subject_id = data['meta_subject_id']
    camera_type = data['meta_camera_type']
    condition = data['meta_condition']

# Normalize ROI data for model training
roi_normalized = roi_forehead / 255.0
```

## Troubleshooting

### MediaPipe Not Available

Error: `MediaPipe not available`

Solution: Install MediaPipe and download the face landmarker model:

```bash
pip install mediapipe
# Download face_landmarker.task from MediaPipe model hub
```

### No Face Detected

Error: `No face detected in the first frame`

Solution: Check video quality and ensure faces are visible. Adjust MediaPipe parameters if needed.

### Memory Errors

Error: `Out of memory`

Solution: Reduce the number of workers or process fewer files at once.

### File Not Found

Error: `File not found`

Solution: Verify dataset paths in the configuration. Ensure all video and PPG sync files exist.

## References

- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [MCD-rPPG Dataset](https://huggingface.co/datasets/MCD-rPPG)
- [Original Preprocessing Scripts](../preprocessing/)

## License

MIT License - Feel free to use and modify for your research.

## Support

For issues or questions, please refer to the main repository:
- GitHub: https://github.com/CrisChir/mcd_rppg
- Dataset: https://huggingface.co/datasets/MCD-rPPG

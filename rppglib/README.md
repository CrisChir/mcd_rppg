# **rPPG Library (rppglib)**
**Remote Photoplethysmography (rPPG) Toolkit for Heart Rate Estimation from Videos**

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![OpenCV](https://img.shields.io/badge/opencv-4.5%2B-orange)
![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-red)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10%2B-yellow)

---

## **📌 Overview**
`rppglib` is a **Python library** for **remote Photoplethysmography (rPPG)**, enabling **heart rate (HR) and blood volume pulse (BVP/PPG) estimation from facial videos**.
It provides **end-to-end pipelines** for:
- **Video preprocessing** (face detection, ROI extraction, masking).
- **Signal processing** (PPG/ECG filtering, resampling, HR estimation).
- **Deep learning models** (SCNN, PhysFormer, RhythmFormer, etc.).
- **Training/evaluation utilities** (datasets, metrics, cross-validation).

**Key Features**:
✅ **Face detection & landmark extraction** (MediaPipe 468-point model).
✅ **Multi-ROI support** (forehead, cheeks, nose, mouth, etc.).
✅ **Preprocessing for rPPG** (cropping, masking, normalization).
✅ **Deep learning models** for rPPG estimation.
✅ **Cross-dataset evaluation** (MMPD, SCAMPS, UBFC, mcd_rppg).
✅ **Metrics calculation** (PPG MAE, HR MAE).

---

## **📦 Installation**
### **Prerequisites**
- Python ≥ 3.8
- CUDA (for GPU acceleration, optional)
- OpenCV (`opencv-python-headless` recommended)
- MediaPipe (`mediapipe` for face detection)
- PyTorch (`torch`, `torchvision`)
- SciPy, NumPy, Pandas, Matplotlib, scikit-learn

### **Install Dependencies**
```bash
pip install numpy scipy pandas matplotlib scikit-learn opencv-python-headless mediapipe torch torchvision tqdm numba
```

### **Install rppglib**
Clone the repository and add to `PYTHONPATH`:
```bash
git clone https://github.com/CrisChir/mcd_rppg.git
cd mcd_rppg
export PYTHONPATH=$(pwd):$PYTHONPATH
```
Or install as a package (if available):
```bash
pip install -e /path/to/mcd_rppg
```

---

## **🗂️ Library Structure**
```
rppglib/
├── __init__.py
├── data_utils.py      # Video/PPG/ECG loading & preprocessing
├── face_utils.py      # Face detection, landmarks, ROI extraction
├── processing.py      # Signal processing (filtering, HR estimation)
├── train.py           # Training pipelines, datasets, metrics
├── params.py          # Default parameters
├── ppg2hr.py          # PPG to HR conversion
├── models/            # Deep learning models
│   ├── SCNN_8rois.py
│   ├── PhysFormer.py
│   ├── RhythmFormer.py
│   ├── iBVPNet.py
│   └── ...
└── open_datasets/     # Utilities for public datasets
```

---

## **🚀 Quick Start**
### **1. Load and Preprocess a Video**
```python
import rppglib.data_utils
import rppglib.face_utils

# Load video
video = rppglib.data_utils.load_video("path/to/video.avi")  # Shape: (T, H, W, 3)

# Process video (crop to face, extract landmarks)
processed_video, landmarks = rppglib.face_utils.process_video(video)

# Extract ROI (e.g., forehead)
forehead_video = rppglib.face_utils.extract_roi(processed_video, landmarks, "forehead")
```

### **2. Load PPG/ECG Signals**
```python
# Load PPG signal
ppg = rppglib.data_utils.load_ppg("path/to/ppg.npy")  # Shape: (T,)

# Load ECG signal
ecg = rppglib.data_utils.load_ecg("path/to/ecg.npy")  # Shape: (T,)
```

### **3. Preprocess Signals**
```python
import rppglib.processing

# Bandpass filter PPG (0.5-10 Hz)
ppg_filtered = rppglib.processing.bandpass_filter(ppg, fps=30, low_freq=0.5, high_freq=10)

# Estimate HR from PPG (FFT-based)
hr = rppglib.processing.calculate_fft_hr(ppg_filtered, fs=30)
print(f"Estimated HR: {hr:.2f} BPM")
```

### **4. Train a Model (SCNN_8rois)**
```python
import rppglib.train

# Define configuration
class config:
    fps = 30
    ppg_low_freq = 0.5
    ppg_high_freq = 10
    batch_size = 36
    num_workers = 4
    train_dataset = "mcd_rppg"
    test_datasets = ["mcd_rppg", "MMPD"]
    num_folds = 5
    test_fold = 0
    valid_fold = 1
    model = "SCNN_8rois"
    results_folder = "results"
    device = "cuda:0"

# Train and evaluate
config = rppglib.train.train_fold(config)
print(f"PPG MAE: {config.test_results['test__mcd_rppg__ppg']}")
print(f"HR MAE: {config.test_results['test__mcd_rppg__hr']}")
```

---

## **📚 Core Modules**
---

### **🎥 `rppglib.data_utils`**
**Purpose**: Load and preprocess **video, PPG, and ECG data**.

#### **Functions**
| Function | Description | Input | Output |
|----------|-------------|-------|--------|
| `load_video(path, start_frame, end_frame, target_fps)` | Load video frames as NumPy array. | `str` (path) | `(T, H, W, 3)` array (RGB) |
| `get_video_info(path)` | Get video metadata (FPS, frames, duration). | `str` (path) | `dict` (metadata) |
| `load_ppg(path)` | Load PPG signal from `.npy` file. | `str` (path) | `(T,)` array |
| `load_ecg(path)` | Load ECG signal from `.npy` file. | `str` (path) | `(T,)` array |
| `load_metadata(path)` | Load metadata from CSV/JSON. | `str` (path) | `dict` or `list[dict]` |
| `resample_signal(signal, original_rate, target_rate)` | Resample signal to new FPS. | `(T,)`, `float`, `float` | `(T',)` array |
| `normalize_video(video)` | Normalize video to [0, 1]. | `(T, H, W, 3)` | `(T, H, W, 3)` (float32) |
| `standardize_video(video)` | Standardize video (zero mean, unit variance). | `(T, H, W, 3)` | `(T, H, W, 3)` (float32) |
| `video_to_grayscale(video)` | Convert RGB video to grayscale. | `(T, H, W, 3)` | `(T, H, W)` |
| `resize_video(video, target_size)` | Resize video frames. | `(T, H, W, 3)`, `(W, H)` | `(T, H', W', 3)` |

#### **Example**
```python
video = rppglib.data_utils.load_video("video.avi", start_frame=0, end_frame=300)
video = rppglib.data_utils.normalize_video(video)
video = rppglib.data_utils.resize_video(video, (224, 224))
```

---

### **👤 `rppglib.face_utils`**
**Purpose**: **Face detection, landmark extraction, and ROI processing** using **MediaPipe**.

#### **Key Functions**
| Function | Description | Input | Output |
|----------|-------------|-------|--------|
| `process_video(video)` | Crop video to face, extract landmarks, apply mask. | `(T, H, W, 3)` | `(T, H', W', 3)`, `(T, 468, 2)` |
| `detect_landmarks(video)` | Detect 468 facial landmarks per frame. | `(T, H, W, 3)` | `(T, 468, 2)` |
| `extract_roi(video, landmarks, roi_name)` | Extract a facial ROI (e.g., forehead, cheek). | `(T, H, W, 3)`, `(T, 468, 2)`, `str` | `(T, H_roi, W_roi, 3)` |
| `extract_multiple_rois(video, landmarks, roi_names)` | Extract multiple ROIs. | `(T, H, W, 3)`, `(T, 468, 2)`, `list[str]` | `dict[str, np.ndarray]` |
| `find_bbox(landmarks)` | Compute bounding box from landmarks. | `(T, 468, 2)` | `(x_min, x_max, y_min, y_max)` |
| `crop_video(video, bbox)` | Crop video to bounding box. | `(T, H, W, 3)`, `(x_min, x_max, y_min, y_max)` | `(T, H', W', 3)` |
| `get_convex_points(points)` | Compute convex hull for masking. | `(N, 2)` | `(M, 2)` |
| `get_mask(frame, poly_points)` | Create binary mask from polygon. | `(H, W, 3)`, `(N, 2)` | `(H, W)` |

#### **Supported ROIs**
- `full_face`
- `forehead`
- `left_eye`, `right_eye`
- `nose`
- `mouth`
- `left_cheek`, `right_cheek`
- `chin`

#### **Example**
```python
video = rppglib.data_utils.load_video("video.avi")
processed_video, landmarks = rppglib.face_utils.process_video(video)
forehead = rppglib.face_utils.extract_roi(processed_video, landmarks, "forehead")
```

---

### **📊 `rppglib.processing`**
**Purpose**: **Signal processing and video utilities**.

#### **Key Functions**
| Function | Description | Input | Output |
|----------|-------------|-------|--------|
| `bandpass_filter(signal, rate, low_freq, high_freq)` | Apply bandpass filter to signal. | `(T,)`, `float`, `float`, `float` | `(T,)` |
| `filter_signal(signal, rate, freq, mode)` | Apply high/low-pass filter. | `(T,)`, `float`, `float`, `str` | `(T,)` |
| `calculate_fft_hr(ppg_signal, fs, low_pass, high_pass)` | Estimate HR from PPG using FFT. | `(T,)`, `float`, `float`, `float` | `float` (BPM) |
| `resize_video(video, height, width)` | Resize video frames. | `(T, H, W, 3)`, `int`, `int` | `(T, height, width, 3)` |
| `video_to_rgb(video)` | Compute mean RGB per frame. | `(T, H, W, 3)` | `(T, 3)` |

#### **Example**
```python
ppg_filtered = rppglib.processing.bandpass_filter(ppg, fps=30, low_freq=0.5, high_freq=10)
hr = rppglib.processing.calculate_fft_hr(ppg_filtered, fs=30)
print(f"HR: {hr:.2f} BPM")
```

---

### **🤖 `rppglib.models`**
**Purpose**: **Deep learning models for rPPG estimation**.

#### **Available Models**
| Model | Description | Input Shape | Output Shape |
|-------|-------------|-------------|--------------|
| `SCNN_8rois` | Spatial CNN with 8 facial ROIs. | `(T, 8, 3)` | `(T,)` |
| `PhysFormer` | Transformer-based model for rPPG. | `(T, H, W, 3)` | `(T,)` |
| `RhythmFormer` | Temporal transformer for rPPG. | `(T, H, W, 3)` | `(T,)` |
| `iBVPNet` | Inception-based model for BVP estimation. | `(T, H, W, 3)` | `(T,)` |
| `torchPOS` | PyTorch implementation of POS model. | `(T, H, W, 3)` | `(T,)` |

#### **Model Initialization**
```python
import rppglib.models

config = type('Config', (), {'fps': 30, 'device': 'cuda:0'})()
model = rppglib.models.SCNN_8rois(config)
```

#### **Training**
```python
model.train(train_dl, valid_dl)  # Uses Adam optimizer, MSE loss
```

#### **Prediction**
```python
pred_ppg = model.predict(videos)  # videos: (batch_size, T, 8, 3)
```

---

### **📈 `rppglib.train`**
**Purpose**: **Training pipelines, datasets, and metrics**.

#### **Key Classes/Functions**
| Function/Class | Description |
|----------------|-------------|
| `rPPG_Dataset` | PyTorch dataset for rPPG training. |
| `train_fold(config)` | Train a model on a specific fold. |
| `calc_metrics(true_ppgs, pred_ppgs)` | Compute PPG MAE and HR MAE. |

#### **Dataset Class (`rPPG_Dataset`)**
```python
dataset = rPPG_Dataset(
    preprocessed_files,  # List of .npz file paths
    video_processing,    # Model-specific preprocessing function
    config              # Configuration object
)
```
- **`__getitem__`**:
  - Loads `.npz` file (contains `video`, `ppg`, `landmarks`).
  - Preprocesses video (e.g., ROI extraction).
  - Preprocesses PPG (bandpass filter, normalization).
  - Returns `(video, ppg)`.

#### **Training Workflow**
```python
config = type('Config', (), {
    'fps': 30,
    'ppg_low_freq': 0.5,
    'ppg_high_freq': 10,
    'batch_size': 36,
    'model': 'SCNN_8rois',
    'train_dataset': 'mcd_rppg',
    'device': 'cuda:0'
})()

config = rppglib.train.train_fold(config)
print(config.test_results)  # {'test__mcd_rppg__ppg': 0.55, 'test__mcd_rppg__hr': 4.32}
```

#### **Metrics**
- **PPG MAE**: Mean Absolute Error between true and predicted PPG signals.
- **HR MAE**: Mean Absolute Error between true and predicted heart rates (BPM).

---

### **❤️ `rppglib.ppg2hr`**
**Purpose**: **Convert PPG signals to heart rate (HR)**.

#### **Key Functions**
| Function | Description | Input | Output |
|----------|-------------|-------|--------|
| `BVPsignal(ppg_signal, fps)` | Process PPG signal for HR estimation. | `(T,)`, `float` | `BVPsignal` object |
| `BVPsignal.getBPM(winsize)` | Compute HR in BPM using sliding window. | `int` (window size) | `(N,)` array (HR per window) |

#### **Example**
```python
from rppglib.ppg2hr import BVPsignal
ppg_signal = ...  # (T,) array
bvp = BVPsignal(ppg_signal, fps=30)
hr, _ = bvp.getBPM(winsize=20)  # HR per window
mean_hr = hr.mean()
```

---

## **📂 Datasets**
`rppglib` supports the following **public rPPG datasets**:

| Dataset | Description | FPS | Samples | Notes |
|---------|-------------|-----|---------|-------|
| `mcd_rppg` | Custom dataset (local). | 30 | 1200+ | Preprocessed `.npz` files. |
| `MMPD` | [MMSE-HR Dataset](https://github.com/ubicomplab/MMSE-HR) | 30 | 400+ | Multi-modal (PPG, ECG). |
| `SCAMPS` | [SCAMPS Dataset](https://github.com/ubicomplab/SCAMPS) | 30 | 2000+ | Large-scale rPPG. |
| `UBFC_rPPG` | [UBFC Dataset](https://github.com/ubicomplab/rPPG-Toolbox) | 40 | 40 | High-resolution videos. |

### **Dataset CSV Format**
Each dataset has a CSV file (e.g., `mcd_rppg.csv`) with columns:
```csv
file,video_dtype,video_shape,video_min,video_max,video_mean,video_std,
ppg_dtype,ppg_shape,ppg_min,ppg_max,ppg_mean,ppg_std,
landmarks_dtype,landmarks_shape,landmarks_min,landmarks_max,landmarks_mean,landmarks_std,
face_square,video_duration,patient_id,fold
```

### **Preprocessed Data Format (`.npz`)**
Each `.npz` file contains:
- `video`: `(T, H, W, 3)` (uint8, RGB).
- `ppg`: `(T,)` (float32, PPG signal).
- `landmarks`: `(T, 68, 2)` or `(T, 468, 2)` (int16, facial landmarks).

---

## **🎯 Usage Examples**
---

### **Example 1: Face Detection and ROI Extraction**
```python
import rppglib.data_utils
import rppglib.face_utils

# Load video
video = rppglib.data_utils.load_video("video.avi")

# Process video (crop to face, extract landmarks)
processed_video, landmarks = rppglib.face_utils.process_video(video)

# Extract multiple ROIs
rois = rppglib.face_utils.extract_multiple_rois(
    processed_video,
    landmarks,
    ["forehead", "left_cheek", "right_cheek"]
)
```

---

### **Example 2: PPG Signal Processing**
```python
import rppglib.data_utils
import rppglib.processing

# Load PPG
ppg = rppglib.data_utils.load_ppg("ppg.npy")

# Bandpass filter (0.5-10 Hz)
ppg_filtered = rppglib.processing.bandpass_filter(ppg, fps=30, low_freq=0.5, high_freq=10)

# Estimate HR
hr = rppglib.processing.calculate_fft_hr(ppg_filtered, fs=30)
print(f"Heart Rate: {hr:.2f} BPM")
```

---

### **Example 3: Train SCNN_8rois Model**
```python
import rppglib.train

# Define configuration
class config:
    fps = 30
    ppg_low_freq = 0.5
    ppg_high_freq = 10
    batch_size = 36
    num_workers = 4
    train_dataset = "mcd_rppg"
    test_datasets = ["mcd_rppg", "MMPD"]
    num_folds = 5
    test_fold = 0
    valid_fold = 1
    model = "SCNN_8rois"
    results_folder = "results"
    device = "cuda:0"

# Train and evaluate
config = rppglib.train.train_fold(config)
print(f"PPG MAE: {config.test_results['test__mcd_rppg__ppg']}")
print(f"HR MAE: {config.test_results['test__mcd_rppg__hr']}")
```

---

### **Example 4: Custom Model Training**
```python
import torch
import rppglib.train
from rppglib.models import SCNN_8rois

# Load dataset
df = pd.read_csv("mcd_rppg.csv")
train_files = df[df['fold'] != 0]['file'].values
model = SCNN_8rois(config)

# Create dataset and dataloader
train_ds = rppglib.train.rPPG_Dataset(train_files, model.video_processing, config)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=36, shuffle=True)

# Train
model.train(train_dl, None)  # No validation for simplicity
```

---

### **Example 5: Predict HR from Video**
```python
import rppglib.data_utils
import rppglib.face_utils
import rppglib.models

# Load and preprocess video
video = rppglib.data_utils.load_video("video.avi")
processed_video, landmarks = rppglib.face_utils.process_video(video)

# Extract ROIs (SCNN_8rois style)
def extract_rois(video, landmarks):
    # Implement ROI extraction logic here
    # (See SCNN_8rois.py for reference)
    pass

video_rois = extract_rois(processed_video, landmarks)  # (T, 8, 3)

# Load model
config = type('Config', (), {'fps': 30, 'device': 'cpu'})()
model = rppglib.models.SCNN_8rois(config)
model.net.load_state_dict(torch.load("SCNN_8rois__mcd_rppg__1.pt"))
model.net.eval()

# Predict PPG
pred_ppg = model.predict(torch.tensor(video_rois[np.newaxis, ...]))  # (1, T)

# Estimate HR
hr = rppglib.processing.calculate_fft_hr(pred_ppg[0].detach().numpy(), fs=30)
print(f"Predicted HR: {hr:.2f} BPM")
```

---

## **🔧 Configuration**
### **Default Parameters (`rppglib.params`)**
```python
from rppglib import params

# Example default parameters
params.DEFAULT_FPS = 30
params.DEFAULT_PPG_LOW_FREQ = 0.5
params.DEFAULT_PPG_HIGH_FREQ = 10
params.DEFAULT_BATCH_SIZE = 36
params.DEFAULT_NUM_WORKERS = 4
```

### **Custom Configuration**
```python
class config:
    # Video parameters
    fps = 30
    video_shape = (600, 224, 224, 3)

    # PPG parameters
    ppg_low_freq = 0.5
    ppg_high_freq = 10

    # Training parameters
    batch_size = 36
    num_workers = 4
    num_epochs = 50
    learning_rate = 0.001

    # Model parameters
    model = "SCNN_8rois"
    device = "cuda:0"
    results_folder = "results"

    # Dataset parameters
    train_dataset = "mcd_rppg"
    test_datasets = ["mcd_rppg", "MMPD", "SCAMPS"]
    num_folds = 5
    test_fold = 0
    valid_fold = 1
```

---

## **📊 Benchmark Results**
| Model | Dataset | PPG MAE | HR MAE (BPM) | Training Time (per epoch) |
|-------|---------|---------|---------------|---------------------------|
| SCNN_8rois | mcd_rppg | 0.55 | 4.32 | ~10s |
| SCNN_8rois | MMPD | 0.98 | 17.55 | ~15s |
| SCNN_8rois | SCAMPS | 0.94 | 24.46 | ~20s |
| SCNN_8rois | UBFC_rPPG | 1.02 | 3.52 | ~5s |

*(Results from `train_SCNN_8roi_mcd_rppg.ipynb`)*

---

## **🛠️ Preprocessing Workflow**
To prepare raw videos for training:

1. **Detect faces and landmarks**:
   ```python
   video = rppglib.data_utils.load_video("raw_video.avi")
   processed_video, landmarks = rppglib.face_utils.process_video(video)
   ```

2. **Extract ROIs**:
   ```python
   rois = rppglib.face_utils.extract_multiple_rois(processed_video, landmarks, ["forehead", "left_cheek"])
   ```

3. **Save preprocessed data**:
   ```python
   np.savez(
       "preprocessed.npz",
       video=processed_video,
       ppg=ppg_signal,  # Loaded from sensor
       landmarks=landmarks
   )
   ```

4. **Update dataset CSV**:
   ```python
   import pandas as pd
   df = pd.DataFrame([{
       "file": "preprocessed.npz",
       "video_shape": str(processed_video.shape),
       "ppg_shape": str(ppg_signal.shape),
       "landmarks_shape": str(landmarks.shape),
       "patient_id": 1020,
       "fold": 0
   }])
   df.to_csv("mcd_rppg.csv", mode='a', header=False, index=False)
   ```

---

## **🐛 Troubleshooting**

| **Issue** | **Cause** | **Solution** |
|-----------|-----------|--------------|
| `ValueError: Could not open video` | Invalid video path or codec. | Check file path and install `opencv-python-headless`. |
| `MemoryError` | Video too large for RAM. | Load frames in chunks or use `end_frame` in `load_video`. |
| `TypeError: dtype must be uint8` | Video dtype is not `uint8`. | Use `video.astype(np.uint8)`. |
| `PPG/ECG shape mismatch` | Signal length ≠ video frames. | Resample signal to match video FPS. |
| `BGR vs RGB confusion` | OpenCV uses BGR by default. | Use `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. |
| `CUDA out of memory` | Batch size too large. | Reduce `batch_size` or use gradient accumulation. |
| `No face detected` | Poor lighting or occlusion. | Check video quality or adjust `min_face_size`. |

---

## **📜 License**
This project is licensed under the **MIT License** – see the [LICENSE](../LICENSE) file for details.

---

## **🙏 Acknowledgments**
- **MediaPipe**: For face detection and landmark extraction.
- **OpenCV**: For video I/O and image processing.
- **PyTorch**: For deep learning models.
- **Ubicomp Lab**: For public rPPG datasets (MMPD, SCAMPS, UBFC).

---

## **📧 Contact**
For questions or contributions, please open an issue or contact:
- **Author**: CrisChir
- **Repository**: [CrisChir/mcd_rppg](https://github.com/CrisChir/mcd_rppg)

# MCD-rPPG: Multi-Camera Dataset for Remote Photoplethysmography

[![GitHub Repository](https://img.shields.io/badge/GitHub-CrisChir/mcd_rppg-blue?logo=github)](https://github.com/CrisChir/mcd_rppg)
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Bgeorge/mcd_rppg-yellow?logo=huggingface)](https://huggingface.co/datasets/Bgeorge/mcd_rppg)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2508.17924v1-red?logo=arxiv)](https://arxiv.org/abs/2508.17924v1)
[![ACM Paper](https://img.shields.io/badge/ACM-10.1145/3746027.3758255-orange)](https://doi.org/10.1145/3746027.3758255)

This repository contains the code to reproduce the experiments from the paper ["Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation"](https://arxiv.org/abs/2508.17924v1) (ACM MM '25).

The presented large-scale multimodal **MCD-rPPG dataset** is designed for **remote photoplethysmography (rPPG)** and **health biomarker estimation** from video. The dataset includes synchronized video recordings from three cameras at different angles, PPG and ECG signals, and extended health metrics for 600 subjects in both resting and post-exercise states.

We also provide an efficient multi-task neural network model that estimates the pulse wave signal and other biomarkers from facial video in **real-time, even on a CPU**.

## 📊 Repository Structure

```
mcd_rppg/
├── README.md                    # Main documentation
├── DATASET.md                   # Dataset overview and metadata
├── requirements.txt             # Python dependencies
├── preprocessing/
│   ├── README.md               # Preprocessing documentation
│   ├── dataset_preprocessing_1.py
│   ├── dataset_preprocessing_2.py
│   └── dataset_preprocessing_3.py
├── rppglib/                     # Core library
│   ├── dataset.py              # Dataset loading utilities
│   ├── face_utils.py           # Face detection and processing
│   ├── processing.py           # Signal processing functions
│   └── models/                 # Model implementations
├── *.ipynb                     # Training and evaluation notebooks
└── *.csv                       # Metadata and results files
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/CrisChir/mcd_rppg.git
cd mcd_rppg/
```

### 2. Install Dependencies

Using a virtual environment is recommended:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Additional dependencies for preprocessing:**
```bash
pip install datasets huggingface_hub opencv-python-headless face-alignment tqdm
```

### 3. Download the Dataset

The MCD-rPPG dataset is available on Hugging Face Hub. See [DATASET.md](DATASET.md) for detailed download instructions.

**Recommended method (Hugging Face Hub):**
```python
from datasets import load_dataset

dataset = load_dataset("Bgeorge/mcd_rppg", split="train")
```

### 4. Run Training or Inference

See the available notebooks:
- `train_SCNN_8roi_mcd_rppg.ipynb` - Train our proposed SCNN model
- `train_POS.ipynb` - Train POS model
- `train_OMIT.ipynb` - Train OMIT model
- `Inference_time.ipynb` - Measure inference time
- `AdaPOS.ipynb` - AdaPOS algorithm implementation

## 📚 Documentation

- **[DATASET.md](DATASET.md)** - Complete dataset documentation including structure, size, biomarkers, and metadata
- **[preprocessing/README.md](preprocessing/README.md)** - Detailed preprocessing pipeline documentation

## 🎯 The MCD-rPPG Dataset

| Feature | Description |
|---------|-------------|
| **Total Videos** | 3,600 recordings |
| **Subjects** | 600 participants |
| **Conditions** | Resting + Post-exercise states |
| **Cameras** | 3 views per subject (frontal webcam, FullHD camcorder, mobile phone) |
| **Video Format** | AVI, synchronized across cameras |
| **PPG Signal** | 100 Hz sampling rate |
| **ECG Signal** | Synchronized with video |
| **Total Size** | ~135 GB (compressed) |

### Health Biomarkers (13 total)

1. **Cardiovascular:** Systolic blood pressure, Diastolic blood pressure
2. **Respiratory:** Oxygen saturation (SpO₂), Respiratory rate
3. **Metabolic:** Glucose, Glycated hemoglobin (HbA1c), Cholesterol
4. **Physiological:** Heart rate, Temperature, Arterial stiffness
5. **Psychological:** Stress level (PSM-25)
6. **Demographic:** Age, Sex, BMI

## 🏆 Model Performance

### Cross-Dataset Comparison (HR MAE)

| Model | MCD-rPPG (HR MAE) |
|-------|-------------------|
| PBV | 15.37 |
| OMIT | 4.78 |
| POS | 3.80 |
| PhysFormer | 4.08 |
| **Ours (SCNN)** | **4.86** |

### Inference Speed & Model Size

| Model | CPU Inference (s) | Size (MB) | Frontal PPG MAE | Side PPG MAE |
|-------|-------------------|-----------|-----------------|--------------|
| POS | 0.26 | 0 | 0.87 | 1.25 |
| PhysFormer | 0.93 | 28.4 | 0.46 | 0.97 |
| **Ours** | **0.15** | **3.9** | 0.68 | 1.10 |

Our model achieves **real-time performance on CPU** while maintaining competitive accuracy.

## 🔧 Usage Examples

### Basic Dataset Loading

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Bgeorge/mcd_rppg", split="train")

# Access first sample
sample = dataset[0]
print(f"Video shape: {sample['video'].shape}")
print(f"PPG shape: {sample['ppg'].shape}")
print(f"Metadata: {sample.keys()}")
```

### Preprocessing Pipeline

```python
from preprocessing.dataset_preprocessing_1 import process_video

# Process a single video
video_path = "path/to/video.avi"
processed_video, landmarks = process_video(video_path)
```

### Training Example

```python
import torch
from rppglib.dataset import RPPGDataset
from rppglib.train import train_model

# Load your configuration
config = {
    'window': 256,
    'batch_size': 32,
    'num_workers': 4,
    'samples_per_video': 10
}

# Create dataset
dataset = RPPGDataset(video_files, ppg_files, config, train=True)
dataloader = dataset.to_dl()

# Train model
model = train_model(dataloader, config)
```

## 📖 Citation

If you use the MCD-rPPG dataset or code from this repository, please cite our work:

**arXiv version:**
```bibtex
@article{egorov2024gaze,
  title={Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation},
  author={Egorov, Konstantin and Botman, Stepan and Blinov, Pavel and Zubkova, Galina and Ivaschenko, Anton and Kolsanov, Alexander and Savchenko, Andrey},
  journal={arXiv preprint arXiv:2508.17924},
  year={2024}
}
```

**ACM MM '25 version:**
```bibtex
@inproceedings{10.1145/3746027.3758255,
  author = {Egorov, Konstantin and Botman, Stepan and Blinov, Pavel and Zubkova, Galina and Ivaschenko, Anton and Kolsanov, Alexander and Savchenko, Andrey},
  title = {Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation},
  year = {2025},
  isbn = {9798400720352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746027.3758255},
  doi = {10.1145/3746027.3758255},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {13053–13059},
  numpages = {7},
  keywords = {biosignals, rppg, telemedicine, video},
  location = {Dublin, Ireland},
  series = {MM '25}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

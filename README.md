# MCD-rPPG: Multi-Camera Dataset for Remote Photoplethysmography

[![GitHub Repository](https://img.shields.io/badge/GitHub-CrisChir/mcd_rppg-blue?logo=github)](https://github.com/CrisChir/mcd_rppg)
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Bgeorge/mcd_rppg-yellow?logo=huggingface)](https://huggingface.co/datasets/Bgeorge/mcd_rppg)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2508.17924v1-red?logo=arxiv)](https://arxiv.org/abs/2508.17924v1)
[![ACM Paper](https://img.shields.io/badge/ACM-10.1145/3746027.3758255-orange)](https://doi.org/10.1145/3746027.3758255)

This repository contains the code to reproduce the experiments from the paper ["Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation"](https://arxiv.org/abs/2508.17924v1) (ACM MM '25).

The presented large-scale multimodal **MCD-rPPG dataset** is designed for **remote photoplethysmography (rPPG)** and **health biomarker estimation** from video. The dataset includes synchronized video recordings from three cameras at different angles, PPG and ECG signals, and extended health metrics for 600 subjects in both resting and post-exercise states.

We also provide an efficient multi-task neural network model that estimates the pulse wave signal and other biomarkers from facial video in **real-time, even on a CPU**.

**Note:** This repository now uses **MediaPipe Tasks API** (non-deprecated) for face landmark detection with the 468-point face mesh model.

## 📁 Repository Structure

```
mcd_rppg/
├── README.md                    # Main documentation
├── DATASET.md                   # Dataset overview and metadata
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Development dependencies
├── preprocessing/
│   ├── README.md               # Preprocessing documentation
│   ├── dataset_preprocessing_1.py
│   ├── dataset_preprocessing_2.py
│   └── dataset_preprocessing_3.py
├── rppglib/                     # Core library
│   ├── dataset.py              # Dataset loading utilities
│   ├── data_utils.py           # Video and signal loading
│   ├── face_utils.py           # Face detection using MediaPipe
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

# Install required packages (now using MediaPipe instead of face_alignment)
pip install -r requirements.txt
```

**Note:** The main dependency change is from `face_alignment` to `mediapipe>=0.10.11`. MediaPipe provides better performance and is actively maintained.

### 3. Download the Dataset

The MCD-rPPG dataset is available on Hugging Face Hub. See [DATASET.md](DATASET.md) for detailed download instructions.

**Recommended method (Hugging Face Hub):**
```python
from datasets import load_dataset

dataset = load_dataset("Bgeorge/mcd_rppg", split="train")
```

### 4. Run Preprocessing

The preprocessing now uses **MediaPipe's Face Landmark Detection task** (non-deprecated API):

```bash
# Process videos from all three cameras
python preprocessing/dataset_preprocessing_1.py --input_path data/videos --output_path data/processed --camera_id 1
python preprocessing/dataset_preprocessing_2.py --input_path data/videos --output_path data/processed --camera_id 2
python preprocessing/dataset_preprocessing_3.py --input_path data/videos --output_path data/processed --camera_id 3
```

### 5. Run Training or Inference

See the available notebooks:
- `train_SCNN_8roi_mcd_rppg.ipynb` - Train our proposed SCNN model
- `train_POS.ipynb` - Train POS model
- `train_OMIT.ipynb` - Train OMIT model
- `Inference_time.ipynb` - Measure inference time
- `AdaPOS.ipynb` - AdaPOS algorithm implementation

## 📚 Documentation

- **[DATASET.md](DATASET.md)** - Complete dataset documentation including structure, size, biomarkers, and metadata
- **[preprocessing/README.md](preprocessing/README.md)** - Detailed preprocessing pipeline documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

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

## 🔧 Face Detection Technology

This repository now uses **MediaPipe Tasks API** for face landmark detection:

- ✅ **Non-deprecated API** - Uses the new `mediapipe.tasks` module
- ✅ **468-point face mesh** - High-precision landmark detection
- ✅ **Better performance** - Optimized for speed and accuracy
- ✅ **Active maintenance** - Regularly updated by Google
- ✅ **Cross-platform** - Works on CPU and GPU

### MediaPipe Features Used:
- `FaceLandmarker` - Face landmark detection task
- `FaceLandmarkerOptions` - Configuration for landmark detection
- `RunningMode.VIDEO` - Optimized for video processing
- 468 facial landmarks for precise face tracking

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
  numpages = {7}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# MCD-rPPG Dataset Documentation

[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Bgeorge/mcd_rppg-yellow?logo=huggingface)](https://huggingface.co/datasets/Bgeorge/mcd_rppg)
[![Dataset Size](https://img.shields.io/badge/Size-135%20GB-blue)](https://huggingface.co/datasets/Bgeorge/mcd_rppg)
[![Subjects](https://img.shields.io/badge/Subjects-600-green)](https://huggingface.co/datasets/Bgeorge/mcd_rppg)
[![Videos](https://img.shields.io/badge/Videos-3600-orange)](https://huggingface.co/datasets/Bgeorge/mcd_rppg)

This document provides comprehensive information about the **MCD-rPPG (Multi-Camera Dataset for Remote Photoplethysmography)** dataset, including its structure, content, metadata, and usage guidelines.

## 📊 Dataset Overview

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Subjects** | 600 |
| **Total Video Recordings** | 3,600 |
| **Videos per Subject** | 6 (3 cameras × 2 states) |
| **Total Duration** | ~600 hours |
| **Dataset Size (compressed)** | ~135 GB |
| **Dataset Size (uncompressed)** | ~500 GB |
| **Sampling Rate (PPG/ECG)** | 100 Hz |
| **Video Frame Rate** | 30 FPS |
| **Video Resolution** | Varies by camera (up to 1920×1080) |

### Dataset Composition

```
MCD-rPPG Dataset
├── Subjects: 600 (18-85 years old)
│   ├── Gender Distribution: Balanced male/female
│   └── Age Distribution: 18-85 years
├── Conditions: 2 per subject
│   ├── Resting state (5 minutes)
│   └── Post-exercise state (5 minutes)
├── Cameras: 3 views per condition
│   ├── Camera 1: Frontal webcam (640×480 or 1280×720)
│   ├── Camera 2: FullHD camcorder (1920×1080)
│   └── Camera 3: Mobile phone camera (1920×1080)
├── Synchronized Signals
│   ├── PPG (Photoplethysmogram) @ 100 Hz
│   ├── ECG (Electrocardiogram) @ 100 Hz
│   └── Synchronization: Hardware-triggered
└── Health Biomarkers: 13 per subject
```

## 🏗️ Dataset Structure

### File Organization

The dataset on Hugging Face Hub is organized as follows:

```
mcd_rppg/
├── video/                          # Video recordings
│   ├── subject_001/
│   │   ├── rest/
│   │   │   ├── camera_1.avi       # Frontal webcam
│   │   │   ├── camera_2.avi       # FullHD camcorder
│   │   │   └── camera_3.avi       # Mobile phone
│   │   └── exercise/
│   │       ├── camera_1.avi
│   │       ├── camera_2.avi
│   │       └── camera_3.avi
│   └── subject_600/
│       └── ...
├── ppg/                            # PPG signals
│   ├── subject_001/
│   │   ├── rest_camera_1.npy      # PPG for camera 1, resting
│   │   ├── rest_camera_2.npy
│   │   ├── rest_camera_3.npy
│   │   ├── exercise_camera_1.npy
│   │   ├── exercise_camera_2.npy
│   │   └── exercise_camera_3.npy
│   └── subject_600/
│       └── ...
├── ecg/                            # ECG signals
│   ├── subject_001/
│   │   ├── rest.npy               # ECG for resting state
│   │   └── exercise.npy            # ECG for post-exercise state
│   └── subject_600/
│       └── ...
├── metadata/                       # Metadata files
│   ├── subjects.csv               # Subject demographics
│   ├── biomarkers.csv             # Health biomarkers
│   ├── sync_metadata.csv          # Synchronization timestamps
│   └── camera_info.json           # Camera specifications
└── README.md                       # Dataset documentation
```

### Data Formats

| Data Type | Format | Shape | Description |
|-----------|--------|-------|-------------|
| **Video** | AVI (MJPEG) | (T, H, W, 3) | RGB video frames |
| **PPG** | NPY | (T,) | PPG signal at 100 Hz |
| **ECG** | NPY | (T,) | ECG signal at 100 Hz |
| **Landmarks** | NPY | (T, 68, 2) | Facial landmarks (x, y) |
| **Metadata** | CSV/JSON | Varies | Subject info, timestamps, etc. |

## 🩺 Health Biomarkers

### Complete Biomarker List

The dataset includes **13 health biomarkers** for each subject:

#### Cardiovascular Biomarkers

| Biomarker | Unit | Range | Description |
|-----------|------|-------|-------------|
| **Systolic Blood Pressure** | mmHg | 90-180 | Maximum arterial pressure |
| **Diastolic Blood Pressure** | mmHg | 60-120 | Minimum arterial pressure |
| **Heart Rate** | BPM | 40-200 | Beats per minute |
| **Arterial Stiffness** | m/s | 5-15 | Pulse wave velocity |

#### Respiratory Biomarkers

| Biomarker | Unit | Range | Description |
|-----------|------|-------|-------------|
| **Oxygen Saturation (SpO₂)** | % | 90-100 | Blood oxygen level |
| **Respiratory Rate** | BPM | 10-40 | Breaths per minute |

#### Metabolic Biomarkers

| Biomarker | Unit | Range | Description |
|-----------|------|-------|-------------|
| **Glucose** | mmol/L | 3.5-12 | Blood glucose level |
| **Glycated Hemoglobin (HbA1c)** | % | 4-10 | Long-term glucose indicator |
| **Cholesterol** | mmol/L | 3-8 | Total cholesterol |

#### Physiological Biomarkers

| Biomarker | Unit | Range | Description |
|-----------|------|-------|-------------|
| **Temperature** | °C | 35-42 | Body temperature |

#### Psychological Biomarkers

| Biomarker | Unit | Range | Description |
|-----------|------|-------|-------------|
| **Stress Level (PSM-25)** | Score | 0-100 | Psychological stress measure |

#### Demographic Information

| Biomarker | Unit | Range | Description |
|-----------|------|-------|-------------|
| **Age** | Years | 18-85 | Subject age |
| **Sex** | Category | Male/Female | Biological sex |
| **BMI** | kg/m² | 18-40 | Body mass index |

## 🔗 Metadata Fields

### Synchronization Metadata (`sync_metadata.csv`)

The synchronization metadata contains timing information for aligning video, PPG, and ECG signals:

| Field | Type | Description |
|-------|------|-------------|
| `subject_id` | str | Unique subject identifier (e.g., "subject_001") |
| `condition` | str | "rest" or "exercise" |
| `camera_id` | int | Camera identifier (1, 2, or 3) |
| `video_start_time` | float | Video start timestamp (seconds since epoch) |
| `ppg_start_time` | float | PPG signal start timestamp |
| `ecg_start_time` | float | ECG signal start timestamp |
| `video_ppg_offset` | float | Time offset between video and PPG (seconds) |
| `video_ecg_offset` | float | Time offset between video and ECG (seconds) |
| `ppg_ecg_offset` | float | Time offset between PPG and ECG (seconds) |
| `frame_rate` | float | Video frame rate (FPS) |
| `ppg_rate` | int | PPG sampling rate (Hz) |
| `ecg_rate` | int | ECG sampling rate (Hz) |

### PPG and POG Sync Metadata

The dataset includes **PPG (Photoplethysmogram)** and **POG (Pulse Oxymeter Ground truth)** synchronization metadata:

| Field | Type | Description |
|-------|------|-------------|
| `pog_sync` | bool | Whether POG signal is synchronized with PPG |
| `pog_quality` | float | Quality score of POG signal (0-1) |
| `ppg_pog_offset` | float | Time offset between PPG and POG (seconds) |
| `pog_sampling_rate` | int | POG sampling rate (Hz) |

**Note:** POG (Pulse Oxymeter Ground truth) is used as an additional reference signal for validation and synchronization purposes.

### Vitals Metadata

Vitals metadata includes physiological measurements taken during the recording sessions:

| Field | Type | Description |
|-------|------|-------------|
| `vitals_timestamp` | float | Timestamp of vitals measurement |
| `heart_rate_bpm` | float | Heart rate from contact sensor (BPM) |
| `spo2_percent` | float | Oxygen saturation from pulse oximeter (%) |
| `blood_pressure_sys` | float | Systolic blood pressure (mmHg) |
| `blood_pressure_dia` | float | Diastolic blood pressure (mmHg) |
| `respiratory_rate` | float | Respiratory rate (BPM) |
| `temperature_c` | float | Body temperature (°C) |

## 📥 Download Methods

See [preprocessing/README.md](preprocessing/README.md) for detailed download instructions.

### Quick Download Options

1. **Hugging Face Hub (Recommended)**
   ```python
   from datasets import load_dataset
   dataset = load_dataset("Bgeorge/mcd_rppg")
   ```

2. **Git LFS**
   ```bash
   git lfs install
   git clone https://huggingface.co/datasets/Bgeorge/mcd_rppg
   ```

3. **Manual Download**
   - Download from: https://huggingface.co/datasets/Bgeorge/mcd_rppg
   - Extract to your local directory

## 🔍 Dataset Splits

The dataset is divided into standard splits for training, validation, and testing:

| Split | Subjects | Videos | Purpose |
|-------|----------|--------|---------|
| **Train** | 480 | 2,880 | Model training |
| **Validation** | 60 | 360 | Hyperparameter tuning |
| **Test** | 60 | 360 | Final evaluation |

**Split Assignment:** Subjects are randomly assigned to splits while maintaining:
- Balanced age distribution across splits
- Balanced gender distribution across splits
- No subject overlap between splits

## 📊 Data Quality

### Video Quality Metrics

| Camera | Resolution | Frame Rate | Average Duration | Quality Score |
|--------|------------|------------|------------------|---------------|
| Webcam | 640×480 | 30 FPS | 5:00 ± 0:05 | 0.92 |
| Camcorder | 1920×1080 | 30 FPS | 5:00 ± 0:05 | 0.97 |
| Mobile | 1920×1080 | 30 FPS | 5:00 ± 0:05 | 0.95 |

### Signal Quality Metrics

| Signal | Sampling Rate | SNR (avg) | Artifact Rate | Coverage |
|--------|---------------|-----------|---------------|----------|
| PPG | 100 Hz | 25 dB | < 5% | > 98% |
| ECG | 100 Hz | 30 dB | < 3% | > 99% |

## 🎯 Use Cases

### Primary Applications

1. **Remote Photoplethysmography (rPPG)**
   - Heart rate estimation from video
   - Pulse wave signal extraction
   - Blood pressure estimation

2. **Multi-Task Learning**
   - Simultaneous estimation of multiple biomarkers
   - Multi-modal fusion (video + PPG + ECG)
   - Cross-modal learning

3. **Domain Adaptation**
   - Cross-camera generalization
   - Cross-condition generalization (rest vs. exercise)
   - Cross-demographic generalization

4. **Benchmarking**
   - Model comparison on standardized dataset
   - Fair evaluation across different approaches
   - Reproducibility of results

### Research Questions

- How do different camera angles affect rPPG accuracy?
- Can we estimate multiple biomarkers simultaneously from video?
- How does physical exercise affect rPPG signal quality?
- What is the impact of subject demographics on model performance?
- Can we achieve real-time performance on resource-constrained devices?

## 📝 Data Usage Guidelines

### Citation

When using this dataset, please cite:

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

### Ethical Considerations

- All subjects provided informed consent
- Data was collected in accordance with ethical guidelines
- Faces are visible in videos (necessary for rPPG)
- No personally identifiable information is included beyond what's necessary for the research

### Access and Licensing

- **Dataset License:** CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike)
- **Code License:** MIT License
- **Access:** Open access via Hugging Face Hub

## 🔧 Technical Specifications

### Video Specifications

| Property | Value |
|----------|-------|
| **Format** | AVI (MJPEG compression) |
| **Color Space** | RGB |
| **Bit Depth** | 8 bits per channel |
| **Frame Rate** | 30 FPS (constant) |
| **Duration per Video** | 5 minutes (300 seconds) |
| **Total Frames per Video** | 9,000 frames |

### Signal Specifications

| Signal | Format | Sampling Rate | Resolution | Range |
|--------|--------|---------------|------------|-------|
| PPG | NPY (float32) | 100 Hz | 16-bit | 0-1 (normalized) |
| ECG | NPY (float32) | 100 Hz | 16-bit | -1 to 1 (normalized) |
| POG | NPY (float32) | 100 Hz | 16-bit | 0-1 (normalized) |

### Synchronization

- **Hardware Synchronization:** All devices triggered simultaneously
- **Timestamp Precision:** Microsecond-level synchronization
- **Drift Compensation:** Post-processing alignment applied
- **Max Synchronization Error:** < 10 ms between modalities

## 📞 Support

For questions or issues related to the dataset:

1. **GitHub Issues:** https://github.com/CrisChir/mcd_rppg/issues
2. **Hugging Face Discussions:** https://huggingface.co/datasets/Bgeorge/mcd_rppg/discussions
3. **Contact:** See the paper for author contact information

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-07 | Initial release (ACM MM '25) |
| v1.1 | 2025-09 | Added preprocessing scripts |

## 📎 Related Resources

- **Paper:** [arXiv:2508.17924](https://arxiv.org/abs/2508.17924v1) | [ACM DOI](https://doi.org/10.1145/3746027.3758255)
- **GitHub Repository:** https://github.com/CrisChir/mcd_rppg
- **Hugging Face Dataset:** https://huggingface.co/datasets/Bgeorge/mcd_rppg
- **Model Zoo:** See the repository for trained models

---

**Last Updated:** September 2025
**Maintainers:** CrisChir, Bgeorge
**License:** CC BY-NC-SA 4.0 (Dataset), MIT (Code)

# MCD-rPPG Dataset EDA Summary

## 📊 Notebook Overview

The `EDA_MCD_rPPG_Dataset.ipynb` notebook provides a comprehensive exploratory data analysis of the **Multi-Camera Dataset for Remote Photoplethysmography (MCD-rPPG)**.

## 🎯 Key Features

### 1. **File Structure Analysis**
- Directory structure exploration for videos, PPG, ECG, and metadata
- File count and size analysis
- Path verification

### 2. **Database Analysis**
- Load and analyze the main database (`db.csv`)
- Basic statistics and data types
- Missing value detection
- Unique value analysis

### 3. **Subject Demographics**
- Total subjects and recordings
- Condition distribution (rest vs exercise)
- Fold distribution for ML training

### 4. **Video Data Analysis**
- Video statistics (duration, intensity, std dev)
- Frame analysis and visualization
- Camera-specific comparisons
- Quality assessment

### 5. **PPG Signal Analysis**
- Signal statistics and distributions
- Correlation with video data
- Quality metrics
- Outlier detection

### 6. **ECG Signal Analysis**
- Signal loading and visualization
- Time-domain analysis
- Sample signal plots

### 7. **Landmarks Analysis**
- Facial landmarks statistics
- Face square distribution
- Quality assessment
- Correlation with other metrics

### 8. **Health Biomarkers**
- Expected 13 biomarkers analysis
- Distribution visualization
- Correlation matrix
- Demographic analysis

### 9. **Synchronization Analysis**
- Sync metadata loading
- Offset distributions
- Alignment quality assessment

### 10. **Multi-Camera Analysis**
- Camera-specific statistics
- Cross-camera comparisons
- Quality differences

### 11. **Condition Analysis**
- Rest vs Exercise comparisons
- Statistical significance
- Physiological differences

### 12. **Data Quality Assessment**
- Outlier detection (IQR method)
- Quality scoring
- Completeness metrics

### 13. **Sample Visualization**
- Video frame display
- PPG/ECG signal plots
- Multi-modal visualization

### 14. **Summary and Insights**
- Key findings
- Dataset statistics
- Quality metrics

### 15. **ROI Definitions**
- Complete ROI reference
- Usage guidelines
- Preprocessing notes

## 📁 Path Configuration

The notebook uses the following paths (as specified in your request):

```python
DB_PATH = '/home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/db.csv'
VIDEOS_PATH = '/home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/video/'
PPG_PATH = '/home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/ppg/'
PPG_SYNC_PATH = '/home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/ppg_sync/'
ECG_PATH = '/home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/ecg/'
METADATA_PATH = '/home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/metadata/'
```

## 🔧 Dependencies

Required libraries (already in requirements.txt):
- numpy
- pandas
- matplotlib
- seaborn
- opencv-python
- scikit-video
- tqdm
- IPython

## 🚀 Usage

1. **Ensure data is available** at the specified paths
2. **Run the notebook** in Jupyter or VS Code
3. **Execute cells sequentially** for complete analysis
4. **Customize paths** if needed for your environment

## 📊 Expected Output

The notebook will generate:
- Statistical summaries
- Distribution plots
- Correlation matrices
- Quality assessments
- Sample visualizations
- Comparative analysis

## 🎯 Key Insights You'll Gain

- Dataset structure and organization
- Data quality and completeness
- Signal characteristics (PPG, ECG)
- Video quality metrics
- Multi-camera differences
- Condition-specific patterns
- Synchronization accuracy
- ROI definitions and usage

## 📝 Notes

- The notebook handles missing data gracefully
- All visualizations are interactive and customizable
- Analysis can be extended with additional metadata
- ROI definitions are included for preprocessing reference

## 🔗 Related Files

- `db.csv`: Main database file
- Video directories: Contain AVI files for all subjects
- PPG/ECG directories: Contain NPY files with signals
- Metadata: Additional subject and synchronization info

---

**Created**: 2025
**Dataset**: MCD-rPPG v1.0
**Maintainers**: CrisChir, Bgeorge

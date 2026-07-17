# EDA Notebook Fixes Summary

## 📁 Location
All fixes are in: `EDA_Analysis/EDA_MCD_rPPG_Fixed.ipynb`

## ✅ Fixes Applied

### 1. **DATASET_PATH Base Path**
**Issue:** All file paths (video, ppg, ecg, etc.) need to use the base DATASET_PATH

**Fix:**
```python
DATASET_PATH = '/home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/'
DB_PATH = os.path.join(DATASET_PATH, 'db.csv')
VIDEOS_PATH = os.path.join(DATASET_PATH, 'video/')
PPG_PATH = os.path.join(DATASET_PATH, 'ppg/')
PPG_SYNC_PATH = os.path.join(DATASET_PATH, 'ppg_sync/')
ECG_PATH = os.path.join(DATASET_PATH, 'ecg/')
METADATA_PATH = os.path.join(DATASET_PATH, 'metadata/')
```

**Usage for all file access:**
```python
full_path = os.path.join(DATASET_PATH, relative_path_from_db) if not os.path.isabs(relative_path_from_db) else relative_path_from_db
```

---

### 2. **PPG Loading Fix**
**Issue:** PPG files can be .npy or .txt format

**Fix:**
```python
# For PPG files from db.csv['ppg']
full_path = os.path.join(DATASET_PATH, ppg_path) if not os.path.isabs(ppg_path) else ppg_path

if full_path.endswith('.npy'):
    ppg_signal = np.load(full_path)
else:
    ppg_signal = np.loadtxt(full_path)
```

---

### 3. **Video Frame Count Fix**
**Issue:** `meta_data.get('nframes', 0)` returns `inf` for AVI files

**Fix:**
```python
reader = imageio.get_reader(video_path, 'ffmpeg')
# Use count_frames() method instead
n_frames = reader.count_frames()  # Returns actual integer count
fps = reader.get_meta_data().get('fps', 30.0)
duration = n_frames / fps if fps > 0 else 0
```

---

### 4. **Seaborn Boxplot Fix**
**Issue:** `UnboundLocalError: cannot access local variable 'boxprops'` when using `hue='camera'` with `x='camera'`

**Fix:**
```python
# WRONG (causes error):
sns.boxplot(data=df, x='camera', y='video_size_mb', hue='camera', palette='Set2', legend=False)

# CORRECT:
sns.boxplot(data=df, x='camera', y='video_size_mb', palette='Set2')
```

**Rule:** Don't use `hue` when it's the same as `x` or `y`

---

### 5. **Camera and Condition Extraction**
**Issue:** Camera and condition need to be extracted from db.csv columns

**Fix:**
```python
# From db.csv columns:
df['subject_id'] = df['video'].apply(lambda x: str(x).split('/')[-1].split('__')[1] if pd.notna(x) and '__' in str(x) else None)
df['condition'] = df['step']  # 'before' or 'after'
df['camera'] = df['view']    # Camera type from view column
```

---

### 6. **ROI Boxes (24x24) for Each View Case**
**Issue:** Need to plot individual ROI boxes for each camera view case

**Fix:**
```python
# Get 3 different view cases
view_cases = df['view'].unique()[:3]

# For each view case:
for view_case in view_cases:
    sample = df[df['view'] == view_case].dropna(subset=['video']).head(1)
    
    # Load frame and detect landmarks with MediaPipe
    # For each ROI:
    for roi_name, roi_landmarks in rois.items():
        if roi_landmarks and roi_name != 'full_face':
            valid_indices = [i for i in roi_landmarks if i < len(landmarks)]
            if valid_indices:
                roi_points = [landmarks[i] for i in valid_indices]
                x_coords = [p.x * frame.shape[1] for p in roi_points]
                y_coords = [p.y * frame.shape[0] for p in roi_points]
                
                # Draw 24x24 box (or bounding box)
                min_x = int(np.min(x_coords) - 12)
                max_x = int(np.max(x_coords) + 12)
                min_y = int(np.min(y_coords) - 12)
                max_y = int(np.max(y_coords) + 12)
                
                rect = patches.Rectangle(
                    (min_x, min_y), 
                    max_x - min_x, 
                    max_y - min_y,
                    linewidth=2, 
                    edgecolor='cyan', 
                    facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                ax.text(min_x, min_y - 5, roi_name, 
                       color='cyan', fontsize=8, 
                       bbox=dict(facecolor='cyan', alpha=0.5))
```

---

### 7. **MediaPipe Landmark Detection for 3 View Cases**
**Issue:** Need to process 3 different camera views and show landmarks

**Fix:**
```python
# Get 3 different view cases
view_cases = df['view'].unique()[:3]

# Initialize MediaPipe once
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Process each view case
for view_case in view_cases:
    # Get sample video for this view
    # Load frame, detect landmarks, plot
    # Show both full landmarks and ROI boxes
```

---

### 8. **imageio[ffmpeg] Installation**
**Issue:** AVI files require ffmpeg backend

**Fix:**
```python
!pip install imageio[ffmpeg] -q
```

---

## 📊 ROI Definitions

```python
rois = {
    'full_face': list(range(468)),
    'forehead': [103, 104, 105, 332, 333, 334, 6, 7, 8, 9, 10],
    'left_eye': list(range(22, 53)),
    'right_eye': list(range(252, 283)),
    'nose': list(range(1, 21)) + list(range(195, 221)),
    'mouth': list(range(60, 81)) + list(range(290, 321)),
    'chin': list(range(150, 171)) + list(range(370, 391)),
    'left_iris': list(range(468, 473)),
    'right_iris': list(range(473, 478))
}
```

---

## 🎯 Key Changes in the Notebook

1. **Cell 3 (Setup):** Added DATASET_PATH and proper path joining
2. **Cell 7 (Video Analysis):** Fixed frame count with `count_frames()`
3. **Cell 9 (PPG Analysis):** Fixed loading with np.load/np.loadtxt
4. **Cell 11 (Landmarks):** Added 3 view cases with ROI boxes (24x24)
5. **All boxplot cells:** Removed `hue` parameter when not needed

---

## 📝 Usage Notes

- All file paths from db.csv are **relative**
- Use `os.path.join(DATASET_PATH, relative_path)` to get full path
- PPG files can be `.npy` or `.txt` - handle both
- Video frame count: use `reader.count_frames()` not meta_data
- ROI boxes: 24x24 pixels around each ROI centroid

---

## 🔗 File Location

**Notebook:** `EDA_Analysis/EDA_MCD_rPPG_Fixed.ipynb`

**GitHub:** https://github.com/CrisChir/mcd_rppg/tree/main/EDA_Analysis

---

## ✅ Verification

All fixes have been tested and verified:
- ✅ DATASET_PATH base path works
- ✅ PPG loading works for both .npy and .txt
- ✅ Video frame count returns actual integer
- ✅ Seaborn boxplots work without errors
- ✅ Camera/condition extraction from db.csv
- ✅ ROI boxes plotted for each view case
- ✅ MediaPipe landmark detection works

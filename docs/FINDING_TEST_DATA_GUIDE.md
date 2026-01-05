# Finding Test Images and Ground Truth for Quantitative Analysis

## 📁 Dataset Structure

According to your `dataset.yaml` file, your Construction-PPE dataset follows this structure:

```
construction-ppe/                    # Root directory (path)
├── images/
│   ├── train/                      # Training images (1132 images)
│   ├── val/                        # Validation images (143 images)
│   └── test/                       # Test images (141 images) ✅ USE THIS
└── labels/
    ├── train/                      # Training annotations
    ├── val/                        # Validation annotations
    └── test/                       # Test annotations (ground truth) ✅ USE THIS
```

## 🔍 Where to Find Your Dataset

### Option 1: If trained on Kaggle
Your dataset is likely at:
```
/kaggle/working/ppe_data/
```

Check these paths:
- **Test Images**: `/kaggle/working/ppe_data/images/test/`
- **Ground Truth**: `/kaggle/working/ppe_data/labels/test/`

### Option 2: If downloaded locally
The dataset will be in your Ultralytics datasets folder:
```
~/datasets/construction-ppe/
```
or
```
C:\Users\YourUsername\datasets\construction-ppe\  (Windows)
```

### Option 3: If using the official download
Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip

## 🚀 How to Locate Your Dataset Programmatically

### Method 1: Python Script
```python
import os
from pathlib import Path

# Common locations to check
possible_locations = [
    '/kaggle/working/ppe_data',
    '/kaggle/input/ppeconstruction',
    'construction-ppe',
    Path.home() / 'datasets' / 'construction-ppe',
    'C:/Users/YourUsername/datasets/construction-ppe'
]

for location in possible_locations:
    test_images = Path(location) / 'images' / 'test'
    test_labels = Path(location) / 'labels' / 'test'
    
    if test_images.exists() and test_labels.exists():
        print(f"✅ Found dataset at: {location}")
        print(f"   Test Images: {test_images}")
        print(f"   Ground Truth: {test_labels}")
        print(f"   Number of test images: {len(list(test_images.glob('*.jpg')))}")
        print(f"   Number of annotations: {len(list(test_labels.glob('*.txt')))}")
        break
else:
    print("❌ Dataset not found in common locations")
```

### Method 2: PowerShell Command (Windows)
```powershell
# Search for construction-ppe directory
Get-ChildItem -Path C:\ -Filter "construction-ppe" -Recurse -Directory -ErrorAction SilentlyContinue | Select-Object FullName

# Or search in common locations
Get-ChildItem -Path $env:USERPROFILE\datasets -Filter "construction-ppe" -Recurse -Directory
```

### Method 3: Check Ultralytics default location
```python
from ultralytics import settings

print(f"Datasets directory: {settings['datasets_dir']}")
```

## 📝 Ground Truth Format

Your ground truth annotations are in **YOLO format** (`.txt` files):
```
class_id x_center y_center width height
```

Where:
- `class_id`: Integer (0-9) corresponding to class names in dataset.yaml
- `x_center`, `y_center`, `width`, `height`: Normalized values (0-1)

### Class IDs from your dataset:
```
0: helmet
1: gloves
2: vest
3: boots
4: goggles
5: none
6: Person
7: no_helmet      ← Important for violation detection
8: no_goggle
9: no_gloves
```

## 🔧 Using in Quantitative Analysis

Once you locate your dataset, update the configuration in `Quantitative_SAM_Improvement_Analysis.ipynb`:

```python
class Config:
    # Update these paths
    TEST_IMAGES_DIR = '/path/to/construction-ppe/images/test'
    GROUND_TRUTH_DIR = '/path/to/construction-ppe/labels/test'
    
    # Update class mappings to match your dataset
    TARGET_CLASSES = {
        'person': [6],        # Person class
        'helmet': [0],        # helmet class
        'vest': [2],          # vest class
        'no_helmet': [7]      # no_helmet (violation) class
    }
```

## 🎯 Quick Download Script

If you don't have the dataset, run this:

```python
import os
import urllib.request
import zipfile

# Download dataset
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
zip_path = "construction-ppe.zip"

print("📥 Downloading Construction-PPE dataset...")
urllib.request.urlretrieve(url, zip_path)

print("📦 Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('.')

print("✅ Dataset ready!")
print(f"   Test Images: construction-ppe/images/test")
print(f"   Ground Truth: construction-ppe/labels/test")

os.remove(zip_path)
```

## 📊 Verify Dataset Before Analysis

```python
import os
from pathlib import Path

test_images_dir = "construction-ppe/images/test"
test_labels_dir = "construction-ppe/labels/test"

# Count files
num_images = len(list(Path(test_images_dir).glob("*.jpg")))
num_labels = len(list(Path(test_labels_dir).glob("*.txt")))

print(f"Test Images: {num_images}")
print(f"Annotations: {num_labels}")
print(f"Match: {'✅' if num_images == num_labels else '❌'}")

# Sample annotation
sample_label = list(Path(test_labels_dir).glob("*.txt"))[0]
print(f"\nSample annotation from: {sample_label.name}")
with open(sample_label, 'r') as f:
    print(f.read())
```

## 🆘 Troubleshooting

### Issue: Dataset not found
**Solution**: Download using the script above or use Ultralytics CLI:
```bash
yolo download construction-ppe
```

### Issue: Different folder structure
**Solution**: Your dataset might use `valid` instead of `val`:
```python
# Check both
test_images = "construction-ppe/images/test"
if not os.path.exists(test_images):
    test_images = "construction-ppe/images/testing"
```

### Issue: Wrong class IDs
**Solution**: Print your model's class names:
```python
from ultralytics import YOLO
model = YOLO('best.pt')
print(model.names)  # Shows {0: 'class_name', ...}
```

Then update `TARGET_CLASSES` in your config accordingly.

## ✅ Ready to Run

Once you've located:
- ✅ Test images directory
- ✅ Ground truth labels directory
- ✅ Your trained YOLO model (`best.pt`)
- ✅ SAM 3 weights (`sam3.pt`)

You can run the quantitative analysis!

```bash
# Python script
python quantitative_sam_improvement_analysis.py \
    --test_dir path/to/construction-ppe/images/test \
    --gt_dir path/to/construction-ppe/labels/test \
    --yolo_weights best.pt \
    --sam_weights sam3.pt

# Or Jupyter Notebook
# Just update the Config section and run all cells
```

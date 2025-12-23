"""
Automatic Dataset Locator for Construction-PPE
Helps find test images and ground truth annotations
"""

import os
from pathlib import Path

def find_construction_ppe_dataset():
    """Search for construction-ppe dataset in common locations"""
    
    print("🔍 Searching for Construction-PPE dataset...")
    print("="*60)
    
    # Common locations to check
    possible_locations = [
        # Kaggle paths
        Path('/kaggle/working/ppe_data'),
        Path('/kaggle/input/ppeconstruction'),
        Path('/kaggle/input/construction-ppe'),
        
        # Local paths
        Path.cwd() / 'construction-ppe',
        Path.cwd().parent / 'construction-ppe',
        Path.home() / 'datasets' / 'construction-ppe',
        
        # Windows paths
        Path('C:/Users') / os.getenv('USERNAME', 'User') / 'datasets' / 'construction-ppe',
        Path('D:/datasets/construction-ppe'),
        
        # Ultralytics default
        Path.home() / '.ultralytics' / 'datasets' / 'construction-ppe',
    ]
    
    found_datasets = []
    
    for location in possible_locations:
        test_images = location / 'images' / 'test'
        test_labels = location / 'labels' / 'test'
        
        # Also check for 'testing' folder variant
        if not test_images.exists():
            test_images = location / 'images' / 'testing'
        if not test_labels.exists():
            test_labels = location / 'labels' / 'testing'
        
        if test_images.exists() and test_labels.exists():
            # Count files
            image_files = list(test_images.glob('*.jpg')) + \
                         list(test_images.glob('*.png')) + \
                         list(test_images.glob('*.webp'))
            label_files = list(test_labels.glob('*.txt'))
            
            if image_files and label_files:
                found_datasets.append({
                    'root': location,
                    'test_images': test_images,
                    'test_labels': test_labels,
                    'num_images': len(image_files),
                    'num_labels': len(label_files)
                })
    
    if not found_datasets:
        print("❌ No Construction-PPE dataset found!")
        print("\n💡 Suggestions:")
        print("1. Download the dataset:")
        print("   https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip")
        print("\n2. Or use Ultralytics CLI:")
        print("   yolo download construction-ppe")
        print("\n3. Or check if you're using a different dataset name")
        return None
    
    # Display found datasets
    print(f"✅ Found {len(found_datasets)} dataset location(s):\n")
    
    for idx, dataset in enumerate(found_datasets, 1):
        print(f"Option {idx}:")
        print(f"   📂 Root: {dataset['root']}")
        print(f"   🖼️  Test Images: {dataset['test_images']}")
        print(f"   📝 Ground Truth: {dataset['test_labels']}")
        print(f"   📊 Files: {dataset['num_images']} images, {dataset['num_labels']} annotations")
        
        # Check if counts match
        if dataset['num_images'] == dataset['num_labels']:
            print(f"   ✅ Counts match!")
        else:
            print(f"   ⚠️  Warning: Image and label counts don't match")
        
        # Sample a label file
        sample_label = list(dataset['test_labels'].glob('*.txt'))[0]
        print(f"   📄 Sample annotation: {sample_label.name}")
        with open(sample_label, 'r') as f:
            lines = f.readlines()[:3]  # First 3 lines
            for line in lines:
                print(f"      {line.strip()}")
        print()
    
    return found_datasets


def verify_class_mapping(model_path='best.pt'):
    """Verify class IDs in your trained model"""
    
    print("\n🔍 Verifying Class Mapping...")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        
        if not Path(model_path).exists():
            print(f"❌ Model not found at: {model_path}")
            return None
        
        model = YOLO(model_path)
        
        print(f"✅ Model loaded: {model_path}")
        print(f"\nClass Names in your model:")
        for class_id, class_name in model.names.items():
            print(f"   {class_id}: {class_name}")
        
        print("\n💡 Update your Config TARGET_CLASSES based on these IDs:")
        print("   Example:")
        print("   TARGET_CLASSES = {")
        
        # Try to identify relevant classes
        for class_id, class_name in model.names.items():
            name_lower = class_name.lower()
            if 'person' in name_lower or 'people' in name_lower:
                print(f"       'person': [{class_id}],  # {class_name}")
            elif 'helmet' in name_lower and 'no' not in name_lower:
                print(f"       'helmet': [{class_id}],  # {class_name}")
            elif 'vest' in name_lower and 'no' not in name_lower:
                print(f"       'vest': [{class_id}],    # {class_name}")
            elif 'no' in name_lower and 'helmet' in name_lower:
                print(f"       'no_helmet': [{class_id}],  # {class_name}")
        
        print("   }")
        
        return model.names
        
    except ImportError:
        print("❌ Ultralytics not installed. Install with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def generate_config_code(datasets, model_names=None):
    """Generate ready-to-use configuration code"""
    
    if not datasets:
        return
    
    print("\n" + "="*60)
    print("📋 COPY-PASTE READY CONFIGURATION")
    print("="*60)
    
    # Use the first found dataset
    dataset = datasets[0]
    
    print("\n# Paste this into your Quantitative_SAM_Improvement_Analysis.ipynb")
    print("# or your Python script:\n")
    
    print("class Config:")
    print("    # Dataset Paths")
    print(f"    TEST_IMAGES_DIR = r'{dataset['test_images']}'")
    print(f"    GROUND_TRUTH_DIR = r'{dataset['test_labels']}'")
    print()
    print("    # Model Paths")
    print("    YOLO_WEIGHTS = 'best.pt'  # Update with your model path")
    print("    SAM_WEIGHTS = 'sam3.pt'   # Update with SAM weights path")
    print()
    print("    # Detection parameters")
    print("    CONFIDENCE_THRESHOLD = 0.4")
    print("    IOU_THRESHOLD = 0.3")
    print("    SAM_IMAGE_SIZE = 1024")
    print()
    print("    # Class mapping (UPDATE BASED ON YOUR MODEL)")
    
    if model_names:
        print("    TARGET_CLASSES = {")
        for class_id, class_name in model_names.items():
            name_lower = class_name.lower()
            if 'person' in name_lower:
                print(f"        'person': [{class_id}],      # {class_name}")
            elif 'helmet' in name_lower and 'no' not in name_lower:
                print(f"        'helmet': [{class_id}],      # {class_name}")
            elif 'vest' in name_lower and 'no' not in name_lower:
                print(f"        'vest': [{class_id}],        # {class_name}")
            elif 'no' in name_lower and 'helmet' in name_lower:
                print(f"        'no_helmet': [{class_id}],   # {class_name}")
        print("    }")
    else:
        print("    TARGET_CLASSES = {")
        print("        'person': [6],       # Update with your class ID")
        print("        'helmet': [0],       # Update with your class ID")
        print("        'vest': [2],         # Update with your class ID")
        print("        'no_helmet': [7]     # Update with your class ID")
        print("    }")
    
    print("\nconfig = Config()")


if __name__ == "__main__":
    print("🚀 Construction-PPE Dataset Locator")
    print("="*60)
    print()
    
    # Step 1: Find dataset
    datasets = find_construction_ppe_dataset()
    
    # Step 2: Verify model class mapping
    model_names = None
    model_path = 'best.pt'
    
    # Try to find model in common locations
    possible_models = [
        Path('best.pt'),
        Path('runs/detect/train/weights/best.pt'),
        Path('runs/detect/train2/weights/best.pt'),
        Path('exp_adamw/weights/best.pt'),
    ]
    
    for model in possible_models:
        if model.exists():
            model_path = str(model)
            break
    
    model_names = verify_class_mapping(model_path)
    
    # Step 3: Generate configuration
    if datasets:
        generate_config_code(datasets, model_names)
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)

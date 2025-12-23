"""
Quick Download Script for Construction-PPE Dataset
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_construction_ppe():
    """Download and extract Construction-PPE dataset"""
    
    print("📥 Downloading Construction-PPE Dataset...")
    print("="*60)
    
    # Dataset URL
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
    zip_path = "construction-ppe.zip"
    extract_path = "construction-ppe"
    
    # Create directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)
    
    try:
        # Download
        print(f"📦 Downloading from: {url}")
        print("   This may take a few minutes (178.4 MB)...")
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r   Progress: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
        print("\n✅ Download complete!")
        
        # Extract
        print("\n📂 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        print("✅ Extraction complete!")
        
        # Clean up zip file
        os.remove(zip_path)
        print(f"🗑️  Removed {zip_path}")
        
        # Verify structure
        print("\n🔍 Verifying dataset structure...")
        test_images = Path('construction-ppe/images/test')
        test_labels = Path('construction-ppe/labels/test')
        
        if test_images.exists() and test_labels.exists():
            num_images = len(list(test_images.glob('*.jpg')))
            num_labels = len(list(test_labels.glob('*.txt')))
            
            print(f"✅ Dataset ready!")
            print(f"\n📊 Dataset Statistics:")
            print(f"   Root: {Path('construction-ppe').absolute()}")
            print(f"   Test Images: {test_images.absolute()}")
            print(f"   Ground Truth: {test_labels.absolute()}")
            print(f"   Number of test images: {num_images}")
            print(f"   Number of annotations: {num_labels}")
            
            if num_images == num_labels:
                print(f"   ✅ Counts match!")
            else:
                print(f"   ⚠️  Warning: Counts don't match")
            
            return True
        else:
            print("❌ Dataset structure verification failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip")
        return False

if __name__ == "__main__":
    success = download_construction_ppe()
    
    if success:
        print("\n" + "="*60)
        print("🎉 Ready to Run Quantitative Analysis!")
        print("="*60)
        print("\nUse these paths in your configuration:")
        print("   TEST_IMAGES_DIR = r'construction-ppe/images/test'")
        print("   GROUND_TRUTH_DIR = r'construction-ppe/labels/test'")
        print("\nClass Mapping (from your model):")
        print("   TARGET_CLASSES = {")
        print("       'person': [6],")
        print("       'helmet': [0],")
        print("       'vest': [2],")
        print("       'no_helmet': [7]")
        print("   }")

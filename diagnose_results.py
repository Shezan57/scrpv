"""
Diagnostic script to understand why quantitative analysis shows 0 TP
"""
import json
from pathlib import Path
from collections import Counter

# Analyze ground truth
def analyze_ground_truth():
    gt_dir = Path(r'd:\SHEZAN\AI\scrpv\labels\test')
    all_classes = []
    
    for label_file in gt_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    all_classes.append(class_id)
    
    class_counts = Counter(all_classes)
    print("=" * 60)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 60)
    print(f"Total annotations: {len(all_classes)}")
    print(f"\nClass distribution:")
    
    class_names = {
        0: 'helmet', 1: 'gloves', 2: 'vest', 3: 'boots',
        4: 'goggles', 5: 'none', 6: 'Person', 7: 'no_helmet',
        8: 'no_goggle', 9: 'no_gloves'
    }
    
    for class_id in sorted(class_counts.keys()):
        class_name = class_names.get(class_id, f"unknown_{class_id}")
        count = class_counts[class_id]
        percentage = (count / len(all_classes)) * 100
        print(f"  Class {class_id} ({class_name:12s}): {count:4d} ({percentage:5.2f}%)")
    
    return class_counts

# Analyze results
def analyze_results():
    results_file = Path(r'd:\SHEZAN\AI\scrpv\results\quantitative_results.json')
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "=" * 60)
    print("DETECTION RESULTS ANALYSIS")
    print("=" * 60)
    
    yolo = results['yolo_only']
    hybrid = results['hybrid']
    
    print("\nYOLO-Only:")
    print(f"  True Positives (TP):  {yolo['tp']}")
    print(f"  False Positives (FP): {yolo['fp']}")
    print(f"  False Negatives (FN): {yolo['fn']}")
    print(f"  Total Detections: {yolo['tp'] + yolo['fp']}")
    print(f"  Total GT Objects: {yolo['tp'] + yolo['fn']}")
    
    print("\nHybrid (YOLO + SAM):")
    print(f"  True Positives (TP):  {hybrid['tp']}")
    print(f"  False Positives (FP): {hybrid['fp']}")
    print(f"  False Negatives (FN): {hybrid['fn']}")
    print(f"  Total Detections: {hybrid['tp'] + hybrid['fp']}")
    print(f"  Total GT Objects: {hybrid['tp'] + hybrid['fn']}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    if yolo['tp'] == 0 and hybrid['tp'] == 0:
        print("\n❌ PROBLEM: Zero True Positives detected!")
        print("\nPossible causes:")
        print("  1. Class mismatch - Detection classes ≠ Ground truth classes")
        print("  2. IOU threshold too strict (current: 0.3)")
        print("  3. Confidence threshold filtering out detections (current: 0.4)")
        print("  4. Wrong target class mapping in config")
        print("  5. Detection looking for 'violations' but GT has 'PPE items'")
        
        print("\n🔍 Key Insight:")
        print("  Your model detects violations (no_helmet, no_vest)")
        print("  But GT dataset labels PPE presence (helmet, vest)")
        print("  These are OPPOSITE concepts!")
        
    if hybrid['fp'] < yolo['fp']:
        print(f"\n✅ SAM is working! False positives reduced by {yolo['fp'] - hybrid['fp']}")
        print(f"   ({((yolo['fp'] - hybrid['fp']) / yolo['fp'] * 100):.1f}% reduction)")
    
    return results

# Check what classes the analysis is looking for
def check_config():
    print("\n" + "=" * 60)
    print("EXPECTED CONFIGURATION")
    print("=" * 60)
    print("\nTarget classes (what analysis is looking for):")
    print("  person: [6]")
    print("  helmet: [0]")
    print("  vest: [2]")
    print("  no_helmet: [7]")
    
    print("\n⚠️  Problem: Class 7 (no_helmet) likely not in ground truth!")
    print("   Ground truth labels what IS there (helmet), not what's MISSING (no_helmet)")

if __name__ == "__main__":
    gt_classes = analyze_ground_truth()
    results = analyze_results()
    check_config()
    
    print("\n" + "=" * 60)
    print("RECOMMENDED FIX")
    print("=" * 60)
    print("\nOption 1: Change target classes to match ground truth")
    print("  TARGET_CLASSES = {")
    print("      'person': [6],")
    print("      'helmet': [0],")
    print("      'vest': [2],")
    print("      'boots': [3],")
    print("      'gloves': [1],")
    print("      'goggles': [4]")
    print("  }")
    print("\nOption 2: Validate hierarchical detection logic separately")
    print("  - Test if person detection → PPE presence/absence inference works")
    print("  - Compare against ground truth PPE items per person")
    
    print("\nOption 3: Use person+PPE co-occurrence metrics")
    print("  - Detect: Person bounding boxes")
    print("  - Verify: Helmet/vest presence within person ROI")
    print("  - Match against GT: Person boxes with overlapping helmet/vest")

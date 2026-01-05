# ============================================================================
# PPE Safety Detection - Hierarchical YOLO + SAM3 Hybrid System
# Consolidated Colab Notebook with FIXED ROI Implementation
# ============================================================================
# 
# This notebook implements the "Sentry-Judge" hybrid architecture:
#   - YOLOv11m as the Sentry (fast detection)
#   - SAM3 as the Judge (forensic verification on ROI crops)
#
# KEY FIX: SAM3 now receives CROPPED ROI images instead of full images
# ============================================================================

# %% [markdown]
# # 1. Setup and Installation

# %%
# @title 1.1 Install Dependencies
# !pip install -q ultralytics opencv-python-headless matplotlib seaborn pandas

# %%
# @title 1.2 Download SAM3 Weights (if not present)
# !wget -q --header="Authorization: Bearer YOUR_HF_TOKEN" \
#     "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt" \
#     -O /content/sam3.pt

# %%
# @title 1.3 Setup Kaggle API and Download Dataset
"""
# Run these commands manually in Colab:

import json
import os
from google.colab import userdata

# Set up Kaggle credentials
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

kaggle_username = userdata.get('KAGGLE_USERNAME')
kaggle_key = userdata.get('KAGGLE_KEY')

if kaggle_username and kaggle_key:
    with open(kaggle_json_path, "w") as f:
        json.dump({"username": kaggle_username, "key": kaggle_key}, f)
    os.chmod(kaggle_json_path, 0o600)
    print("Kaggle API configured!")

# Download dataset
!kaggle datasets download -d rjn0007/ppeconstruction
!unzip -q ppeconstruction.zip -d ppeconstruction
!rm ppeconstruction.zip
"""

# %% [markdown]
# # 2. Imports and Configuration

# %%
# @title 2.1 Imports
import os
import cv2
import json
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from datetime import datetime
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor

print("✅ All imports successful!")

# %%
# @title 2.2 Configuration
class Config:
    """Configuration for PPE Detection Experiments"""
    
    # Paths - MODIFY THESE FOR YOUR SETUP
    YOLO_WEIGHTS = '/content/best.pt'            # Your trained YOLO model
    SAM_WEIGHTS = '/content/sam3.pt'             # SAM 3 weights
    TEST_IMAGES_DIR = '/content/ppeconstruction/images/test'
    GROUND_TRUTH_DIR = '/content/ppeconstruction/labels/test'
    OUTPUT_DIR = '/content/results'
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.5
    SAM_CONFIDENCE = 0.15
    
    # ROI parameters for SAM (as fraction of person bbox)
    HEAD_ROI_RATIO = 0.4      # Top 40% for head/helmet
    TORSO_START_RATIO = 0.2   # Body starts at 20% from top
    TORSO_END_RATIO = 1.0     # Body ends at bottom
    
    # SAM image size for ROI processing
    # Use smaller size for ROI (faster!) vs full image
    SAM_ROI_SIZE = 640        # For cropped ROI
    SAM_FULL_SIZE = 1024      # For full image (legacy)
    
    # Class definitions
    CLASS_NAMES = {
        0: 'helmet', 1: 'gloves', 2: 'vest', 3: 'boots',
        4: 'goggles', 5: 'none', 6: 'Person', 7: 'no_helmet',
        8: 'no_goggle', 9: 'no_gloves'
    }
    
    # Target classes for hierarchical system
    TARGET_CLASSES = {
        'person': [6],
        'helmet': [0],
        'vest': [2],
        'no_helmet': [7]
    }

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
print("✅ Configuration loaded")
print(f"📊 Target classes: {list(config.TARGET_CLASSES.keys())}")

# %% [markdown]
# # 3. Utility Functions

# %%
# @title 3.1 Annotation Parsing and IoU Calculation
def parse_yolo_annotation(txt_path, img_width, img_height):
    """Parse YOLO format annotation file"""
    annotations = []
    if not os.path.exists(txt_path):
        return annotations
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            annotations.append([class_id, x_min, y_min, x_max, y_max])
    
    return annotations


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def box_overlap_ratio(person_box, item_box):
    """Check if item overlaps with person (for PPE matching)"""
    x1 = max(person_box[0], item_box[0])
    y1 = max(person_box[1], item_box[1])
    x2 = min(person_box[2], item_box[2])
    y2 = min(person_box[3], item_box[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0
    
    item_area = (item_box[2] - item_box[0]) * (item_box[3] - item_box[1])
    return inter_area / item_area if item_area > 0 else 0


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, F1-score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

print("✅ Utility functions loaded")

# %% [markdown]
# # 4. Detector Classes

# %%
# @title 4.1 YOLO-Only Detector (Baseline)
class YOLOOnlyDetector:
    """Baseline detector using only YOLO (no SAM verification)"""
    
    def __init__(self):
        print("🔧 Initializing YOLO-Only Baseline...")
        self.model = YOLO(config.YOLO_WEIGHTS)
        self.inference_times = []
        print("✅ YOLO-Only Baseline Ready")
    
    def detect(self, image_path):
        """Run YOLO detection"""
        start = time.time()
        results = self.model.predict(
            image_path, 
            conf=config.CONFIDENCE_THRESHOLD, 
            verbose=False
        )
        self.inference_times.append(time.time() - start)
        
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy().astype(int)
            detections.append([cls, coords[0], coords[1], coords[2], coords[3], conf])
        return detections
    
    def evaluate_violations(self, image_path):
        """Check for PPE violations using only YOLO"""
        detections = self.detect(image_path)
        
        persons = [d for d in detections if d[0] in config.TARGET_CLASSES['person']]
        helmets = [d for d in detections if d[0] in config.TARGET_CLASSES['helmet']]
        vests = [d for d in detections if d[0] in config.TARGET_CLASSES['vest']]
        no_helmets = [d for d in detections if d[0] in config.TARGET_CLASSES['no_helmet']]
        
        violations = []
        
        for person in persons:
            p_box = person[1:5]
            has_helmet = False
            has_vest = False
            
            # Check helmet overlap
            for helmet in helmets:
                if box_overlap_ratio(p_box, helmet[1:5]) > 0.3:
                    has_helmet = True
                    break
            
            # Check vest overlap
            for vest in vests:
                if box_overlap_ratio(p_box, vest[1:5]) > 0.3:
                    has_vest = True
                    break
            
            # Check explicit no_helmet
            for no_helmet in no_helmets:
                if box_overlap_ratio(p_box, no_helmet[1:5]) > 0.3:
                    has_helmet = False
                    break
            
            is_violation = not has_helmet or not has_vest
            
            violations.append({
                'bbox': p_box,
                'has_helmet': has_helmet,
                'has_vest': has_vest,
                'is_violation': is_violation,
                'decision_path': 'YOLO-Only',
                'confidence': person[5]
            })
        
        return violations
    
    def get_avg_fps(self):
        if len(self.inference_times) == 0:
            return 0
        return 1.0 / np.mean(self.inference_times)

print("✅ YOLOOnlyDetector class defined")

# %%
# @title 4.2 Hybrid Detector (YOLO + SAM) - FIXED ROI IMPLEMENTATION
class HybridDetector:
    """
    Hybrid detector using YOLO + SAM3 with proper ROI cropping
    
    KEY FIX: SAM3 receives CROPPED ROI images, not full images
    This matches the paper's claim of "Geometric Prompt Engineering"
    """
    
    def __init__(self):
        print("🔧 Initializing YOLO + SAM Hybrid System...")
        self.yolo_model = YOLO(config.YOLO_WEIGHTS)
        
        # Initialize SAM3
        overrides = dict(
            model=config.SAM_WEIGHTS, 
            task="segment", 
            mode="predict", 
            conf=config.SAM_CONFIDENCE
        )
        self.sam_model = SAM3SemanticPredictor(overrides=overrides)
        
        # Timing stats
        self.yolo_times = []
        self.sam_times = []
        self.total_times = []
        self.sam_activations = 0
        self.total_persons = 0
        
        print("✅ Hybrid System Ready (with FIXED ROI cropping)")
    
    def run_sam_rescue(self, img_array, search_prompts, roi_box, h, w):
        """
        Runs SAM 3 on CROPPED ROI (not full image) - FIXED VERSION
        
        Args:
            img_array: Full image array (H, W, 3) - numpy RGB array
            search_prompts: List of text prompts, e.g., ["helmet"] or ["vest"]
            roi_box: [x_min, y_min, x_max, y_max] - ROI coordinates
            h, w: Full image dimensions (for bounds checking)
        
        Returns:
            bool: True if object found in ROI, False otherwise
        """
        sam_start = time.time()
        
        try:
            # ========================================
            # 🔧 FIX: Extract ROI BEFORE calling SAM
            # ========================================
            x_min, y_min, x_max, y_max = roi_box
            
            # Clamp to image bounds
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(w, int(x_max))
            y_max = min(h, int(y_max))
            
            # Extract ROI from image
            roi_img = img_array[y_min:y_max, x_min:x_max]
            
            # Validate ROI size
            if roi_img.size == 0 or roi_img.shape[0] < 20 or roi_img.shape[1] < 20:
                return False
            
            # Calculate appropriate SAM size for this ROI
            roi_size = max(roi_img.shape[0], roi_img.shape[1])
            sam_size = min(config.SAM_ROI_SIZE, roi_size)
            sam_size = max(sam_size, 256)  # Minimum size for SAM
            
            # Run SAM on cropped ROI (much faster!)
            res = self.sam_model(
                roi_img,  # 🔧 FIXED: Pass cropped ROI, not full image
                text=search_prompts,
                imgsz=sam_size,
                verbose=False
            )
            
            self.sam_times.append(time.time() - sam_start)
            
            if not res[0].masks:
                return False
            
            # Check if any mask has sufficient coverage
            masks = [m.cpu().numpy().astype(np.uint8) for m in res[0].masks.data]
            roi_h, roi_w = roi_img.shape[:2]
            
            for m in masks:
                # Resize mask to ROI dimensions if needed
                if m.shape[:2] != (roi_h, roi_w):
                    m = cv2.resize(m, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                
                # Check mask coverage (at least 5% of ROI should be covered)
                coverage = np.sum(m) / m.size
                if coverage > 0.05:
                    return True
            
            return False
            
        except Exception as e:
            # Log error for debugging if needed
            # print(f"   ⚠️ SAM Error: {e}")
            return False
    
    def evaluate_violations(self, image_path):
        """
        Check for PPE violations using YOLO + SAM hybrid approach
        with the 5-path decision logic (FIXED ROI version)
        """
        total_start = time.time()
        
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # YOLO Detection
        yolo_start = time.time()
        results = self.yolo_model.predict(
            image_path, 
            conf=config.CONFIDENCE_THRESHOLD, 
            verbose=False
        )
        self.yolo_times.append(time.time() - yolo_start)
        
        # Parse detections
        detections = {'person': [], 'helmet': [], 'vest': [], 'no_helmet': []}
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy().astype(int)
            
            for key, ids in config.TARGET_CLASSES.items():
                if cls in ids:
                    detections[key].append(coords)
        
        violations = []
        
        # Hierarchical Logic for each person
        for p_box in detections['person']:
            self.total_persons += 1
            has_helmet, has_vest, unsafe_explicit = False, False, False
            decision_path = ""
            
            # Check YOLO overlaps
            for eq in detections['helmet']:
                if box_overlap_ratio(p_box, eq) > 0.3:
                    has_helmet = True
            for eq in detections['vest']:
                if box_overlap_ratio(p_box, eq) > 0.3:
                    has_vest = True
            for eq in detections['no_helmet']:
                if box_overlap_ratio(p_box, eq) > 0.3:
                    unsafe_explicit = True
            
            # ========================================
            # 5-PATH DECISION LOGIC
            # ========================================
            
            # Path 1: Fast Violation (explicit no_helmet)
            if unsafe_explicit:
                decision_path = "Fast Violation"
                has_helmet = False
            
            # Path 0: Fast Safe (both PPE detected)
            elif has_helmet and has_vest:
                decision_path = "Fast Safe"
            
            # Path 3: Rescue Body (helmet found, vest missing)
            elif has_helmet and not has_vest:
                decision_path = "Rescue Body"
                self.sam_activations += 1
                
                # Define torso ROI
                body_roi = [
                    p_box[0], 
                    int(p_box[1] + (p_box[3] - p_box[1]) * config.TORSO_START_RATIO),
                    p_box[2], 
                    p_box[3]
                ]
                # 🔧 FIXED: Pass image array, not path
                has_vest = self.run_sam_rescue(img_rgb, ["vest", "safety vest"], body_roi, h, w)
            
            # Path 2: Rescue Head (vest found, helmet missing)
            elif has_vest and not has_helmet:
                decision_path = "Rescue Head"
                self.sam_activations += 1
                
                # Define head ROI
                head_roi = [
                    p_box[0], 
                    p_box[1], 
                    p_box[2], 
                    int(p_box[1] + (p_box[3] - p_box[1]) * config.HEAD_ROI_RATIO)
                ]
                # 🔧 FIXED: Pass image array, not path
                has_helmet = self.run_sam_rescue(img_rgb, ["helmet", "hard hat"], head_roi, h, w)
            
            # Path 4: Critical (both PPE missing or uncertain)
            else:
                decision_path = "Critical"
                self.sam_activations += 2  # Two SAM calls
                
                head_roi = [
                    p_box[0], p_box[1], p_box[2], 
                    int(p_box[1] + (p_box[3] - p_box[1]) * config.HEAD_ROI_RATIO)
                ]
                body_roi = [
                    p_box[0], 
                    int(p_box[1] + (p_box[3] - p_box[1]) * config.TORSO_START_RATIO),
                    p_box[2], 
                    p_box[3]
                ]
                # 🔧 FIXED: Pass image array, not path
                has_helmet = self.run_sam_rescue(img_rgb, ["helmet", "hard hat"], head_roi, h, w)
                has_vest = self.run_sam_rescue(img_rgb, ["vest", "safety vest"], body_roi, h, w)
            
            is_violation = not has_helmet or not has_vest
            
            violations.append({
                'bbox': list(p_box),
                'has_helmet': has_helmet,
                'has_vest': has_vest,
                'is_violation': is_violation,
                'decision_path': decision_path,
                'confidence': 0.85
            })
        
        self.total_times.append(time.time() - total_start)
        return violations
    
    def get_timing_stats(self):
        """Get performance statistics"""
        stats = {
            'avg_yolo_ms': np.mean(self.yolo_times) * 1000 if self.yolo_times else 0,
            'avg_sam_ms': np.mean(self.sam_times) * 1000 if self.sam_times else 0,
            'avg_total_ms': np.mean(self.total_times) * 1000 if self.total_times else 0,
            'avg_fps': 1.0 / np.mean(self.total_times) if self.total_times and np.mean(self.total_times) > 0 else 0,
            'sam_activation_rate': self.sam_activations / self.total_persons if self.total_persons > 0 else 0,
            'total_persons': self.total_persons,
            'total_sam_calls': self.sam_activations
        }
        return stats

print("✅ HybridDetector class defined (with FIXED ROI cropping)")

# %% [markdown]
# # 5. Evaluation Functions

# %%
# @title 5.1 Match Detections to Ground Truth
def match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.5):
    """Match detected violations to ground truth"""
    gt_violations = [gt for gt in ground_truth if gt[0] in config.TARGET_CLASSES['no_helmet']]
    
    matched_gt = set()
    true_positives = 0
    false_positives = 0
    
    for det in detections:
        if not det['is_violation']:
            continue
        
        det_box = det['bbox']
        matched = False
        
        for idx, gt in enumerate(gt_violations):
            if idx in matched_gt:
                continue
            
            gt_box = gt[1:5]
            iou = calculate_iou(det_box, gt_box)
            
            if iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(idx)
                matched = True
                break
        
        if not matched:
            false_positives += 1
    
    false_negatives = len(gt_violations) - len(matched_gt)
    
    return true_positives, false_positives, false_negatives

print("✅ Evaluation functions loaded")

# %% [markdown]
# # 6. Run Experiments

# %%
# @title 6.1 Run Full Evaluation (YOLO-Only vs Hybrid)
def run_full_evaluation():
    """Run complete quantitative evaluation comparing YOLO-only vs Hybrid"""
    print("=" * 80)
    print("🔬 RUNNING QUANTITATIVE EVALUATION")
    print("   Comparing YOLO-Only (Baseline) vs YOLO+SAM (Hybrid with Fixed ROI)")
    print("=" * 80)
    
    # Initialize detectors
    yolo_detector = YOLOOnlyDetector()
    hybrid_detector = HybridDetector()
    
    # Get test images
    test_images = glob.glob(f"{config.TEST_IMAGES_DIR}/*.jpg") + \
                  glob.glob(f"{config.TEST_IMAGES_DIR}/*.png") + \
                  glob.glob(f"{config.TEST_IMAGES_DIR}/*.webp")
    
    print(f"\n📸 Found {len(test_images)} test images\n")
    
    # Results accumulators
    yolo_results = {'tp': 0, 'fp': 0, 'fn': 0}
    hybrid_results = {'tp': 0, 'fp': 0, 'fn': 0}
    decision_paths = []
    detailed_results = []
    
    for idx, img_path in enumerate(test_images):
        if idx % 20 == 0:
            print(f"   Processing image {idx+1}/{len(test_images)}...")
        
        img_name = os.path.basename(img_path)
        gt_path = os.path.join(
            config.GROUND_TRUTH_DIR,
            img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.webp', '.txt')
        )
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        ground_truth = parse_yolo_annotation(gt_path, w, h)
        
        if len(ground_truth) == 0:
            continue
        
        try:
            # YOLO-only evaluation
            yolo_detections = yolo_detector.evaluate_violations(img_path)
            yolo_tp, yolo_fp, yolo_fn = match_detections_to_ground_truth(yolo_detections, ground_truth)
            yolo_results['tp'] += yolo_tp
            yolo_results['fp'] += yolo_fp
            yolo_results['fn'] += yolo_fn
            
            # Hybrid evaluation
            hybrid_detections = hybrid_detector.evaluate_violations(img_path)
            hybrid_tp, hybrid_fp, hybrid_fn = match_detections_to_ground_truth(hybrid_detections, ground_truth)
            hybrid_results['tp'] += hybrid_tp
            hybrid_results['fp'] += hybrid_fp
            hybrid_results['fn'] += hybrid_fn
            
            # Collect decision paths
            for det in hybrid_detections:
                if 'decision_path' in det:
                    decision_paths.append(det['decision_path'])
            
            detailed_results.append({
                'image': img_name,
                'yolo_tp': yolo_tp, 'yolo_fp': yolo_fp, 'yolo_fn': yolo_fn,
                'hybrid_tp': hybrid_tp, 'hybrid_fp': hybrid_fp, 'hybrid_fn': hybrid_fn
            })
            
        except Exception as e:
            print(f"   ⚠️ Error on {img_name}: {e}")
            continue
    
    # Get timing stats
    yolo_fps = yolo_detector.get_avg_fps()
    hybrid_stats = hybrid_detector.get_timing_stats()
    
    return {
        'yolo_results': yolo_results,
        'hybrid_results': hybrid_results,
        'decision_paths': decision_paths,
        'detailed_results': detailed_results,
        'yolo_fps': yolo_fps,
        'hybrid_stats': hybrid_stats
    }

# %%
# @title 6.2 Run Evaluation and Display Results
# Uncomment to run:
# eval_results = run_full_evaluation()

# %% [markdown]
# # 7. Results Analysis and Visualization

# %%
# @title 7.1 Display Results Summary
def display_results(eval_results):
    """Display comprehensive results summary"""
    
    yolo_results = eval_results['yolo_results']
    hybrid_results = eval_results['hybrid_results']
    decision_paths = eval_results['decision_paths']
    yolo_fps = eval_results['yolo_fps']
    hybrid_stats = eval_results['hybrid_stats']
    
    # Calculate metrics
    yolo_prec, yolo_rec, yolo_f1 = calculate_metrics(
        yolo_results['tp'], yolo_results['fp'], yolo_results['fn']
    )
    hybrid_prec, hybrid_rec, hybrid_f1 = calculate_metrics(
        hybrid_results['tp'], hybrid_results['fp'], hybrid_results['fn']
    )
    
    # Improvements
    prec_imp = ((hybrid_prec - yolo_prec) / yolo_prec * 100) if yolo_prec > 0 else 0
    rec_imp = ((hybrid_rec - yolo_rec) / yolo_rec * 100) if yolo_rec > 0 else 0
    f1_imp = ((hybrid_f1 - yolo_f1) / yolo_f1 * 100) if yolo_f1 > 0 else 0
    fp_red = ((yolo_results['fp'] - hybrid_results['fp']) / yolo_results['fp'] * 100) if yolo_results['fp'] > 0 else 0
    fn_red = ((yolo_results['fn'] - hybrid_results['fn']) / yolo_results['fn'] * 100) if yolo_results['fn'] > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n1️⃣  YOLO-Only Baseline:")
    print(f"   Precision: {yolo_prec:.4f}")
    print(f"   Recall:    {yolo_rec:.4f}")
    print(f"   F1-Score:  {yolo_f1:.4f}")
    print(f"   TP: {yolo_results['tp']}, FP: {yolo_results['fp']}, FN: {yolo_results['fn']}")
    print(f"   FPS: {yolo_fps:.1f}")
    
    print("\n2️⃣  YOLO + SAM Hybrid (FIXED ROI):")
    print(f"   Precision: {hybrid_prec:.4f} ({prec_imp:+.2f}%)")
    print(f"   Recall:    {hybrid_rec:.4f} ({rec_imp:+.2f}%)")
    print(f"   F1-Score:  {hybrid_f1:.4f} ({f1_imp:+.2f}%)")
    print(f"   TP: {hybrid_results['tp']}, FP: {hybrid_results['fp']}, FN: {hybrid_results['fn']}")
    print(f"   FPS: {hybrid_stats['avg_fps']:.1f}")
    
    print("\n3️⃣  Performance Improvements:")
    print(f"   False Positive Reduction: {fp_red:.2f}%")
    print(f"   False Negative Reduction: {fn_red:.2f}%")
    print(f"   Precision Improvement:    {prec_imp:+.2f}%")
    print(f"   Recall Improvement:       {rec_imp:+.2f}%")
    print(f"   F1-Score Improvement:     {f1_imp:+.2f}%")
    
    print("\n4️⃣  Latency Analysis (FIXED ROI):")
    print(f"   Avg YOLO Time:  {hybrid_stats['avg_yolo_ms']:.1f} ms")
    print(f"   Avg SAM Time:   {hybrid_stats['avg_sam_ms']:.1f} ms (on ROI crops)")
    print(f"   Avg Total Time: {hybrid_stats['avg_total_ms']:.1f} ms")
    print(f"   Effective FPS:  {hybrid_stats['avg_fps']:.1f}")
    
    if decision_paths:
        path_counts = Counter(decision_paths)
        total = len(decision_paths)
        
        print("\n5️⃣  Decision Path Distribution:")
        for path in ['Fast Safe', 'Fast Violation', 'Rescue Head', 'Rescue Body', 'Critical']:
            count = path_counts.get(path, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"   {path:20s}: {count:4d} ({pct:5.1f}%)")
        
        sam_paths = ['Rescue Head', 'Rescue Body', 'Critical']
        sam_count = sum([path_counts.get(p, 0) for p in sam_paths])
        sam_rate = (sam_count / total * 100) if total > 0 else 0
        bypass_count = path_counts.get('Fast Safe', 0) + path_counts.get('Fast Violation', 0)
        bypass_rate = (bypass_count / total * 100) if total > 0 else 0
        
        print(f"\n   SAM Activation Rate:  {sam_rate:.1f}%")
        print(f"   SAM Bypass Rate:      {bypass_rate:.1f}%")
    
    print("=" * 80)
    
    return {
        'yolo_metrics': {'precision': yolo_prec, 'recall': yolo_rec, 'f1': yolo_f1},
        'hybrid_metrics': {'precision': hybrid_prec, 'recall': hybrid_rec, 'f1': hybrid_f1},
        'improvements': {'precision': prec_imp, 'recall': rec_imp, 'f1': f1_imp, 'fp_reduction': fp_red, 'fn_reduction': fn_red}
    }

# %%
# @title 7.2 Generate Visualizations
def generate_visualizations(eval_results, output_dir):
    """Generate publication-ready visualizations"""
    
    yolo_results = eval_results['yolo_results']
    hybrid_results = eval_results['hybrid_results']
    decision_paths = eval_results['decision_paths']
    
    # Calculate metrics
    yolo_prec, yolo_rec, yolo_f1 = calculate_metrics(
        yolo_results['tp'], yolo_results['fp'], yolo_results['fn']
    )
    hybrid_prec, hybrid_rec, hybrid_f1 = calculate_metrics(
        hybrid_results['tp'], hybrid_results['fp'], hybrid_results['fn']
    )
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Metrics Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    yolo_vals = [yolo_prec, yolo_rec, yolo_f1]
    hybrid_vals = [hybrid_prec, hybrid_rec, hybrid_f1]
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, yolo_vals, width, label='YOLO Only', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, hybrid_vals, width, label='YOLO + SAM (Fixed ROI)', color='#4ECDC4', alpha=0.8)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Error Reduction
    ax2 = fig.add_subplot(gs[0, 1])
    error_types = ['False Positives', 'False Negatives']
    yolo_errors = [yolo_results['fp'], yolo_results['fn']]
    hybrid_errors = [hybrid_results['fp'], hybrid_results['fn']]
    
    x = np.arange(len(error_types))
    bars1 = ax2.bar(x - width/2, yolo_errors, width, label='YOLO Only', color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, hybrid_errors, width, label='YOLO + SAM', color='#4ECDC4', alpha=0.8)
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Error Reduction', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(error_types)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. FPS Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    hybrid_stats = eval_results['hybrid_stats']
    fps_values = [eval_results['yolo_fps'], hybrid_stats['avg_fps']]
    fps_labels = ['YOLO Only', 'Hybrid (Fixed ROI)']
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax3.bar(fps_labels, fps_values, color=colors, alpha=0.8)
    ax3.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax3.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax3.axhline(y=24, color='green', linestyle='--', label='Real-time threshold (24 FPS)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, fps_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Decision Path Distribution
    if decision_paths:
        ax4 = fig.add_subplot(gs[1, :2])
        path_counts = Counter(decision_paths)
        paths = ['Fast Safe', 'Fast Violation', 'Rescue Head', 'Rescue Body', 'Critical']
        counts = [path_counts.get(p, 0) for p in paths]
        total = sum(counts)
        percentages = [c/total*100 if total > 0 else 0 for c in counts]
        
        colors_map = ['#2ECC71', '#E74C3C', '#F39C12', '#E67E22', '#8E44AD']
        bars = ax4.bar(paths, percentages, color=colors_map, alpha=0.8)
        
        for bar, pct, cnt in zip(bars, percentages, counts):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=10)
        
        ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Decision Path', fontsize=12, fontweight='bold')
        ax4.set_title('Decision Path Distribution (5-Path Logic)', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15)
    
    # 5. SAM Activation Analysis
    ax5 = fig.add_subplot(gs[1, 2])
    sam_paths = ['Rescue Head', 'Rescue Body', 'Critical']
    sam_count = sum([path_counts.get(p, 0) for p in sam_paths])
    bypass_count = path_counts.get('Fast Safe', 0) + path_counts.get('Fast Violation', 0)
    
    sizes = [bypass_count, sam_count]
    labels = [f'SAM Bypassed\n({bypass_count})', f'SAM Triggered\n({sam_count})']
    colors = ['#2ECC71', '#E67E22']
    explode = (0.05, 0)
    
    ax5.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax5.set_title('SAM Activation Rate', fontsize=14, fontweight='bold')
    
    plt.suptitle('PPE Detection: YOLO vs Hybrid (Fixed ROI) Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    output_path = os.path.join(output_dir, 'comparison_results_fixed_roi.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization saved: {output_path}")

# %%
# @title 7.3 Save Results to Files
def save_results(eval_results, metrics_summary, output_dir):
    """Save all results to files"""
    
    # Save JSON summary
    json_output = {
        'yolo_results': eval_results['yolo_results'],
        'hybrid_results': eval_results['hybrid_results'],
        'yolo_fps': eval_results['yolo_fps'],
        'hybrid_stats': eval_results['hybrid_stats'],
        'metrics_summary': metrics_summary,
        'decision_path_distribution': dict(Counter(eval_results['decision_paths']))
    }
    
    json_path = os.path.join(output_dir, 'evaluation_results_fixed_roi.json')
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=4)
    print(f"💾 Results saved: {json_path}")
    
    # Save detailed CSV
    df = pd.DataFrame(eval_results['detailed_results'])
    csv_path = os.path.join(output_dir, 'detailed_results_per_image.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 Detailed results saved: {csv_path}")
    
    # Save SAM activation analysis
    sam_path = os.path.join(output_dir, 'sam_activation_analysis.txt')
    with open(sam_path, 'w') as f:
        path_counts = Counter(eval_results['decision_paths'])
        total = len(eval_results['decision_paths'])
        
        f.write("SAM Activation Analysis (Fixed ROI Implementation)\n")
        f.write("=" * 50 + "\n\n")
        
        for path in ['Fast Safe', 'Fast Violation', 'Rescue Head', 'Rescue Body', 'Critical']:
            count = path_counts.get(path, 0)
            pct = (count / total * 100) if total > 0 else 0
            f.write(f"{path:20s}: {count:4d} ({pct:5.1f}%)\n")
        
        sam_paths = ['Rescue Head', 'Rescue Body', 'Critical']
        sam_count = sum([path_counts.get(p, 0) for p in sam_paths])
        bypass_count = path_counts.get('Fast Safe', 0) + path_counts.get('Fast Violation', 0)
        
        f.write(f"\nSAM Triggered: {sam_count} ({sam_count/total*100:.1f}%)\n")
        f.write(f"SAM Bypassed:  {bypass_count} ({bypass_count/total*100:.1f}%)\n")
    
    print(f"💾 SAM analysis saved: {sam_path}")

# %% [markdown]
# # 8. Main Execution

# %%
# @title 8.1 Run Complete Experiment
def main():
    """Run the complete experiment"""
    print("=" * 80)
    print("🚀 PPE SAFETY DETECTION - HYBRID YOLO + SAM3 EVALUATION")
    print("   With FIXED ROI Implementation (matching paper claims)")
    print("=" * 80)
    
    # Run evaluation
    print("\n📊 Starting evaluation...")
    eval_results = run_full_evaluation()
    
    # Display results
    print("\n📈 Analyzing results...")
    metrics_summary = display_results(eval_results)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    generate_visualizations(eval_results, config.OUTPUT_DIR)
    
    # Save results
    print("\n💾 Saving results...")
    save_results(eval_results, metrics_summary, config.OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("✅ EXPERIMENT COMPLETE!")
    print(f"📁 All results saved to: {config.OUTPUT_DIR}")
    print("=" * 80)
    
    return eval_results, metrics_summary

# Uncomment to run:
# eval_results, metrics = main()

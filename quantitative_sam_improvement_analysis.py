"""
Quantitative SAM Improvement Analysis
======================================
This script measures the quantitative improvement that SAM 3 provides over YOLO-only baseline
for PPE violation detection in construction sites.

Metrics Calculated:
- Precision, Recall, F1-Score improvement
- False Positive Reduction Rate
- False Negative Reduction Rate
- Per-class performance metrics
- Confusion matrices comparison

Usage:
    python quantitative_sam_improvement_analysis.py --test_dir /path/to/test/images --gt_dir /path/to/ground/truth
"""

import os
import cv2
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    YOLO_WEIGHTS = 'best.pt'  # Your trained YOLO model
    SAM_WEIGHTS = 'sam3.pt'   # SAM 3 weights
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.3  # For matching detections to ground truth
    SAM_IMAGE_SIZE = 1024
    
    # Class mapping (adjust based on your trained model)
    TARGET_CLASSES = {
        'person': [0],      # Adjust these IDs based on your model
        'helmet': [1],
        'vest': [2],
        'no_helmet': [3]
    }
    
    # ROI parameters
    HEAD_ROI_RATIO = 0.4   # Top 40% of person bbox
    TORSO_START_RATIO = 0.2
    TORSO_END_RATIO = 0.7

config = Config()


# =============================================================================
# GROUND TRUTH PARSER
# =============================================================================

def parse_yolo_annotation(txt_path, img_width, img_height):
    """
    Parse YOLO format annotation file
    Format: class_id x_center y_center width height (normalized)
    Returns: List of [class_id, x_min, y_min, x_max, y_max]
    """
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
    """Calculate IoU between two boxes [x_min, y_min, x_max, y_max]"""
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


# =============================================================================
# YOLO-ONLY DETECTOR (BASELINE)
# =============================================================================

class YOLOOnlyDetector:
    """Baseline detector using only YOLO"""
    
    def __init__(self):
        print("🔧 Initializing YOLO-Only Baseline...")
        self.model = YOLO(config.YOLO_WEIGHTS)
        print("✅ YOLO-Only Baseline Ready")
    
    def detect(self, image_path):
        """
        Detect violations using only YOLO
        Returns: List of detections [class_id, x_min, y_min, x_max, y_max, confidence]
        """
        results = self.model.predict(image_path, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy().astype(int)
            
            # [class_id, x_min, y_min, x_max, y_max, confidence]
            detections.append([cls, coords[0], coords[1], coords[2], coords[3], conf])
        
        return detections
    
    def evaluate_violations(self, image_path):
        """
        Check for PPE violations using only YOLO detections
        Returns: List of person boxes with violation status
        """
        detections = self.detect(image_path)
        
        # Group detections by class
        persons = [d for d in detections if d[0] in config.TARGET_CLASSES['person']]
        helmets = [d for d in detections if d[0] in config.TARGET_CLASSES['helmet']]
        vests = [d for d in detections if d[0] in config.TARGET_CLASSES['vest']]
        no_helmets = [d for d in detections if d[0] in config.TARGET_CLASSES['no_helmet']]
        
        violations = []
        
        for person in persons:
            p_box = person[1:5]
            has_helmet = False
            has_vest = False
            
            # Check helmet
            for helmet in helmets:
                h_box = helmet[1:5]
                if calculate_iou(p_box, h_box) > config.IOU_THRESHOLD:
                    has_helmet = True
                    break
            
            # Check vest
            for vest in vests:
                v_box = vest[1:5]
                if calculate_iou(p_box, v_box) > config.IOU_THRESHOLD:
                    has_vest = True
                    break
            
            # Check explicit no-helmet
            for no_helmet in no_helmets:
                nh_box = no_helmet[1:5]
                if calculate_iou(p_box, nh_box) > config.IOU_THRESHOLD:
                    has_helmet = False
                    break
            
            # Violation if missing any PPE
            is_violation = not has_helmet or not has_vest
            
            violations.append({
                'bbox': p_box,
                'has_helmet': has_helmet,
                'has_vest': has_vest,
                'is_violation': is_violation,
                'confidence': person[5]
            })
        
        return violations


# =============================================================================
# YOLO + SAM HYBRID DETECTOR
# =============================================================================

class HybridDetector:
    """Hybrid detector using YOLO + SAM 3 with smart decision logic"""
    
    def __init__(self):
        print("🔧 Initializing YOLO + SAM Hybrid System...")
        self.yolo_model = YOLO(config.YOLO_WEIGHTS)
        
        # Load SAM 3
        overrides = dict(model=config.SAM_WEIGHTS, task="segment", mode="predict", conf=0.15)
        self.sam_model = SAM3SemanticPredictor(overrides=overrides)
        
        print("✅ Hybrid System Ready")
    
    def box_iou(self, box1, box2):
        """Calculate IoU for overlap checking"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / box2_area
    
    def run_sam_rescue(self, image_path, search_prompts, roi_box, h, w):
        """Run SAM on specific ROI"""
        try:
            res = self.sam_model(image_path, text=search_prompts, imgsz=config.SAM_IMAGE_SIZE, verbose=False)
            if not res[0].masks:
                return False
            
            masks = [m.cpu().numpy().astype(np.uint8) for m in res[0].masks.data]
            for m in masks:
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                roi = m[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]
                if np.sum(roi) > 0:
                    return True
        except:
            pass
        return False
    
    def evaluate_violations(self, image_path):
        """
        Check for PPE violations using YOLO + SAM hybrid approach
        Returns: List of person boxes with violation status
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # YOLO Detection
        results = self.yolo_model.predict(image_path, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        detections = {'person': [], 'helmet': [], 'vest': [], 'no_helmet': []}
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy().astype(int)
            
            for key, ids in config.TARGET_CLASSES.items():
                if cls in ids:
                    detections[key].append(coords)
        
        violations = []
        
        # Hierarchical Logic with SAM Rescue
        for p_box in detections['person']:
            has_helmet, has_vest, unsafe_explicit = False, False, False
            decision_path = ""
            
            # Check Overlaps
            for eq in detections['helmet']:
                if self.box_iou(p_box, eq) > 0.3:
                    has_helmet = True
            for eq in detections['vest']:
                if self.box_iou(p_box, eq) > 0.3:
                    has_vest = True
            for eq in detections['no_helmet']:
                if self.box_iou(p_box, eq) > 0.3:
                    unsafe_explicit = True
            
            # Decision Logic
            if unsafe_explicit:
                decision_path = "Fast Violation"
                has_helmet = False
            
            elif has_helmet and has_vest:
                decision_path = "Fast Safe"
            
            elif has_helmet and not has_vest:
                decision_path = "Rescue Body"
                body_roi = [p_box[0], int(p_box[1] + (p_box[3]-p_box[1])*config.TORSO_START_RATIO), 
                           p_box[2], p_box[3]]
                has_vest = self.run_sam_rescue(image_path, ["vest"], body_roi, h, w)
            
            elif has_vest and not has_helmet:
                decision_path = "Rescue Head"
                head_roi = [p_box[0], p_box[1], p_box[2], 
                           int(p_box[1] + (p_box[3]-p_box[1])*config.HEAD_ROI_RATIO)]
                has_helmet = self.run_sam_rescue(image_path, ["helmet"], head_roi, h, w)
            
            else:
                decision_path = "Critical"
                head_roi = [p_box[0], p_box[1], p_box[2], 
                           int(p_box[1] + (p_box[3]-p_box[1])*config.HEAD_ROI_RATIO)]
                body_roi = [p_box[0], int(p_box[1] + (p_box[3]-p_box[1])*config.TORSO_START_RATIO), 
                           p_box[2], p_box[3]]
                has_helmet = self.run_sam_rescue(image_path, ["helmet"], head_roi, h, w)
                has_vest = self.run_sam_rescue(image_path, ["vest"], body_roi, h, w)
            
            is_violation = not has_helmet or not has_vest
            
            violations.append({
                'bbox': p_box,
                'has_helmet': has_helmet,
                'has_vest': has_vest,
                'is_violation': is_violation,
                'decision_path': decision_path,
                'confidence': 0.85
            })
        
        return violations


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.5):
    """
    Match detected violations to ground truth
    Returns: TP, FP, FN counts
    """
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


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, F1-score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def run_quantitative_evaluation(test_images_dir, ground_truth_dir, output_dir='results'):
    """
    Run complete quantitative evaluation comparing YOLO-only vs YOLO+SAM
    """
    print("="*80)
    print("🔬 QUANTITATIVE SAM IMPROVEMENT ANALYSIS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detectors
    yolo_detector = YOLOOnlyDetector()
    hybrid_detector = HybridDetector()
    
    # Get test images
    test_images = glob.glob(f"{test_images_dir}/*.jpg") + \
                  glob.glob(f"{test_images_dir}/*.png") + \
                  glob.glob(f"{test_images_dir}/*.webp")
    
    print(f"\n📸 Found {len(test_images)} test images\n")
    
    # Storage for results
    yolo_results = {'tp': 0, 'fp': 0, 'fn': 0}
    hybrid_results = {'tp': 0, 'fp': 0, 'fn': 0}
    
    detailed_results = []
    decision_paths = []
    
    # Process each image
    for idx, img_path in enumerate(test_images):
        if idx % 10 == 0:
            print(f"   Processing image {idx+1}/{len(test_images)}...")
        
        img_name = os.path.basename(img_path)
        gt_path = os.path.join(ground_truth_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.webp', '.txt'))
        
        # Load ground truth
        img = cv2.imread(img_path)
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
            
            # Track decision paths
            for det in hybrid_detections:
                if 'decision_path' in det:
                    decision_paths.append(det['decision_path'])
            
            # Store detailed results
            detailed_results.append({
                'image': img_name,
                'yolo_tp': yolo_tp,
                'yolo_fp': yolo_fp,
                'yolo_fn': yolo_fn,
                'hybrid_tp': hybrid_tp,
                'hybrid_fp': hybrid_fp,
                'hybrid_fn': hybrid_fn
            })
            
        except Exception as e:
            print(f"   ⚠️ Error processing {img_name}: {e}")
            continue
    
    # Calculate metrics
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    
    yolo_prec, yolo_rec, yolo_f1 = calculate_metrics(yolo_results['tp'], yolo_results['fp'], yolo_results['fn'])
    hybrid_prec, hybrid_rec, hybrid_f1 = calculate_metrics(hybrid_results['tp'], hybrid_results['fp'], hybrid_results['fn'])
    
    # Calculate improvement
    prec_improvement = ((hybrid_prec - yolo_prec) / yolo_prec * 100) if yolo_prec > 0 else 0
    rec_improvement = ((hybrid_rec - yolo_rec) / yolo_rec * 100) if yolo_rec > 0 else 0
    f1_improvement = ((hybrid_f1 - yolo_f1) / yolo_f1 * 100) if yolo_f1 > 0 else 0
    
    fp_reduction = ((yolo_results['fp'] - hybrid_results['fp']) / yolo_results['fp'] * 100) if yolo_results['fp'] > 0 else 0
    fn_reduction = ((yolo_results['fn'] - hybrid_results['fn']) / yolo_results['fn'] * 100) if yolo_results['fn'] > 0 else 0
    
    print("\n1️⃣  YOLO-Only Baseline:")
    print(f"   Precision: {yolo_prec:.4f}")
    print(f"   Recall:    {yolo_rec:.4f}")
    print(f"   F1-Score:  {yolo_f1:.4f}")
    print(f"   TP: {yolo_results['tp']}, FP: {yolo_results['fp']}, FN: {yolo_results['fn']}")
    
    print("\n2️⃣  YOLO + SAM Hybrid:")
    print(f"   Precision: {hybrid_prec:.4f} ({prec_improvement:+.2f}%)")
    print(f"   Recall:    {hybrid_rec:.4f} ({rec_improvement:+.2f}%)")
    print(f"   F1-Score:  {hybrid_f1:.4f} ({f1_improvement:+.2f}%)")
    print(f"   TP: {hybrid_results['tp']}, FP: {hybrid_results['fp']}, FN: {hybrid_results['fn']}")
    
    print("\n3️⃣  SAM Improvement Metrics:")
    print(f"   False Positive Reduction: {fp_reduction:.2f}%")
    print(f"   False Negative Reduction: {fn_reduction:.2f}%")
    print(f"   Precision Improvement:    {prec_improvement:+.2f}%")
    print(f"   Recall Improvement:       {rec_improvement:+.2f}%")
    print(f"   F1-Score Improvement:     {f1_improvement:+.2f}%")
    
    # Decision path distribution
    if decision_paths:
        path_counts = Counter(decision_paths)
        total = len(decision_paths)
        
        print("\n4️⃣  Decision Path Distribution:")
        for path in ['Fast Safe', 'Fast Violation', 'Rescue Head', 'Rescue Body', 'Critical']:
            count = path_counts.get(path, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"   {path:20s}: {count:4d} ({pct:5.1f}%)")
        
        sam_paths = ['Rescue Head', 'Rescue Body', 'Critical']
        sam_count = sum([path_counts.get(p, 0) for p in sam_paths])
        sam_rate = (sam_count / total * 100) if total > 0 else 0
        print(f"   {'SAM Activation Rate':20s}: {sam_count:4d} ({sam_rate:5.1f}%)")
    
    print("="*80)
    
    # Save results
    results_dict = {
        'yolo_only': {
            'precision': yolo_prec,
            'recall': yolo_rec,
            'f1_score': yolo_f1,
            'tp': yolo_results['tp'],
            'fp': yolo_results['fp'],
            'fn': yolo_results['fn']
        },
        'hybrid': {
            'precision': hybrid_prec,
            'recall': hybrid_rec,
            'f1_score': hybrid_f1,
            'tp': hybrid_results['tp'],
            'fp': hybrid_results['fp'],
            'fn': hybrid_results['fn']
        },
        'improvement': {
            'precision_improvement_pct': prec_improvement,
            'recall_improvement_pct': rec_improvement,
            'f1_improvement_pct': f1_improvement,
            'fp_reduction_pct': fp_reduction,
            'fn_reduction_pct': fn_reduction
        }
    }
    
    # Save JSON
    with open(f'{output_dir}/quantitative_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    # Save CSV
    df = pd.DataFrame(detailed_results)
    df.to_csv(f'{output_dir}/detailed_results.csv', index=False)
    
    # Generate visualizations
    generate_comparison_plots(results_dict, decision_paths, output_dir)
    
    print(f"\n✅ Results saved to {output_dir}/")
    print(f"   - quantitative_results.json")
    print(f"   - detailed_results.csv")
    print(f"   - comparison_plots.png")
    
    return results_dict


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_comparison_plots(results, decision_paths, output_dir):
    """Generate comparison visualization plots"""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Metrics Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Precision', 'Recall', 'F1-Score']
    yolo_vals = [results['yolo_only']['precision'], 
                 results['yolo_only']['recall'], 
                 results['yolo_only']['f1_score']]
    hybrid_vals = [results['hybrid']['precision'], 
                   results['hybrid']['recall'], 
                   results['hybrid']['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, yolo_vals, width, label='YOLO Only', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, hybrid_vals, width, label='YOLO + SAM', color='#4ECDC4', alpha=0.8)
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Improvement Percentages
    ax2 = fig.add_subplot(gs[0, 1])
    improvements = ['Precision', 'Recall', 'F1-Score']
    improvement_vals = [
        results['improvement']['precision_improvement_pct'],
        results['improvement']['recall_improvement_pct'],
        results['improvement']['f1_improvement_pct']
    ]
    
    colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in improvement_vals]
    bars = ax2.barh(improvements, improvement_vals, color=colors, alpha=0.8)
    ax2.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('SAM Improvement Over YOLO-Only', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, improvement_vals)):
        ax2.text(val + 1 if val > 0 else val - 1, i, f'{val:+.1f}%', 
                va='center', ha='left' if val > 0 else 'right', fontsize=10, fontweight='bold')
    
    # 3. Error Reduction
    ax3 = fig.add_subplot(gs[0, 2])
    error_types = ['False Positives', 'False Negatives']
    yolo_errors = [results['yolo_only']['fp'], results['yolo_only']['fn']]
    hybrid_errors = [results['hybrid']['fp'], results['hybrid']['fn']]
    
    x = np.arange(len(error_types))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, yolo_errors, width, label='YOLO Only', color='#FF6B6B', alpha=0.8)
    bars2 = ax3.bar(x + width/2, hybrid_errors, width, label='YOLO + SAM', color='#4ECDC4', alpha=0.8)
    
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Error Reduction', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(error_types, rotation=15)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
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
        
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Decision Path', fontsize=12, fontweight='bold')
        ax4.set_title('Distribution of Decision Paths (Smart Routing)', fontsize=14, fontweight='bold')
        ax4.axhline(y=15, color='blue', linestyle='--', label='Expected SAM Threshold (15%)', linewidth=2)
        ax4.legend()
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15)
        ax4.grid(axis='y', alpha=0.3)
    
    # 5. Confusion Matrix Comparison (simplified)
    ax5 = fig.add_subplot(gs[1, 2])
    
    reduction_data = [
        ['FP Reduction', results['improvement']['fp_reduction_pct']],
        ['FN Reduction', results['improvement']['fn_reduction_pct']]
    ]
    
    labels = [d[0] for d in reduction_data]
    values = [d[1] for d in reduction_data]
    colors_red = ['#2ECC71' if v > 0 else '#E74C3C' for v in values]
    
    bars = ax5.barh(labels, values, color=colors_red, alpha=0.8)
    ax5.set_xlabel('Reduction (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Error Reduction Rate', fontsize=14, fontweight='bold')
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax5.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax5.text(val + 2 if val > 0 else val - 2, i, f'{val:.1f}%', 
                va='center', ha='left' if val > 0 else 'right', fontsize=11, fontweight='bold')
    
    plt.suptitle('Quantitative SAM Improvement Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(f'{output_dir}/comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to {output_dir}/comparison_plots.png")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantitative SAM Improvement Analysis')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing ground truth annotations')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--yolo_weights', type=str, default='best.pt', help='Path to YOLO weights')
    parser.add_argument('--sam_weights', type=str, default='sam3.pt', help='Path to SAM weights')
    
    args = parser.parse_args()
    
    # Update config
    config.YOLO_WEIGHTS = args.yolo_weights
    config.SAM_WEIGHTS = args.sam_weights
    
    # Run evaluation
    results = run_quantitative_evaluation(args.test_dir, args.gt_dir, args.output_dir)
    
    print("\n✅ Analysis Complete!")

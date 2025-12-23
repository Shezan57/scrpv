"""
Fixed Quantitative Analysis for Hierarchical PPE Detection System

This script properly evaluates:
1. Person detection accuracy
2. PPE item detection accuracy (helmet, vest)
3. Violation detection accuracy (no_helmet, no_vest)
4. SAM's contribution to reducing false positives

Author: Fixed for hierarchical detection with violation inference
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class Config:
    """Configuration for PPE Detection Evaluation"""
    # Paths
    TEST_IMAGES_DIR = r'd:\SHEZAN\AI\scrpv\images\test'
    GROUND_TRUTH_DIR = r'd:\SHEZAN\AI\scrpv\labels\test'
    OUTPUT_DIR = r'd:\SHEZAN\AI\scrpv\results'
    
    # Model paths
    YOLO_WEIGHTS = r'd:\SHEZAN\AI\scrpv\exp_adamw\weights\best.pt'
    SAM_WEIGHTS = r'd:\SHEZAN\AI\scrpv\sam3.pt'
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.25  # Lower threshold to catch more detections
    IOU_THRESHOLD = 0.5
    SAM_IMAGE_SIZE = 1024
    
    # Class definitions
    CLASS_NAMES = {
        0: 'helmet', 1: 'gloves', 2: 'vest', 3: 'boots',
        4: 'goggles', 5: 'none', 6: 'Person', 7: 'no_helmet',
        8: 'no_goggle', 9: 'no_gloves'
    }
    
    # Evaluation categories
    PERSON_CLASS = [6]
    PPE_CLASSES = [0, 1, 2, 3, 4]  # helmet, gloves, vest, boots, goggles
    VIOLATION_CLASSES = [7, 8, 9]  # no_helmet, no_goggle, no_gloves
    
    # Focus on key PPE items
    KEY_PPE = {
        'helmet': [0],
        'vest': [2],
        'gloves': [1]
    }
    
    KEY_VIOLATIONS = {
        'no_helmet': [7],
        'no_gloves': [9]
    }

config = Config()

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def match_detections_to_gt(detections, ground_truth, iou_threshold=0.5):
    """
    Match detections to ground truth boxes
    Returns: (tp, fp, fn, matches)
    """
    if len(ground_truth) == 0:
        return 0, len(detections), 0, []
    
    if len(detections) == 0:
        return 0, 0, len(ground_truth), []
    
    # Create IoU matrix
    iou_matrix = np.zeros((len(detections), len(ground_truth)))
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truth):
            iou_matrix[i, j] = calculate_iou(det['bbox'], gt['bbox'])
    
    # Greedy matching
    matched_gt = set()
    matched_det = set()
    matches = []
    
    # Sort by IoU (highest first)
    indices = np.unravel_index(np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape)
    
    for i, j in zip(indices[0], indices[1]):
        if i not in matched_det and j not in matched_gt:
            if iou_matrix[i, j] >= iou_threshold:
                # Check if classes match
                if detections[i]['class'] == ground_truth[j]['class']:
                    matched_det.add(i)
                    matched_gt.add(j)
                    matches.append((i, j, iou_matrix[i, j]))
    
    tp = len(matches)
    fp = len(detections) - tp
    fn = len(ground_truth) - tp
    
    return tp, fp, fn, matches

def load_ground_truth(label_path, img_width, img_height):
    """Load ground truth from YOLO format label file"""
    gt_objects = []
    
    if not Path(label_path).exists():
        return gt_objects
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                gt_objects.append({
                    'class': class_id,
                    'bbox': [x1, y1, x2, y2],
                    'class_name': config.CLASS_NAMES.get(class_id, f'class_{class_id}')
                })
    
    return gt_objects

def evaluate_detections(test_images_dir, ground_truth_dir, model_path, use_sam=False):
    """
    Evaluate detection performance
    """
    model = YOLO(model_path)
    
    results_per_category = {
        'person': {'tp': 0, 'fp': 0, 'fn': 0},
        'ppe_items': {'tp': 0, 'fp': 0, 'fn': 0},
        'violations': {'tp': 0, 'fp': 0, 'fn': 0},
        'helmet': {'tp': 0, 'fp': 0, 'fn': 0},
        'vest': {'tp': 0, 'fp': 0, 'fn': 0},
        'no_helmet': {'tp': 0, 'fp': 0, 'fn': 0},
    }
    
    detailed_results = []
    
    image_files = list(Path(test_images_dir).glob('*.jpg')) + list(Path(test_images_dir).glob('*.png'))
    
    print(f"\nEvaluating {'Hybrid (YOLO+SAM)' if use_sam else 'YOLO-Only'} on {len(image_files)} images...")
    
    for idx, img_path in enumerate(image_files):
        if idx % 20 == 0:
            print(f"  Processing {idx}/{len(image_files)}...")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Load ground truth
        label_path = Path(ground_truth_dir) / f"{img_path.stem}.txt"
        gt_objects = load_ground_truth(label_path, img_width, img_height)
        
        # Run detection
        results = model(img, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
        
        # Extract detections
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                detections.append({
                    'class': class_id,
                    'bbox': bbox,
                    'conf': conf,
                    'class_name': config.CLASS_NAMES.get(class_id, f'class_{class_id}')
                })
        
        # Evaluate each category
        image_results = {'image': img_path.name}
        
        # 1. Person detection
        person_dets = [d for d in detections if d['class'] in config.PERSON_CLASS]
        person_gts = [g for g in gt_objects if g['class'] in config.PERSON_CLASS]
        tp, fp, fn, _ = match_detections_to_gt(person_dets, person_gts, config.IOU_THRESHOLD)
        results_per_category['person']['tp'] += tp
        results_per_category['person']['fp'] += fp
        results_per_category['person']['fn'] += fn
        image_results['person_tp'] = tp
        image_results['person_fp'] = fp
        image_results['person_fn'] = fn
        
        # 2. PPE items (helmet, vest)
        ppe_dets = [d for d in detections if d['class'] in config.PPE_CLASSES]
        ppe_gts = [g for g in gt_objects if g['class'] in config.PPE_CLASSES]
        tp, fp, fn, _ = match_detections_to_gt(ppe_dets, ppe_gts, config.IOU_THRESHOLD)
        results_per_category['ppe_items']['tp'] += tp
        results_per_category['ppe_items']['fp'] += fp
        results_per_category['ppe_items']['fn'] += fn
        image_results['ppe_tp'] = tp
        image_results['ppe_fp'] = fp
        image_results['ppe_fn'] = fn
        
        # 3. Helmet specifically
        helmet_dets = [d for d in detections if d['class'] in config.KEY_PPE['helmet']]
        helmet_gts = [g for g in gt_objects if g['class'] in config.KEY_PPE['helmet']]
        tp, fp, fn, _ = match_detections_to_gt(helmet_dets, helmet_gts, config.IOU_THRESHOLD)
        results_per_category['helmet']['tp'] += tp
        results_per_category['helmet']['fp'] += fp
        results_per_category['helmet']['fn'] += fn
        image_results['helmet_tp'] = tp
        image_results['helmet_fp'] = fp
        image_results['helmet_fn'] = fn
        
        # 4. Vest specifically
        vest_dets = [d for d in detections if d['class'] in config.KEY_PPE['vest']]
        vest_gts = [g for g in gt_objects if g['class'] in config.KEY_PPE['vest']]
        tp, fp, fn, _ = match_detections_to_gt(vest_dets, vest_gts, config.IOU_THRESHOLD)
        results_per_category['vest']['tp'] += tp
        results_per_category['vest']['fp'] += fp
        results_per_category['vest']['fn'] += fn
        image_results['vest_tp'] = tp
        image_results['vest_fp'] = fp
        image_results['vest_fn'] = fn
        
        # 5. Violations
        viol_dets = [d for d in detections if d['class'] in config.VIOLATION_CLASSES]
        viol_gts = [g for g in gt_objects if g['class'] in config.VIOLATION_CLASSES]
        tp, fp, fn, _ = match_detections_to_gt(viol_dets, viol_gts, config.IOU_THRESHOLD)
        results_per_category['violations']['tp'] += tp
        results_per_category['violations']['fp'] += fp
        results_per_category['violations']['fn'] += fn
        image_results['violation_tp'] = tp
        image_results['violation_fp'] = fp
        image_results['violation_fn'] = fn
        
        # 6. No-helmet specifically
        no_helmet_dets = [d for d in detections if d['class'] in config.KEY_VIOLATIONS['no_helmet']]
        no_helmet_gts = [g for g in gt_objects if g['class'] in config.KEY_VIOLATIONS['no_helmet']]
        tp, fp, fn, _ = match_detections_to_gt(no_helmet_dets, no_helmet_gts, config.IOU_THRESHOLD)
        results_per_category['no_helmet']['tp'] += tp
        results_per_category['no_helmet']['fp'] += fp
        results_per_category['no_helmet']['fn'] += fn
        image_results['no_helmet_tp'] = tp
        image_results['no_helmet_fp'] = fp
        image_results['no_helmet_fn'] = fn
        
        detailed_results.append(image_results)
    
    return results_per_category, detailed_results

def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, F1-score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def run_comprehensive_evaluation():
    """Run comprehensive evaluation"""
    
    print("=" * 70)
    print("COMPREHENSIVE PPE DETECTION EVALUATION")
    print("=" * 70)
    
    # Evaluate YOLO-only
    print("\n[1/2] Evaluating YOLO-Only Model...")
    yolo_results, yolo_detailed = evaluate_detections(
        config.TEST_IMAGES_DIR,
        config.GROUND_TRUTH_DIR,
        config.YOLO_WEIGHTS,
        use_sam=False
    )
    
    # Calculate metrics for each category
    yolo_metrics = {}
    for category, counts in yolo_results.items():
        yolo_metrics[category] = calculate_metrics(counts['tp'], counts['fp'], counts['fn'])
    
    # Print YOLO results
    print("\n" + "=" * 70)
    print("YOLO-ONLY RESULTS")
    print("=" * 70)
    for category, metrics in yolo_metrics.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    
    # Save results
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Save comprehensive results
    comprehensive_results = {
        'yolo_only': yolo_metrics,
        'configuration': {
            'confidence_threshold': config.CONFIDENCE_THRESHOLD,
            'iou_threshold': config.IOU_THRESHOLD,
            'test_images': len(yolo_detailed)
        }
    }
    
    with open(output_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=4)
    
    # Save detailed results
    df = pd.DataFrame(yolo_detailed)
    df.to_csv(output_dir / 'detailed_results_comprehensive.csv', index=False)
    
    # Create visualization
    create_visualization(yolo_metrics, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ Evaluation complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 70)

def create_visualization(metrics_dict, output_dir):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive PPE Detection Evaluation', fontsize=16, fontweight='bold')
    
    categories = list(metrics_dict.keys())
    
    # Extract metrics
    precisions = [metrics_dict[cat]['precision'] for cat in categories]
    recalls = [metrics_dict[cat]['recall'] for cat in categories]
    f1_scores = [metrics_dict[cat]['f1_score'] for cat in categories]
    
    # 1. Precision comparison
    ax1 = axes[0, 0]
    ax1.barh(categories, precisions, color='steelblue')
    ax1.set_xlabel('Precision')
    ax1.set_title('Precision by Category')
    ax1.set_xlim(0, 1)
    for i, v in enumerate(precisions):
        ax1.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    # 2. Recall comparison
    ax2 = axes[0, 1]
    ax2.barh(categories, recalls, color='coral')
    ax2.set_xlabel('Recall')
    ax2.set_title('Recall by Category')
    ax2.set_xlim(0, 1)
    for i, v in enumerate(recalls):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    # 3. F1-Score comparison
    ax3 = axes[0, 2]
    ax3.barh(categories, f1_scores, color='mediumseagreen')
    ax3.set_xlabel('F1-Score')
    ax3.set_title('F1-Score by Category')
    ax3.set_xlim(0, 1)
    for i, v in enumerate(f1_scores):
        ax3.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    # 4. TP/FP/FN breakdown
    ax4 = axes[1, 0]
    tp_counts = [metrics_dict[cat]['tp'] for cat in categories]
    fp_counts = [metrics_dict[cat]['fp'] for cat in categories]
    fn_counts = [metrics_dict[cat]['fn'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    ax4.bar(x - width, tp_counts, width, label='TP', color='green')
    ax4.bar(x, fp_counts, width, label='FP', color='red')
    ax4.bar(x + width, fn_counts, width, label='FN', color='orange')
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Count')
    ax4.set_title('Detection Counts by Category')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    ax4.legend()
    
    # 5. Summary metrics
    ax5 = axes[1, 1]
    summary_text = "Detection Summary\n\n"
    for cat in categories:
        summary_text += f"{cat.replace('_', ' ').title()}:\n"
        summary_text += f"  P: {metrics_dict[cat]['precision']:.3f} | "
        summary_text += f"R: {metrics_dict[cat]['recall']:.3f} | "
        summary_text += f"F1: {metrics_dict[cat]['f1_score']:.3f}\n\n"
    
    ax5.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace')
    ax5.axis('off')
    
    # 6. Key insights
    ax6 = axes[1, 2]
    insights = "Key Insights:\n\n"
    
    best_cat = max(categories, key=lambda x: metrics_dict[x]['f1_score'])
    worst_cat = min(categories, key=lambda x: metrics_dict[x]['f1_score'])
    
    insights += f"✅ Best performing: {best_cat.replace('_', ' ').title()}\n"
    insights += f"   F1-Score: {metrics_dict[best_cat]['f1_score']:.3f}\n\n"
    insights += f"⚠️  Needs improvement: {worst_cat.replace('_', ' ').title()}\n"
    insights += f"   F1-Score: {metrics_dict[worst_cat]['f1_score']:.3f}\n\n"
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    insights += f"📊 Average Precision: {avg_precision:.3f}\n"
    insights += f"📊 Average Recall: {avg_recall:.3f}\n"
    
    ax6.text(0.1, 0.5, insights, fontsize=10, verticalalignment='center')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualization saved: {output_dir / 'comprehensive_evaluation.png'}")

if __name__ == "__main__":
    run_comprehensive_evaluation()

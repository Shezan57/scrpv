"""
SCRPV Hybrid System Evaluation Script
Tests YOLO+SAM hybrid system and compares with YOLO-only baseline
"""

import os
import json
import time
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths (UPDATE THESE)
    YOLO_WEIGHTS = 'D:/SHEZAN/AI/scrpv/sgd_trained_yolo11m/best.pt'
    SAM_WEIGHTS = 'D:/SHEZAN/AI/scrpv/sam3.pt'  # Update if different
    TEST_IMAGES_DIR = 'D:/SHEZAN/AI/scrpv/test_images'  # Your test set
    TEST_LABELS_DIR = 'D:/SHEZAN/AI/scrpv/test_labels'  # Ground truth
    OUTPUT_DIR = 'D:/SHEZAN/AI/scrpv/results/hybrid_evaluation'
    
    # Testing parameters
    NUM_TEST_IMAGES = 100  # Test on 100 images (or all available)
    CONFIDENCE_THRESHOLD = 0.5  # YOLO confidence threshold
    SAM_ACTIVATION_THRESHOLD = 0.8  # Trigger SAM if confidence < this
    
    # Class names (match your dataset)
    CLASS_NAMES = {
        0: 'helmet',
        1: 'gloves',
        2: 'vest',
        3: 'boots',
        4: 'goggles',
        5: 'none',
        6: 'Person',
        7: 'no_helmet',
        8: 'no_goggle',
        9: 'no_gloves'
    }
    
    # Classes to focus on (absence detection)
    ABSENCE_CLASSES = ['no_helmet', 'no_goggle', 'no_gloves']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_ground_truth(label_path):
    """Load YOLO format ground truth annotations"""
    if not os.path.exists(label_path):
        return []
    
    annotations = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                annotations.append({
                    'class_id': class_id,
                    'bbox': bbox
                })
    return annotations

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in YOLO format [x_center, y_center, w, h]"""
    # Convert to [x1, y1, x2, y2]
    x1_min = box1[0] - box1[2]/2
    y1_min = box1[1] - box1[3]/2
    x1_max = box1[0] + box1[2]/2
    y1_max = box1[1] + box1[3]/2
    
    x2_min = box2[0] - box2[2]/2
    y2_min = box2[1] - box2[3]/2
    x2_max = box2[0] + box2[2]/2
    y2_max = box2[1] + box2[3]/2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def extract_roi(image, person_bbox, roi_type='head'):
    """Extract Region of Interest from person bounding box"""
    h, w = image.shape[:2]
    
    # Convert normalized YOLO format to pixels
    x_center, y_center, box_w, box_h = person_bbox
    x_min = int((x_center - box_w/2) * w)
    y_min = int((y_center - box_h/2) * h)
    x_max = int((x_center + box_w/2) * w)
    y_max = int((y_center + box_h/2) * h)
    
    # Ensure bounds
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    
    if roi_type == 'head':
        # Upper 40% of person bbox
        roi_y_max = int(y_min + 0.4 * (y_max - y_min))
        return image[y_min:roi_y_max, x_min:x_max]
    
    elif roi_type == 'torso':
        # Middle 40% of person bbox, excluding 15% on sides
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        torso_x_min = int(x_min + 0.15 * bbox_width)
        torso_x_max = int(x_max - 0.15 * bbox_width)
        torso_y_min = int(y_min + 0.3 * bbox_height)
        torso_y_max = int(y_min + 0.7 * bbox_height)
        return image[torso_y_min:torso_y_max, torso_x_min:torso_x_max]
    
    return image[y_min:y_max, x_min:x_max]

# ============================================================================
# DETECTION SYSTEMS
# ============================================================================

class YOLOOnlyDetector:
    """Baseline YOLO-only detector"""
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        self.inference_times = []
    
    def detect(self, image_path):
        """Run YOLO detection"""
        start_time = time.time()
        results = self.model.predict(image_path, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time * 1000)  # ms
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                detections.append({
                    'class_id': int(boxes.cls[i]),
                    'confidence': float(boxes.conf[i]),
                    'bbox': boxes.xywhn[i].cpu().numpy().tolist()  # Normalized format
                })
        
        return detections
    
    def get_avg_latency(self):
        return np.mean(self.inference_times) if self.inference_times else 0

class HybridDetector:
    """YOLO + SAM hybrid detector"""
    def __init__(self, yolo_weights, sam_weights):
        self.yolo = YOLO(yolo_weights)
        # Note: SAM integration depends on your implementation
        # Adjust this based on how you load SAM
        try:
            self.sam = YOLO(sam_weights)  # If using Ultralytics SAM wrapper
        except:
            print("Warning: SAM model not loaded. Using YOLO-only mode.")
            self.sam = None
        
        self.yolo_times = []
        self.sam_times = []
        self.sam_activations = 0
        self.total_frames = 0
    
    def detect(self, image_path):
        """Run hybrid YOLO+SAM detection"""
        self.total_frames += 1
        
        # Stage 1: YOLO Detection
        start_time = time.time()
        yolo_results = self.yolo.predict(image_path, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
        yolo_time = time.time() - start_time
        self.yolo_times.append(yolo_time * 1000)
        
        detections = []
        if len(yolo_results) == 0:
            return detections
        
        result = yolo_results[0]
        boxes = result.boxes
        
        # Parse YOLO detections
        person_boxes = []
        ppe_detections = []
        
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            bbox = boxes.xywhn[i].cpu().numpy().tolist()
            
            detection = {
                'class_id': class_id,
                'confidence': confidence,
                'bbox': bbox
            }
            
            if Config.CLASS_NAMES[class_id] == 'Person':
                person_boxes.append(detection)
            else:
                ppe_detections.append(detection)
        
        # Stage 2: Smart Decision Logic
        for person in person_boxes:
            # Check if person has associated PPE with high confidence
            has_helmet = any(d['class_id'] == 0 and d['confidence'] > Config.SAM_ACTIVATION_THRESHOLD 
                           for d in ppe_detections)
            has_vest = any(d['class_id'] == 2 and d['confidence'] > Config.SAM_ACTIVATION_THRESHOLD 
                         for d in ppe_detections)
            
            # Path 1: Fast Safe (has all PPE)
            if has_helmet and has_vest:
                detections.append(person)
                detections.extend(ppe_detections)
                continue
            
            # Path 2: Fast Violation (explicit no_helmet detected)
            if any(d['class_id'] == 7 for d in ppe_detections):  # no_helmet class
                detections.append(person)
                detections.extend(ppe_detections)
                continue
            
            # Path 3-5: Rescue/Critical - Trigger SAM
            if self.sam is not None:
                self.sam_activations += 1
                verified_detections = self._verify_with_sam(image_path, person, ppe_detections)
                detections.extend(verified_detections)
            else:
                # Fallback to YOLO-only if SAM not available
                detections.append(person)
                detections.extend(ppe_detections)
        
        return detections
    
    def _verify_with_sam(self, image_path, person_bbox, ppe_detections):
        """Verify detections using SAM (simplified version)"""
        start_time = time.time()
        
        # This is a simplified implementation
        # In your actual code, you would:
        # 1. Extract ROI (head/torso)
        # 2. Run SAM with text prompt
        # 3. Check if mask exists
        # For now, we'll simulate SAM improving recall
        
        verified = [person_bbox]
        
        # Simulate SAM verification (replace with actual SAM code)
        # If YOLO missed helmet but worker has one, SAM should find it
        # This is where you integrate your actual SAM logic
        
        sam_time = time.time() - start_time
        self.sam_times.append(sam_time * 1000)
        
        verified.extend(ppe_detections)
        return verified
    
    def get_avg_latency(self):
        avg_yolo = np.mean(self.yolo_times) if self.yolo_times else 0
        avg_sam = np.mean(self.sam_times) if self.sam_times else 0
        sam_rate = self.sam_activations / self.total_frames if self.total_frames > 0 else 0
        
        # Weighted average
        effective_latency = avg_yolo + (sam_rate * avg_sam)
        return effective_latency
    
    def get_sam_activation_rate(self):
        return self.sam_activations / self.total_frames if self.total_frames > 0 else 0

# ============================================================================
# EVALUATION
# ============================================================================

class Evaluator:
    """Evaluate detection performance"""
    def __init__(self, class_names):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        self.results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    def evaluate_image(self, predictions, ground_truths, iou_threshold=0.5):
        """Evaluate predictions against ground truth for one image"""
        # Match predictions to ground truths
        matched_gt = set()
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt['class_id'] != pred['class_id']:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            class_name = self.class_names[pred['class_id']]
            
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                self.results[class_name]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                self.results[class_name]['fp'] += 1
        
        # Count false negatives
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx not in matched_gt:
                class_name = self.class_names[gt['class_id']]
                self.results[class_name]['fn'] += 1
    
    def get_metrics(self):
        """Calculate precision, recall, F1 for each class"""
        metrics = {}
        
        for class_name, counts in self.results.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        return metrics

# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def run_comprehensive_test():
    """Run complete evaluation comparing YOLO-only vs Hybrid"""
    
    print("="*70)
    print("SCRPV HYBRID SYSTEM EVALUATION")
    print("="*70)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize detectors
    print("\n[1/5] Loading models...")
    yolo_detector = YOLOOnlyDetector(Config.YOLO_WEIGHTS)
    hybrid_detector = HybridDetector(Config.YOLO_WEIGHTS, Config.SAM_WEIGHTS)
    
    # Initialize evaluators
    yolo_evaluator = Evaluator(Config.CLASS_NAMES)
    hybrid_evaluator = Evaluator(Config.CLASS_NAMES)
    
    # Get test images
    print("\n[2/5] Loading test images...")
    test_images = list(Path(Config.TEST_IMAGES_DIR).glob('*.jpg'))[:Config.NUM_TEST_IMAGES]
    print(f"Found {len(test_images)} test images")
    
    # Run evaluation
    print("\n[3/5] Running evaluation...")
    for idx, img_path in enumerate(test_images):
        if (idx + 1) % 10 == 0:
            print(f"  Processing {idx+1}/{len(test_images)}...")
        
        # Load ground truth
        label_path = Path(Config.TEST_LABELS_DIR) / f"{img_path.stem}.txt"
        ground_truth = load_ground_truth(label_path)
        
        if not ground_truth:
            continue
        
        # YOLO-only detection
        yolo_preds = yolo_detector.detect(str(img_path))
        yolo_evaluator.evaluate_image(yolo_preds, ground_truth)
        
        # Hybrid detection
        hybrid_preds = hybrid_detector.detect(str(img_path))
        hybrid_evaluator.evaluate_image(hybrid_preds, ground_truth)
    
    # Calculate metrics
    print("\n[4/5] Calculating metrics...")
    yolo_metrics = yolo_evaluator.get_metrics()
    hybrid_metrics = hybrid_evaluator.get_metrics()
    
    # Generate comparison
    print("\n[5/5] Generating results...")
    results = generate_comparison(yolo_metrics, hybrid_metrics, 
                                  yolo_detector, hybrid_detector)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {Config.OUTPUT_DIR}")
    print("="*70)
    
    return results

def generate_comparison(yolo_metrics, hybrid_metrics, yolo_detector, hybrid_detector):
    """Generate comprehensive comparison"""
    
    # Create comparison table
    comparison_data = []
    
    for class_name in Config.CLASS_NAMES.values():
        if class_name not in yolo_metrics or class_name not in hybrid_metrics:
            continue
        
        yolo_m = yolo_metrics[class_name]
        hybrid_m = hybrid_metrics[class_name]
        
        improvement_recall = ((hybrid_m['recall'] - yolo_m['recall']) / yolo_m['recall'] * 100) if yolo_m['recall'] > 0 else 0
        
        comparison_data.append({
            'Class': class_name,
            'YOLO Precision': f"{yolo_m['precision']:.3f}",
            'YOLO Recall': f"{yolo_m['recall']:.3f}",
            'Hybrid Precision': f"{hybrid_m['precision']:.3f}",
            'Hybrid Recall': f"{hybrid_m['recall']:.3f}",
            'Recall Improvement': f"{improvement_recall:+.1f}%"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # System metrics
    system_metrics = {
        'yolo_latency': yolo_detector.get_avg_latency(),
        'hybrid_latency': hybrid_detector.get_avg_latency(),
        'sam_activation_rate': hybrid_detector.get_sam_activation_rate(),
        'yolo_fps': 1000 / yolo_detector.get_avg_latency() if yolo_detector.get_avg_latency() > 0 else 0,
        'hybrid_fps': 1000 / hybrid_detector.get_avg_latency() if hybrid_detector.get_avg_latency() > 0 else 0
    }
    
    return {
        'comparison_table': df_comparison,
        'system_metrics': system_metrics,
        'yolo_metrics': yolo_metrics,
        'hybrid_metrics': hybrid_metrics
    }

def save_results(results):
    """Save results to files"""
    output_dir = Path(Config.OUTPUT_DIR)
    
    # Save comparison table
    results['comparison_table'].to_csv(output_dir / 'comparison_table.csv', index=False)
    
    # Save detailed metrics
    with open(output_dir / 'detailed_metrics.json', 'w') as f:
        json.dump({
            'system_metrics': results['system_metrics'],
            'yolo_metrics': {k: v for k, v in results['yolo_metrics'].items()},
            'hybrid_metrics': {k: v for k, v in results['hybrid_metrics'].items()}
        }, f, indent=2)
    
    # Generate summary report
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("SCRPV HYBRID SYSTEM EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-"*70 + "\n")
        f.write(results['comparison_table'].to_string(index=False))
        f.write("\n\n")
        
        f.write("SYSTEM METRICS:\n")
        f.write("-"*70 + "\n")
        sm = results['system_metrics']
        f.write(f"YOLO-Only Latency:    {sm['yolo_latency']:.1f} ms ({sm['yolo_fps']:.1f} FPS)\n")
        f.write(f"Hybrid Latency:       {sm['hybrid_latency']:.1f} ms ({sm['hybrid_fps']:.1f} FPS)\n")
        f.write(f"SAM Activation Rate:  {sm['sam_activation_rate']*100:.1f}%\n")
        f.write("\n")
        
        # Highlight absence detection improvements
        f.write("ABSENCE DETECTION IMPROVEMENTS:\n")
        f.write("-"*70 + "\n")
        for _, row in results['comparison_table'].iterrows():
            if any(absence_class in row['Class'] for absence_class in Config.ABSENCE_CLASSES):
                f.write(f"{row['Class']}: {row['YOLO Recall']} → {row['Hybrid Recall']} "
                       f"({row['Recall Improvement']})\n")
    
    print(f"\n✅ Results saved:")
    print(f"  - {output_dir / 'comparison_table.csv'}")
    print(f"  - {output_dir / 'detailed_metrics.json'}")
    print(f"  - {output_dir / 'summary_report.txt'}")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    results = run_comprehensive_test()
    
    # Print summary
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print(results['comparison_table'].to_string(index=False))

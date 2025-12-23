"""
COPY-PASTE READY CODE FOR GOOGLE COLAB
Complete Fixed Quantitative Analysis

Instructions:
1. Replace Cell 4 (Configuration) with the Config class below
2. Add this entire file as Cell 6 (after utilities)
3. Run the evaluation at the end
"""

# ============================================================================
# CELL 4: CONFIGURATION (REPLACE EXISTING)
# ============================================================================

class Config:
    # Paths - MODIFY FOR YOUR COLAB ENVIRONMENT
    YOLO_WEIGHTS = '/content/best.pt'
    SAM_WEIGHTS = '/content/sam3.pt'
    TEST_IMAGES_DIR = '/content/images/test'
    GROUND_TRUTH_DIR = '/content/labels/test'
    OUTPUT_DIR = '/content/results'
    
    # Detection parameters - OPTIMIZED
    CONFIDENCE_THRESHOLD = 0.25  # Lower to detect more
    IOU_THRESHOLD = 0.5
    SAM_IMAGE_SIZE = 1024
    
    # Class definitions from your model
    CLASS_NAMES = {
        0: 'helmet', 1: 'gloves', 2: 'vest', 3: 'boots',
        4: 'goggles', 5: 'none', 6: 'Person', 7: 'no_helmet',
        8: 'no_goggle', 9: 'no_gloves'
    }
    
    # Categories to evaluate - Hierarchical Decision System Core Classes
    KEY_CLASSES = {
        'person': [6],      # Step 1: Person detected?
        'helmet': [0],      # Step 2: Helmet present?
        'vest': [2],        # Step 2: Vest present?
        'no_helmet': [7]    # Step 3: Violation fast path
    }

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
print("✅ Configuration loaded")


# ============================================================================
# CELL 6: MULTI-CATEGORY EVALUATION FUNCTIONS (ADD NEW CELL)
# ============================================================================

def match_detections_with_gt(detections, ground_truths, iou_threshold):
    """
    Match detections to ground truth boxes using IoU threshold
    Returns: (true_positives, false_positives, false_negatives)
    """
    if len(ground_truths) == 0:
        return 0, len(detections), 0
    
    if len(detections) == 0:
        return 0, 0, len(ground_truths)
    
    matched_gt_indices = set()
    true_positives = 0
    
    # For each detection, find best matching ground truth
    for det in detections:
        best_iou = 0
        best_gt_idx = -1
        
        for idx, gt in enumerate(ground_truths):
            if idx in matched_gt_indices:
                continue  # Already matched
            
            iou = calculate_iou(det['bbox'], gt['bbox'])
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = idx
        
        if best_gt_idx >= 0:
            true_positives += 1
            matched_gt_indices.add(best_gt_idx)
    
    false_positives = len(detections) - true_positives
    false_negatives = len(ground_truths) - true_positives
    
    return true_positives, false_positives, false_negatives


def evaluate_per_category(model, test_dir, gt_dir, categories_to_eval):
    """
    Evaluate model performance for each category separately
    
    Args:
        model: YOLO model
        test_dir: Path to test images
        gt_dir: Path to ground truth labels
        categories_to_eval: Dict of category names to class IDs
    
    Returns:
        Dict with metrics for each category
    """
    results = {cat: {'tp': 0, 'fp': 0, 'fn': 0} for cat in categories_to_eval.keys()}
    detailed_results = []
    
    image_files = glob.glob(os.path.join(test_dir, '*.jpg'))
    total_images = len(image_files)
    
    print(f"📊 Evaluating {total_images} images...")
    
    for idx, img_path in enumerate(image_files):
        if idx % 30 == 0:
            print(f"   Progress: {idx}/{total_images}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        img_name = os.path.basename(img_path)
        
        # Load ground truth
        gt_path = os.path.join(gt_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        gt_annotations = parse_yolo_annotation(gt_path, w, h)
        
        # Run detection
        predictions = model.predict(img_path, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
        
        # Process each category
        image_result = {'image': img_name}
        
        for cat_name, class_ids in categories_to_eval.items():
            # Filter detections for this category
            cat_detections = []
            if predictions.boxes is not None:
                for box in predictions.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in class_ids:
                        coords = box.xyxy[0].cpu().numpy()
                        cat_detections.append({
                            'class': cls_id,
                            'bbox': [float(coords[0]), float(coords[1]), 
                                   float(coords[2]), float(coords[3])]
                        })
            
            # Filter ground truth for this category
            cat_ground_truths = []
            for gt in gt_annotations:
                if gt[0] in class_ids:
                    cat_ground_truths.append({
                        'class': gt[0],
                        'bbox': [float(gt[1]), float(gt[2]), float(gt[3]), float(gt[4])]
                    })
            
            # Calculate TP/FP/FN
            tp, fp, fn = match_detections_with_gt(
                cat_detections, 
                cat_ground_truths, 
                config.IOU_THRESHOLD
            )
            
            results[cat_name]['tp'] += tp
            results[cat_name]['fp'] += fp
            results[cat_name]['fn'] += fn
            
            image_result[f'{cat_name}_tp'] = tp
            image_result[f'{cat_name}_fp'] = fp
            image_result[f'{cat_name}_fn'] = fn
        
        detailed_results.append(image_result)
    
    # Calculate metrics for each category
    metrics = {}
    for cat_name, counts in results.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        
        metrics[cat_name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_detections': tp + fp,
            'total_ground_truth': tp + fn
        }
    
    return metrics, detailed_results


def create_comparison_visualization(metrics_dict, output_path):
    """Create comprehensive visualization of results"""
    
    categories = list(metrics_dict.keys())
    precisions = [metrics_dict[cat]['precision'] for cat in categories]
    recalls = [metrics_dict[cat]['recall'] for cat in categories]
    f1_scores = [metrics_dict[cat]['f1_score'] for cat in categories]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PPE Detection Performance by Category', fontsize=16, fontweight='bold')
    
    # 1. Precision comparison
    ax1 = axes[0, 0]
    bars1 = ax1.barh(categories, precisions, color='steelblue')
    ax1.set_xlabel('Precision', fontsize=12)
    ax1.set_title('Precision by Category', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    for i, (bar, val) in enumerate(zip(bars1, precisions)):
        ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Recall comparison
    ax2 = axes[0, 1]
    bars2 = ax2.barh(categories, recalls, color='coral')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_title('Recall by Category', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    for i, (bar, val) in enumerate(zip(bars2, recalls)):
        ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. F1-Score comparison
    ax3 = axes[1, 0]
    bars3 = ax3.barh(categories, f1_scores, color='mediumseagreen')
    ax3.set_xlabel('F1-Score', fontsize=12)
    ax3.set_title('F1-Score by Category', fontsize=13, fontweight='bold')
    ax3.set_xlim(0, 1.0)
    for i, (bar, val) in enumerate(zip(bars3, f1_scores)):
        ax3.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. TP/FP/FN counts
    ax4 = axes[1, 1]
    tp_counts = [metrics_dict[cat]['tp'] for cat in categories]
    fp_counts = [metrics_dict[cat]['fp'] for cat in categories]
    fn_counts = [metrics_dict[cat]['fn'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    ax4.bar(x - width, tp_counts, width, label='True Positive', color='green', alpha=0.8)
    ax4.bar(x, fp_counts, width, label='False Positive', color='red', alpha=0.8)
    ax4.bar(x + width, fn_counts, width, label='False Negative', color='orange', alpha=0.8)
    
    ax4.set_xlabel('Category', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Detection Counts (TP/FP/FN)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    ax4.legend(loc='upper right')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {output_path}")
    plt.show()


def print_detailed_results(metrics):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("DETECTION PERFORMANCE BY CATEGORY")
    print("="*80)
    print(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-"*80)
    
    for cat_name, metric in metrics.items():
        print(f"{cat_name:<15} "
              f"{metric['precision']:<12.3f} "
              f"{metric['recall']:<12.3f} "
              f"{metric['f1_score']:<12.3f} "
              f"{metric['tp']:<6} "
              f"{metric['fp']:<6} "
              f"{metric['fn']:<6}")
    
    print("="*80)
    
    # Summary statistics
    avg_precision = np.mean([m['precision'] for m in metrics.values()])
    avg_recall = np.mean([m['recall'] for m in metrics.values()])
    avg_f1 = np.mean([m['f1_score'] for m in metrics.values()])
    
    print(f"\n📊 AVERAGE METRICS:")
    print(f"   Precision: {avg_precision:.3f}")
    print(f"   Recall:    {avg_recall:.3f}")
    print(f"   F1-Score:  {avg_f1:.3f}")


print("✅ Multi-category evaluation functions loaded!")


# ============================================================================
# CELL 7: RUN EVALUATION (REPLACE EXISTING EVALUATION CELL)
# ============================================================================

# Load model
print("🔧 Loading YOLO model...")
yolo_model = YOLO(config.YOLO_WEIGHTS)
print("✅ Model loaded!")

# Run evaluation
print("\n🚀 Starting comprehensive evaluation...")
metrics, detailed_results = evaluate_per_category(
    model=yolo_model,
    test_dir=config.TEST_IMAGES_DIR,
    gt_dir=config.GROUND_TRUTH_DIR,
    categories_to_eval=config.KEY_CLASSES
)

# Print results
print_detailed_results(metrics)

# Create visualization
viz_path = os.path.join(config.OUTPUT_DIR, 'category_performance.png')
create_comparison_visualization(metrics, viz_path)

# Save results to JSON
results_path = os.path.join(config.OUTPUT_DIR, 'category_metrics.json')
with open(results_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"💾 Results saved: {results_path}")

# Save detailed CSV
df = pd.DataFrame(detailed_results)
csv_path = os.path.join(config.OUTPUT_DIR, 'detailed_results_per_image.csv')
df.to_csv(csv_path, index=False)
print(f"💾 Detailed results saved: {csv_path}")

print("\n✅ EVALUATION COMPLETE!")
print(f"📁 All results saved to: {config.OUTPUT_DIR}")

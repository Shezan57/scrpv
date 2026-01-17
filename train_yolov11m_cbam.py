"""
Training Script for YOLOv11m-CBAM
Week 1, Day 4-7: Train attention-enhanced model for 200 epochs

Expected Results:
- Epoch 50: mAP@50 >0.70
- Epoch 100: mAP@50 >0.80
- Epoch 150: mAP@50 >0.85
- Best model: mAP@50 0.88-0.92

Training Time: ~40-50 hours on T4 GPU
"""

from ultralytics import YOLO
import torch
import yaml

# Verify GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Training configuration (optimized from your successful baseline)
train_config = {
    # ============ DATA ============
    'data': 'ppe_construction.yaml',
    'imgsz': 640,
    'batch': 16,  # Adjust based on GPU memory
    'cache': False,  # Set True if dataset fits in RAM
    
    # ============ EPOCHS ============
    'epochs': 200,
    'patience': 50,  # Early stopping
    'save': True,
    'save_period': 10,  # Save checkpoint every 10 epochs
    
    # ============ OPTIMIZER (SGD - your proven choice) ============
    'optimizer': 'SGD',
    'lr0': 0.01,       # Initial learning rate
    'lrf': 0.01,       # Final learning rate (0.01 of lr0)
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    
    # ============ AUGMENTATION (your settings) ============
    # Strong augmentation for imbalanced dataset
    'mosaic': 1.0,      # Mosaic augmentation probability
    'mixup': 0.15,      # MixUp probability
    'copy_paste': 0.0,  # Copy-paste augmentation
    
    # Geometric augmentations
    'degrees': 0.0,     # Image rotation (+/- deg)
    'translate': 0.1,   # Image translation (+/- fraction)
    'scale': 0.5,       # Image scale (+/- gain)
    'shear': 0.0,       # Image shear (+/- deg)
    'perspective': 0.0, # Image perspective (+/- fraction)
    'flipud': 0.0,      # Flip up-down probability
    'fliplr': 0.5,      # Flip left-right probability
    
    # Color augmentations
    'hsv_h': 0.015,     # Hue modification
    'hsv_s': 0.7,       # Saturation modification
    'hsv_v': 0.4,       # Value modification
    
    # ============ LOSS ============
    'box': 7.5,         # Box loss weight
    'cls': 0.5,         # Class loss weight
    'dfl': 1.5,         # DFL loss weight
    
    # ============ HARDWARE ============
    'device': device,
    'workers': 8,
    'amp': True,        # Automatic Mixed Precision
    
    # ============ LOGGING ============
    'project': 'runs/train',
    'name': 'yolov11m_cbam_sgd',
    'exist_ok': True,
    'pretrained': True,
    'verbose': True,
    'plots': True,
    'val': True,
}

def main():
    print("="*60)
    print("YOLOv11m-CBAM Training")
    print("="*60)
    
    # TODO: After implementing CBAM integration
    # For now, train baseline to validate config
    print("\n⚠️  CBAM integration not complete yet")
    print("Training baseline YOLOv11m to validate configuration...")
    
    # Load model
    model = YOLO('yolov11m.pt')  # or 'yolov11m.yaml' for training from scratch
    
    # Display configuration
    print("\nTraining Configuration:")
    print(f"  Data: {train_config['data']}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch Size: {train_config['batch']}")
    print(f"  Optimizer: {train_config['optimizer']}")
    print(f"  Initial LR: {train_config['lr0']}")
    print(f"  Device: {device}")
    
    # Train
    print("\nStarting training...")
    results = model.train(**train_config)
    
    # Validation
    print("\n" + "="*60)
    print("Final Validation")
    print("="*60)
    metrics = model.val()
    
    # Print key metrics
    print(f"\nBest Results:")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class mAP@50:")
    class_names = ['Helmet', 'Gloves', 'Vest', 'Boots', 'Goggles', 
                   'None', 'Person', 'No_helmet', 'No_goggle', 'No_gloves', 'No_vest']
    for idx, name in enumerate(class_names):
        if idx < len(metrics.box.maps):
            print(f"  {name:15s}: {metrics.box.maps[idx]:.4f}")
    
    print("\n✅ Training Complete!")
    print(f"Best weights saved to: {model.trainer.best}")
    
    return results, metrics

if __name__ == "__main__":
    # Check if data config exists
    import os
    if not os.path.exists('ppe_construction.yaml'):
        print("⚠️  Warning: ppe_construction.yaml not found!")
        print("Please create data configuration file first.")
        print("\nExample ppe_construction.yaml:")
        print("""
# PPE Construction Dataset Configuration
path: ../datasets/ppe_construction
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: Helmet
  1: Gloves
  2: Vest
  3: Boots
  4: Goggles
  5: None
  6: Person
  7: No_helmet
  8: No_goggle
  9: No_gloves
  10: No_vest
        """)
    else:
        # Run training
        results, metrics = main()

from ultralytics import YOLO
# 1. Load your trained model and get all metrics
model = YOLO(r'D:\SHEZAN\AI\scrpv\sgd_trained_yolo11m\kaggle\working\runs\paper_replication_run\weights\best.pt')
dataset_path = r"D:\SHEZAN\AI\scrpv\results\dataset.yaml"
# 2. Run validation to get metrics
results = model.val(data=dataset_path)

# 3. Print everything you need for tables
print("=== METRICS FOR PAPER ===")
print(f"Overall mAP@50: {results.box.map50:.3f}")
print(f"Overall mAP@50-95: {results.box.map:.3f}")
print("\nPer-class results:")
for i, class_name in enumerate(model.names.values()):
    print(f"{class_name}: mAP={results.box.maps[i]:.3f}, "
          f"Precision={results.box.p[i]:.3f}, "
          f"Recall={results.box.r[i]:.3f}")
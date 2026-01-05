"""
Generate publication-ready comparison figures for the paper
Using the corrected experimental results
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# ============================================================
# DATA FROM CORRECTED EXPERIMENTS (2026-01-04)
# ============================================================
yolo_metrics = {'precision': 0.5882, 'recall': 0.5063, 'f1': 0.5442}
hybrid_metrics = {'precision': 0.625, 'recall': 0.5063, 'f1': 0.5594}

yolo_errors = {'tp': 40, 'fp': 28, 'fn': 39}
hybrid_errors = {'tp': 40, 'fp': 24, 'fn': 39}

decision_paths = {
    'Fast Safe': 145,
    'Fast Violation': 25,
    'Rescue Head': 6,
    'Rescue Body': 11,
    'Critical': 26
}

timing = {
    'yolo_fps': 35.5,
    'hybrid_fps_weighted': 28.6,
    'sam_ms': 1268.7,
    'yolo_ms': 28.3
}

# ============================================================
# FIGURE 1: COMPREHENSIVE COMPARISON (2x2 grid)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('PPE Detection: YOLO-Only vs Hybrid (Fixed ROI) Comparison', 
             fontsize=16, fontweight='bold', y=0.98)

# --- Plot 1: Performance Metrics Comparison ---
ax1 = axes[0, 0]
metrics_names = ['Precision', 'Recall', 'F1-Score']
yolo_vals = [yolo_metrics['precision'], yolo_metrics['recall'], yolo_metrics['f1']]
hybrid_vals = [hybrid_metrics['precision'], hybrid_metrics['recall'], hybrid_metrics['f1']]
x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax1.bar(x - width/2, yolo_vals, width, label='YOLO Only', color='#FF6B6B', alpha=0.85)
bars2 = ax1.bar(x + width/2, hybrid_vals, width, label='Hybrid (YOLO+SAM)', color='#4ECDC4', alpha=0.85)

ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('Performance Metrics Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_names)
ax1.legend(loc='lower right')
ax1.set_ylim(0, 0.8)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Plot 2: Error Reduction ---
ax2 = axes[0, 1]
error_types = ['True Positives', 'False Positives', 'False Negatives']
yolo_err = [yolo_errors['tp'], yolo_errors['fp'], yolo_errors['fn']]
hybrid_err = [hybrid_errors['tp'], hybrid_errors['fp'], hybrid_errors['fn']]

x2 = np.arange(len(error_types))
bars1 = ax2.bar(x2 - width/2, yolo_err, width, label='YOLO Only', color='#FF6B6B', alpha=0.85)
bars2 = ax2.bar(x2 + width/2, hybrid_err, width, label='Hybrid', color='#4ECDC4', alpha=0.85)

ax2.set_ylabel('Count', fontweight='bold')
ax2.set_title('Detection Counts (TP/FP/FN)', fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(error_types, rotation=10)
ax2.legend()

# Highlight FP reduction
ax2.annotate('−14.3%', xy=(1 + width/2, hybrid_errors['fp']), 
            xytext=(1.5, 32), fontsize=11, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

# --- Plot 3: Decision Path Distribution ---
ax3 = axes[1, 0]
paths = list(decision_paths.keys())
counts = list(decision_paths.values())
total = sum(counts)
percentages = [c/total*100 for c in counts]

# Color coding: green for bypass, orange for SAM
colors = ['#2ECC71', '#2ECC71', '#F39C12', '#E67E22', '#8E44AD']
bars = ax3.bar(paths, percentages, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

ax3.set_ylabel('Percentage (%)', fontweight='bold')
ax3.set_xlabel('Decision Path', fontweight='bold')
ax3.set_title('5-Path Decision Distribution (213 Workers)', fontweight='bold')
ax3.set_ylim(0, 85)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=20, ha='right')

# Add value labels with counts
for bar, pct, cnt in zip(bars, percentages, counts):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
            f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=9)

# Add bypass/SAM summary box
bypass_pct = (decision_paths['Fast Safe'] + decision_paths['Fast Violation']) / total * 100
sam_pct = 100 - bypass_pct
textstr = f'SAM Bypassed: {bypass_pct:.1f}%\nSAM Activated: {sam_pct:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax3.text(0.98, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

# --- Plot 4: Throughput/FPS Comparison ---
ax4 = axes[1, 1]

# Create grouped bar for FPS
fps_labels = ['YOLO Only\n(35.5 FPS)', 'Hybrid Raw\n(1.4 FPS)', 'Hybrid Weighted\n(28.6 FPS)']
fps_values = [35.5, 1.4, 28.6]
colors_fps = ['#FF6B6B', '#95A5A6', '#4ECDC4']

bars = ax4.bar(fps_labels, fps_values, color=colors_fps, alpha=0.85, edgecolor='black', linewidth=0.5)
ax4.set_ylabel('Frames Per Second (FPS)', fontweight='bold')
ax4.set_title('Throughput Analysis', fontweight='bold')
ax4.axhline(y=24, color='green', linestyle='--', linewidth=2, label='Real-time threshold (24 FPS)')
ax4.legend(loc='upper right')
ax4.set_ylim(0, 45)

# Value labels
for bar, val in zip(bars, fps_values):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add explanation text
ax4.text(0.5, 0.55, 'Weighted FPS:\n79.8% × 35.5 + 20.2% × 1.4\n= 28.6 FPS', 
        transform=ax4.transAxes, fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('results/paper_comparison_figure.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Saved: results/paper_comparison_figure.png")
plt.show()

# ============================================================
# FIGURE 2: SAM Activation Pie Chart
# ============================================================
fig2, ax = plt.subplots(figsize=(8, 8))

sam_bypass = decision_paths['Fast Safe'] + decision_paths['Fast Violation']
sam_activated = sum(counts) - sam_bypass

sizes = [sam_bypass, sam_activated]
labels = [f'SAM Bypassed\n({sam_bypass} cases, 79.8%)', 
          f'SAM Activated\n({sam_activated} cases, 20.2%)']
colors = ['#2ECC71', '#E67E22']
explode = (0.03, 0)

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                   autopct='%1.1f%%', shadow=True, startangle=90,
                                   textprops={'fontsize': 12})
autotexts[0].set_fontsize(14)
autotexts[0].set_fontweight('bold')
autotexts[1].set_fontsize(14)
autotexts[1].set_fontweight('bold')

ax.set_title('SAM 3 Activation Rate\n(Intelligent Bypass Mechanism)', 
             fontsize=14, fontweight='bold')

plt.savefig('results/sam_activation_pie.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ Saved: results/sam_activation_pie.png")
plt.show()

print("\n🎉 All figures generated successfully!")
print("📁 Files saved to results/ folder")

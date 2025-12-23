"""
Generate Publication-Ready Visualizations from Category Metrics
This creates proper figures for your research paper based on the NEW results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load your NEW results
results_path = r'd:\SHEZAN\AI\scrpv\results\category_metrics.json'
with open(results_path, 'r') as f:
    metrics = json.load(f)

output_dir = Path(r'd:\SHEZAN\AI\scrpv\results')

print("📊 Generating publication-ready visualizations...")
print("=" * 60)

# ============================================================================
# FIGURE 1: Performance Summary (Single Panel - Clean)
# ============================================================================

fig1, ax = plt.subplots(1, 1, figsize=(12, 7))

categories = list(metrics.keys())
precisions = [metrics[cat]['precision'] for cat in categories]
recalls = [metrics[cat]['recall'] for cat in categories]
f1_scores = [metrics[cat]['f1_score'] for cat in categories]

x = np.arange(len(categories))
width = 0.25

bars1 = ax.bar(x - width, precisions, width, label='Precision', 
               color='#3498DB', alpha=0.85, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, recalls, width, label='Recall', 
               color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', 
               color='#2ECC71', alpha=0.85, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Detection Category', fontsize=14, fontweight='bold')
ax.set_title('YOLO Baseline Performance - Hierarchical PPE Detection System', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], 
                    fontsize=12, fontweight='bold')
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

plt.tight_layout()
fig1_path = output_dir / 'figure1_yolo_baseline_performance.png'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure 1 saved: {fig1_path}")
plt.close()

# ============================================================================
# FIGURE 2: Hierarchical System Stages (Multi-panel)
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Hierarchical Detection System Performance by Stage', 
              fontsize=18, fontweight='bold', y=0.995)

# Panel A: Person Detection (Entry Gate)
ax1 = axes[0, 0]
person_metrics = ['Precision', 'Recall', 'F1-Score']
person_vals = [metrics['person']['precision'], 
               metrics['person']['recall'], 
               metrics['person']['f1_score']]
bars = ax1.bar(person_metrics, person_vals, color='#9B59B6', 
               alpha=0.85, edgecolor='black', linewidth=1.2)
ax1.set_title('STEP 1: Person Detection (Entry Gate)', 
              fontsize=13, fontweight='bold')
ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.text(0.5, 0.5, f'F1 = {person_vals[2]:.3f}\n✅ STRONG', 
         transform=ax1.transAxes, fontsize=14, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: PPE Detection (Safety Check)
ax2 = axes[0, 1]
ppe_categories = ['Helmet', 'Vest']
ppe_f1 = [metrics['helmet']['f1_score'], metrics['vest']['f1_score']]
bars = ax2.bar(ppe_categories, ppe_f1, color=['#3498DB', '#E67E22'], 
               alpha=0.85, edgecolor='black', linewidth=1.2)
ax2.set_title('STEP 2: PPE Detection (Safety Check)', 
              fontsize=13, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, ppe_f1):
    ax2.text(bar.get_x() + bar.get_width()/2., val + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.text(0.5, 0.5, f'Avg F1 = {np.mean(ppe_f1):.3f}\n⭐ OUTSTANDING', 
         transform=ax2.transAxes, fontsize=14, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Panel C: Violation Detection (Fast Path)
ax3 = axes[1, 0]
viol_metrics = ['Precision', 'Recall', 'F1-Score']
viol_vals = [metrics['no_helmet']['precision'], 
             metrics['no_helmet']['recall'], 
             metrics['no_helmet']['f1_score']]
bars = ax3.bar(viol_metrics, viol_vals, color='#E74C3C', 
               alpha=0.85, edgecolor='black', linewidth=1.2)
ax3.set_title('STEP 3: Violation Detection (Fast Path)', 
              fontsize=13, fontweight='bold')
ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_ylim(0, 1.0)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, viol_vals):
    ax3.text(bar.get_x() + bar.get_width()/2., val + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.text(0.5, 0.5, f'F1 = {viol_vals[2]:.3f}\n⚠️ LOW\n(Rare Class)', 
         transform=ax3.transAxes, fontsize=14, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Panel D: Detection Counts (TP/FP/FN)
ax4 = axes[1, 1]
categories_short = ['Person', 'Helmet', 'Vest', 'No_helmet']
tp_counts = [metrics[cat]['tp'] for cat in ['person', 'helmet', 'vest', 'no_helmet']]
fp_counts = [metrics[cat]['fp'] for cat in ['person', 'helmet', 'vest', 'no_helmet']]
fn_counts = [metrics[cat]['fn'] for cat in ['person', 'helmet', 'vest', 'no_helmet']]

x = np.arange(len(categories_short))
width = 0.25

ax4.bar(x - width, tp_counts, width, label='True Positive', 
        color='#2ECC71', alpha=0.85, edgecolor='black', linewidth=1.2)
ax4.bar(x, fp_counts, width, label='False Positive', 
        color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1.2)
ax4.bar(x + width, fn_counts, width, label='False Negative', 
        color='#F39C12', alpha=0.85, edgecolor='black', linewidth=1.2)

ax4.set_title('Detection Counts (TP/FP/FN)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
ax4.set_xlabel('Category', fontsize=11, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories_short, rotation=15, ha='right')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig2_path = output_dir / 'figure2_hierarchical_stages.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure 2 saved: {fig2_path}")
plt.close()

# ============================================================================
# FIGURE 3: Performance Gap Analysis (For Paper Discussion)
# ============================================================================

fig3, ax = plt.subplots(1, 1, figsize=(10, 7))

categories_ordered = ['helmet', 'vest', 'person', 'no_helmet']
labels_ordered = ['Helmet\n(PPE)', 'Vest\n(PPE)', 'Person\n(Entry)', 'No_helmet\n(Violation)']
f1_ordered = [metrics[cat]['f1_score'] for cat in categories_ordered]
colors = ['#2ECC71', '#2ECC71', '#3498DB', '#E74C3C']

bars = ax.bar(range(len(labels_ordered)), f1_ordered, color=colors, 
              alpha=0.85, edgecolor='black', linewidth=1.5)

ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Detection Category', fontsize=14, fontweight='bold')
ax.set_title('Performance Gap: PPE Detection vs Violation Detection', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(labels_ordered)))
ax.set_xticklabels(labels_ordered, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars, f1_ordered):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
            f'{val:.3f}', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

# Highlight the gap
ax.axhline(y=np.mean(f1_ordered[:3]), color='green', linestyle='--', 
           linewidth=2, alpha=0.7, label='Avg PPE/Person (0.868)')
ax.axhline(y=f1_ordered[3], color='red', linestyle='--', 
           linewidth=2, alpha=0.7, label='Violation (0.145)')

# Add gap annotation
gap = np.mean(f1_ordered[:3]) - f1_ordered[3]
ax.annotate('', xy=(3, np.mean(f1_ordered[:3])), xytext=(3, f1_ordered[3]),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(3.3, (np.mean(f1_ordered[:3]) + f1_ordered[3])/2, 
        f'Gap:\n{gap:.3f}\n(72%)', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.legend(loc='upper right', fontsize=11)

plt.tight_layout()
fig3_path = output_dir / 'figure3_performance_gap.png'
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure 3 saved: {fig3_path}")
plt.close()

# ============================================================================
# FIGURE 4: Summary Statistics Table (As Image)
# ============================================================================

fig4, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = [
    ['Category', 'Precision', 'Recall', 'F1-Score', 'TP', 'FP', 'FN', 'Status'],
]

for cat in ['person', 'helmet', 'vest', 'no_helmet']:
    m = metrics[cat]
    status = '⭐ Outstanding' if m['f1_score'] > 0.9 else \
             '✅ Very Good' if m['f1_score'] > 0.8 else \
             '✅ Good' if m['f1_score'] > 0.6 else \
             '⚠️ Low (Rare Class)'
    
    table_data.append([
        cat.replace('_', ' ').title(),
        f"{m['precision']:.3f}",
        f"{m['recall']:.3f}",
        f"{m['f1_score']:.3f}",
        str(m['tp']),
        str(m['fp']),
        str(m['fn']),
        status
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.12, 0.12, 0.12, 0.08, 0.08, 0.08, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(8):
    cell = table[(0, i)]
    cell.set_facecolor('#3498DB')
    cell.set_text_props(weight='bold', color='white')

# Color code rows
colors = ['#E8F8F5', '#E8F8F5', '#E8F8F5', '#FCF3CF']
for i in range(1, 5):
    for j in range(8):
        table[(i, j)].set_facecolor(colors[i-1])

plt.title('YOLO Baseline Performance Summary', 
          fontsize=16, fontweight='bold', pad=20)

fig4_path = output_dir / 'figure4_summary_table.png'
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure 4 saved: {fig4_path}")
plt.close()

print("\n" + "=" * 60)
print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Generated Files:")
print(f"  1. figure1_yolo_baseline_performance.png")
print(f"  2. figure2_hierarchical_stages.png")
print(f"  3. figure3_performance_gap.png")
print(f"  4. figure4_summary_table.png")
print("\n🎯 These figures are publication-ready for your research paper!")
print("=" * 60)

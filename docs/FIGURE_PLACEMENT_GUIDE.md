# 📊 FIGURE PLACEMENT GUIDE FOR ENHANCED PAPER
## IEEE Paper Figure Integration Instructions

---

## 🎯 FIGURES TO INSERT (4 Publication-Ready)

All figures are in: `d:\SHEZAN\AI\scrpv\results\`

---

## 📌 FIGURE 1: YOLO Baseline Performance
**File:** `figure1_yolo_baseline_performance.png`

**Location in Paper:** Section 4.3 (After new Table I)

**LaTeX Code:**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{results/figure1_yolo_baseline_performance.png}
    \caption{YOLOv11m Baseline Performance on Core Classes. The bar chart illustrates precision (blue), recall (orange), and F1-score (green) for the 4 hierarchical decision classes. Note the excellent performance on PPE detection (Helmet: 91.38\%, Vest: 86.08\%) contrasted with poor violation detection (No\_helmet: 14.49\%).}
    \label{fig:baseline_performance}
\end{figure}
```

**Insert After:** Table I (tab:quantitative)

**Reference in Text:** 
```latex
Figure \ref{fig:baseline_performance} illustrates the performance asymmetry...
```

---

## ⭐ FIGURE 2: Performance Gap (MOST IMPORTANT!)
**File:** `figure3_performance_gap.png`

**Location in Paper:** Section 4.3.3 (The 76% Performance Gap subsection)

**LaTeX Code:**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.85\columnwidth]{results/figure3_performance_gap.png}
    \caption{The 76\% Performance Gap Between Presence and Absence Detection. This visualization quantifies the core limitation of discriminative classifiers: while YOLO achieves 86.8\% average F1-score on PPE presence detection, it fails catastrophically on violation detection (14.5\% F1), missing 87.5\% of safety violations. This gap directly justifies the SAM 3 rescue mechanism.}
    \label{fig:performance_gap}
\end{figure}
```

**Insert After:** Section 4.3.3 first paragraph

**Reference in Text:**
```latex
As shown in Figure \ref{fig:performance_gap}, the quantitative gap between 
PPE detection (88.8\%) and violation detection (14.5\%) is \textbf{76 percentage points}.
```

**THIS IS YOUR KEY RESEARCH CONTRIBUTION FIGURE!** 🌟

---

## 📌 FIGURE 3: Hierarchical Stages Breakdown
**File:** `figure2_hierarchical_stages.png`

**Location in Paper:** Section 3.3 or 4.3.1

**LaTeX Code:**
```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{results/figure2_hierarchical_stages.png}
    \caption{Hierarchical Detection System Stage-by-Stage Performance. Four-panel breakdown showing: (a) Person detection as entry gate (F1=80.75\%), (b) PPE detection performance (Helmet \& Vest averaging 88.8\% F1), (c) Violation detection failure (No\_helmet: 14.49\% F1), and (d) instance counts demonstrating severe class imbalance (40 violations vs 175 helmets).}
    \label{fig:hierarchical_stages}
\end{figure*}
```

**Note:** Uses `figure*` for two-column width (IEEE format)

**Insert After:** Section 3.3 (methodology) or Section 4.3.1

**Reference in Text:**
```latex
The hierarchical system performance is detailed in Figure \ref{fig:hierarchical_stages}...
```

---

## 📌 FIGURE 4: SAM Activation Distribution
**File:** `sam_activation.png` (from results folder)

**Location in Paper:** Section 4.4 (SAM Rescue Path Activation Analysis)

**LaTeX Code:**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{results/sam_activation.png}
    \caption{Decision Path Distribution Across 199 Detected Persons. The conditional triggering logic achieves 64.8\% bypass rate (Fast Safe + Fast Violation), with SAM rescue activated in only 35.2\% of ambiguous cases (Rescue Head: 5.5\%, Rescue Body: 9.5\%, Critical: 20.1\%). This distribution validates the efficiency of the hybrid architecture.}
    \label{fig:sam_activation}
\end{figure}
```

**Insert After:** Table II (tab:sam_activation)

**Reference in Text:**
```latex
Figure \ref{fig:sam_activation} illustrates the decision path distribution...
```

---

## 📌 FIGURE 5: Summary Table (Optional)
**File:** `figure4_summary_table.png`

**Location in Paper:** Can replace Table I or be added as supplementary

**LaTeX Code:**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{results/figure4_summary_table.png}
    \caption{Complete Performance Metrics Summary Table. Comprehensive view of all evaluation metrics including precision, recall, F1-score, true positives, false positives, false negatives, and ground truth counts for the 4 core hierarchical decision classes.}
    \label{fig:summary_table}
\end{figure}
```

**Note:** This might be better as a traditional LaTeX table (already in paper as Table I)

---

## 📌 EXISTING FIGURES TO KEEP/REPLACE

### Figure: Qualitative Cases
**Files:** `Figure3_Hybrid_Checklist1.png`, `Figure3_Hybrid_Checklist2.png`

**Location:** Section 4.5 (Qualitative Analysis)

**Status:** ✅ Already in paper, keep as-is

### Figure: Training Curves
**File:** `results.png`

**Location:** Section 4.2

**Status:** ✅ Already in paper, can optionally replace with better version

---

## 🎯 RECOMMENDED FIGURE ORDER IN PAPER

1. **Figure 1** (Section 3.3): Hierarchical stages breakdown → Shows system design
2. **Figure 2** (Section 4.2): Training curves → Shows convergence
3. **Figure 3** (Section 4.3): Baseline performance bars → Shows overall results
4. **⭐ Figure 4** (Section 4.3.3): **Performance gap** → **KEY CONTRIBUTION!**
5. **Figure 5** (Section 4.4): SAM activation distribution → Shows efficiency
6. **Figure 6-7** (Section 4.5): Qualitative cases → Shows real-world examples

---

## ✅ QUICK INSERTION CHECKLIST

For each figure:
- [ ] Copy PNG file to LaTeX project folder
- [ ] Add `\usepackage{graphicx}` to preamble (already exists)
- [ ] Insert `\begin{figure}...\end{figure}` block
- [ ] Update `\caption{}` with descriptive text
- [ ] Assign unique `\label{fig:...}`
- [ ] Add reference in text: `Figure \ref{fig:...}`
- [ ] Check width: `\columnwidth` (single) or `\textwidth` (double)
- [ ] Verify placement: `[h]` (here), `[t]` (top), `[b]` (bottom)

---

## 📐 LATEX FIGURE TIPS

### Column Width
```latex
\includegraphics[width=\columnwidth]{file.png}     % Single column
\includegraphics[width=0.85\columnwidth]{file.png} % 85% of column
\includegraphics[width=\textwidth]{file.png}       % Full page width
```

### Figure Position
```latex
[h]   % Here (preferred position)
[t]   % Top of page
[b]   % Bottom of page
[!h]  % Force here
[H]   % Absolutely here (requires \usepackage{float})
```

### Two-Column Figure (IEEE format)
```latex
\begin{figure*}[t]  % Note the asterisk for full width
    ...
\end{figure*}
```

---

## 🎯 PRIORITY INSERTION ORDER

**Do these first (highest impact):**
1. ⭐⭐⭐ **Figure: Performance Gap** (`figure3_performance_gap.png`) - Section 4.3.3
   - This is your PRIMARY contribution visualization!
   
2. ⭐⭐ **Figure: Baseline Performance** (`figure1_yolo_baseline_performance.png`) - Section 4.3
   - Shows complete results overview

3. ⭐⭐ **Figure: SAM Activation** (`sam_activation.png`) - Section 4.4
   - Validates efficiency claim

**Optional (nice to have):**
4. ⭐ **Figure: Hierarchical Stages** (`figure2_hierarchical_stages.png`) - Section 3.3
   - Good for methodology visualization

---

## 📊 ALL FIGURES AT A GLANCE

| Figure | File | Section | Priority | Purpose |
|--------|------|---------|----------|---------|
| Performance Gap | `figure3_performance_gap.png` | 4.3.3 | ⭐⭐⭐ | KEY: 76% gap visualization |
| Baseline Performance | `figure1_yolo_baseline_performance.png` | 4.3 | ⭐⭐ | Overall results |
| SAM Activation | `sam_activation.png` | 4.4 | ⭐⭐ | Efficiency validation |
| Hierarchical Stages | `figure2_hierarchical_stages.png` | 3.3/4.3 | ⭐ | System breakdown |
| Summary Table | `figure4_summary_table.png` | 4.3 | Optional | Alternative to Table I |

---

## ✅ DONE!

Your paper now has:
- ✅ Complete quantitative results in text
- ✅ 2 enhanced tables (Table I & II)
- ✅ 4 publication-ready figures ready to insert
- ✅ All figure captions and labels prepared
- ✅ Cross-references ready

**Next step:** Insert the LaTeX code blocks above into your `.tex` file! 🚀

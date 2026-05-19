# 👁️ Eye-Tracking & Visualization Literacy (Mini-VLAT)

This repository contains the analysis code for an eye-tracking study investigating **visualization literacy** using the **Mini-VLAT test**.

The project focuses on how visual search behavior (gaze patterns) relates to task performance and visualization literacy.

---

## 📌 Project Overview

This study investigates the relationship between:

- Visualization literacy (Mini-VLAT)
- Eye-tracking behavior (fixations, saccades, scanpaths)
- Task performance (accuracy, score)
- Cognitive load and time pressure

Eye-tracking data was recorded using **Tobii eye trackers** and processed using Python-based analysis pipelines.

---

## 🧠 Research Goals

The main objective is to understand how differences in visualization literacy are reflected in gaze behavior.

Key research questions include:

- Do high- and low-performing participants differ in attention allocation?
- How does visual search efficiency (TTFF) relate to performance?
- Are structured scanpaths associated with better performance?
- How do cognitive load and time pressure influence gaze behavior?

---

## 📁 Project Structure


EYETRACKING-VISUALIZATION-LITERACY/
│
├── data/
│ └── testA/
│ ├── .csv / .tsv (AOI metrics, gaze data, answers)
│
├── scripts/
│ └── testA/
│ ├── AOI_analysis/
│ ├── fixation_plots
│ ├── correlation_.py
│ ├── regression_.py
│ ├── scanpath_analysis.py
│ ├── global_statistic_analysis.py
│ ├── cognitive_load_likert.py
│ ├── testB/
│ ├── testC/
│ └── testD/
│
├── results/
│ └── testA/
│ ├── plots/
│ ├── statistics/
│ └── reports/
│
└── README.md


---

## ⚙️ Setup

### Requirements

- Python **3.11**
- pip
- virtual environment (recommended)

---

### 1️⃣ Create virtual environment

```bash
python -m venv .venv
```

### 2️⃣ Activate environment

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Running the Analysis

Navigate to the test condition:

```bash
cd scripts/testA
```

---

## 📊 Core Analyses

### Correlation Analysis
```bash
python correlation_irrelevant_ratio.py
```

```bash
python correlation_transitions.py
```

```bash
python correlation_ttff.py
```

### Regression Analysis
```bash
python regression_irrelevant_ratio.py
```

### Fixation & Heatmaps
```bash
python fixations_plot.py
```

```bash
python fixation_heat.py
```

### Scanpath Analysis
```bash
python scanpath_analysis.py
```

### Global Statistics
```bash
python global_statistic_analysis.py
```

### Cognitive Load Analysis
```bash
python cognitive_load_likert.py
```

---

## 📊 Output

Many Results are automatically saved in:

results/testA/

Including:

Scatterplots (correlations), 
Fixation plots, 
Heatmaps, 
Statistical summaries, 
Regression outputs

---

### 📄 Data Description

The analysis uses:

Eye-Tracking Data
- Fixations
- Saccades
- TTFF (Time to First Fixation)

AOI-Based Metrics
- Dwell time
- Fixation count
- Irrelevant fixation ratio
- Transition counts

Performance Data
- Task scores
- Accuracy per participant

---

## 🧪 Methodology

- Eye-tracking data is segmented using URL Start / URL End events
- Each fixation is mapped to predefined Areas of Interest (AOIs)
- Analysis is performed at:
  - Task level
  - Participant level
  - Cross-task level

Key measures:

- TTFF → visual search efficiency
- Irrelevant fixation ratio → attention allocation
- Transitions → structural gaze behavior
- Scanpaths → temporal organization

---

## 📈 Notes
- Data is aggregated across multiple Mini-VLAT tasks
- Statistical analysis uses Spearman correlations
- Regression models are exploratory (not causal)

---

## 👥 Authors

Assiele Meragi, 
Melissa Posluoglu, 
Gülsen Uzunoglu, 
Eliana Elshani, 
Jessica Belovs 

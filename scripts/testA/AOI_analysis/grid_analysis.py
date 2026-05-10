from gridfix import *
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
import numpy as np

# =========================
# SETTINGS
PARTICIPANT = ("Participant20")
QUESTION_ID =7
GRID_SIZE = (6, 6)
CROP_BOX = (570, 20, 1326, 980)
# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = PROJECT_ROOT / "data" / "testA" / f"{PARTICIPANT}.tsv"
IMAGE_PATH = PROJECT_ROOT / "data" / "testA" / "stimuli" / f"Question{QUESTION_ID}.png"

OUTPUT_DIR = PROJECT_ROOT / "results" / "testA" / PARTICIPANT.lower() / "grid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PATH = OUTPUT_DIR / f"{PARTICIPANT}_Question{QUESTION_ID}_GRID.png"
CSV_PATH = OUTPUT_DIR / f"{PARTICIPANT}_Question{QUESTION_ID}_GRID.csv"

# =========================
# LOAD TSV
# =========================

df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# =========================
# DETECT QUESTION TIME WINDOW
# =========================

events = df[
    (df["Event"] == "URLStart") &
    (df["Event value"].astype(str).str.contains("Question", na=False))
    ].sort_values("Recording timestamp")

current_event = events[
    events["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False)
]

if current_event.empty:
    raise ValueError(f"Question {QUESTION_ID} not found")

start_time = current_event["Recording timestamp"].iloc[0]

next_events = events[events["Recording timestamp"] > start_time]
if not next_events.empty:
    end_time = next_events["Recording timestamp"].iloc[0]
else:
    end_time = df["Recording timestamp"].max()

print(f"Question {QUESTION_ID} window: {start_time} -> {end_time}")

# =========================
# EXTRACT + CLEAN FIXATIONS
# =========================

fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp"] >= start_time) &
    (df["Recording timestamp"] < end_time)
    ].copy()

fix = fix[
    (fix["Gaze event duration"] >= 80) &
    (fix["Gaze event duration"] <= 1000)
    ]

fix = fix.drop_duplicates(subset="Eye movement type index")

print(f"Clean Fixations for Question {QUESTION_ID}: {len(fix)}")

if fix.empty:
    raise ValueError("No valid fixations after cleaning")

# =========================
# LOAD IMAGE + CONVERT TO PX
# =========================

img_full = Image.open(IMAGE_PATH)
full_w, full_h = img_full.size

fix["X_px"] = fix["Fixation point X (MCSnorm)"] * full_w
fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * full_h

# =========================
# MANUAL CALIBRATION OFFSET
# =========================
OFFSETS = {
    "Participant4":(0,-15),
    "Participant5": (0, -35),
    "Participant6": (0, -20),
    "Participant8":(0,-30),
    "Participant10":(0,-30),
    "Participant11":(0,-20),
    "Participant13":(0,-35),
    "Participant15":(0,-30),

    "Participant22":(0,-30),
    "Participant23":(0,-30),
    "Participant24": (0, -35),
    "Participant12": (0, 0),
    "Participant20": (0, -28),
}

x_off, y_off = OFFSETS.get(PARTICIPANT, (0, 0))

fix["X_px"] += x_off
fix["Y_px"] += y_off

# =========================
# APPLY CROP
# =========================

crop_left, crop_top, crop_right, crop_bottom = CROP_BOX

img = img_full.crop(CROP_BOX)
img_w, img_h = img.size

fix["X_crop"] = fix["X_px"] - crop_left
fix["Y_crop"] = fix["Y_px"] - crop_top

fix = fix[
    (fix["X_crop"] >= 0) &
    (fix["X_crop"] < img_w) &
    (fix["Y_crop"] >= 0) &
    (fix["Y_crop"] < img_h)
    ].copy()

print(f"Fixations inside crop: {len(fix)}")

if fix.empty:
    raise ValueError("No fixations inside crop")

# =========================
# CREATE GRID
# =========================

grid = GridRegionSet(size=(img_w, img_h), gridsize=GRID_SIZE, label=f"Q{QUESTION_ID}Grid")

print("Grid:", grid.gridsize)
print("Cells:", len(grid.cells))

# =========================
# MAP FIXATIONS TO GRID CELLS
# =========================

def find_region_id(x, y, cells):
    for i, (left, top, right, bottom) in enumerate(cells, start=1):
        if left <= x < right and top <= y < bottom:
            return i
    return None

fix["regionid"] = fix.apply(
    lambda row: find_region_id(row["X_crop"], row["Y_crop"], grid.cells),
    axis=1
)

fix = fix.dropna(subset=["regionid"]).copy()
fix["regionid"] = fix["regionid"].astype(int)

# =========================
# GRID ENTROPY
# =========================

# Anzahl Fixationen pro Grid-Zelle
counts = fix["regionid"].value_counts().sort_index()

# Wahrscheinlichkeiten pro Grid-Zelle
probs = counts / counts.sum()

# Entropy berechnen
grid_entropy = -np.sum(probs * np.log2(probs))

# Maximale Entropy bei 6x6 Grid
total_regions = GRID_SIZE[0] * GRID_SIZE[1]
max_entropy = np.log2(total_regions)

# Normalisierte Entropy zwischen 0 und 1
normalized_entropy = grid_entropy / max_entropy

print("grid_entropy:", round(grid_entropy, 3))
print("normalized_entropy:", round(normalized_entropy, 3))


# =========================
# TRANSITION MATRIX
# =========================

import pandas as pd

sequence = fix["regionid"].tolist()

# alle Übergänge sammeln
transitions = []

for i in range(len(sequence) - 1):
    from_region = sequence[i]
    to_region = sequence[i + 1]
    transitions.append((from_region, to_region))

# DataFrame
trans_df = pd.DataFrame(transitions, columns=["from_region", "to_region"])

# Matrix
transition_matrix = pd.crosstab(
    trans_df["from_region"],
    trans_df["to_region"]
)

print("Transition Matrix:")
print(transition_matrix)

# speichern
TRANS_PATH = OUTPUT_DIR / f"{PARTICIPANT}_Question{QUESTION_ID}_TRANSITIONS.csv"
transition_matrix.to_csv(TRANS_PATH)

print("Saved transitions:", TRANS_PATH)
# =========================
# ADVANCED GRID METRICS
# =========================

# Reihenfolge der Regionen
sequence = fix["regionid"].tolist()

# 1. first fixated region
first_fixated_region = sequence[0] if sequence else None

# 2. number of visited regions
number_of_visited_regions = len(set(sequence))

# total number of grid cells
total_regions = GRID_SIZE[0] * GRID_SIZE[1]

# 3. grid coverage = Anteil besuchter Grid-Zellen
grid_coverage = number_of_visited_regions / total_regions

# 4. proportion empty regions = Anteil nicht besuchter Grid-Zellen
proportion_empty_regions = 1 - grid_coverage

# 4. revisit count
seen = set()
revisit_count = 0

for r in sequence:
    if r in seen:
        revisit_count += 1
    else:
        seen.add(r)

# 5. transition count
transition_count = max(len(sequence) - 1, 0)

print("first_fixated_region:", first_fixated_region)
print("number_of_visited_regions:", number_of_visited_regions)
print("proportion_empty_regions:", round(proportion_empty_regions, 3))
print("revisit_count:", revisit_count)
print("transition_count:", transition_count)
print("number_of_visited_regions:", number_of_visited_regions)
print("grid_coverage:", round(grid_coverage, 3))
print("grid_coverage_percent:", round(grid_coverage * 100, 1), "%")
print("proportion_empty_regions:", round(proportion_empty_regions, 3))

# =========================
# SUMMARY
# =========================

summary = (
    fix.groupby("regionid")
    .agg(
        fixation_count=("regionid", "size"),
        dwell_time=("Gaze event duration", "sum"),
        mean_duration=("Gaze event duration", "mean")
    )
    .reset_index()
)

grid_regions = grid.info[["regionid"]].copy()
grid_regions["regionid"] = grid_regions["regionid"].astype(int)

summary = grid_regions.merge(summary, on="regionid", how="left")
summary["fixation_count"] = summary["fixation_count"].fillna(0).astype(int)
summary["dwell_time"] = summary["dwell_time"].fillna(0)
summary["mean_duration"] = summary["mean_duration"].fillna(0)

summary.to_csv(CSV_PATH, index=False)
print(f"Saved CSV: {CSV_PATH}")
print(summary)

# =========================
# PLOT GRID + FIXATIONS
# =========================

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
ax.axis("off")

# Grid lines
for cell in grid.cells:
    left, top, right, bottom = cell
    rect = plt.Rectangle(
        (left, top),
        right - left,
        bottom - top,
        fill=False,
        linewidth=2
    )
    ax.add_patch(rect)

# Cell labels
for i, cell in enumerate(grid.cells):
    left, top, right, bottom = cell
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    ax.text(cx, cy, str(i + 1), ha="center", va="center", fontsize=12)

# Fixation sizes based on duration
dur = fix["Gaze event duration"].to_numpy()
dur_scaled = np.clip(dur, 80, 600)
sizes = dur_scaled / 6

ax.scatter(
    fix["X_crop"],
    fix["Y_crop"],
    s=sizes * 1.5,
    color="#cc0000",
    edgecolors="black",
    linewidth=0.6,
    alpha=0.9
)

ax.set_title(f"{PARTICIPANT} - Question {QUESTION_ID} - {GRID_SIZE[0]}x{GRID_SIZE[1]} Grid AOI")

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot: {PLOT_PATH}")
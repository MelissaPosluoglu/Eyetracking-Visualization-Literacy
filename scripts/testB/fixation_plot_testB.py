import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np


# DAS IST FÜR PARTICIPANT 20-30

# ----------------------------------------------------
# Non-interactive backend
# ----------------------------------------------------
matplotlib.use("Agg")

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
PARTICIPANT = ("Participant41")
QUESTION_ID = 12

DATA_FILE = os.path.join("..","..", "..", "data", "testB", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..","..", "..", "data", "testB", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..","..", "..", "results", "testB", PARTICIPANT.lower(), "fixations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Load TSV
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# Detect question time window
# ----------------------------------------------------
events = df[
    (df["Event"] == "URLStart") &
    (df["Event value"].astype(str).str.contains("Question", na=False))
    ].sort_values("Recording timestamp [ms]")

current_event = events[
    events["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False)
]

if current_event.empty:
    print(f"⚠️ Question {QUESTION_ID} not found")
    exit()

start_time = current_event["Recording timestamp [ms]"].iloc[0]

next_events = events[events["Recording timestamp [ms]"] > start_time]

if not next_events.empty:
    end_time = next_events["Recording timestamp [ms]"].iloc[0]
else:
    end_time = df["Recording timestamp [ms]"].max()

# ----------------------------------------------------
# Extract + CLEAN fixations
# ----------------------------------------------------
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp [ms]"] >= start_time) &
    (df["Recording timestamp [ms]"] < end_time)
    ].copy()

# Duration filter
fix = fix[
    (fix["Gaze event duration [ms]"] >= 80) &
    (fix["Gaze event duration [ms]"] <= 1000)
    ]

# Remove duplicates
fix = fix.drop_duplicates(subset="Eye movement type index")

print(f"Clean Fixations for Question {QUESTION_ID}: {len(fix)}")

if fix.empty:
    print("⚠️ No valid fixations after cleaning")
    exit()

# ----------------------------------------------------
# Load stimulus image
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

fix["X_px"] = fix["Fixation point X [MCS norm]"] * w
fix["Y_px"] = fix["Fixation point Y [MCS norm]"] * h

# ----------------------------------------------------
# Scale fixation size by duration
# ----------------------------------------------------
dur = fix["Gaze event duration [ms]"].to_numpy()
dur_scaled = np.clip(dur, 80, 600)
size = dur_scaled / 6

# ----------------------------------------------------
# Plot
# ----------------------------------------------------
dpi = 100
fig_w = w / dpi
fig_h = h / dpi

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

ax.imshow(img)
ax.axis("off")

ax.scatter(
    fix["X_px"],
    fix["Y_px"],
    s=size * 1.5,
    color="#cc0000",
    edgecolors="black",
    linewidth=0.6,
    alpha=0.9
)

plt.tight_layout()

# ----------------------------------------------------
# Save
# ----------------------------------------------------
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_FIXATIONS_CLEAN.png"
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ Clean fixation plot saved.")
print("📁 Saved to:", out_path)
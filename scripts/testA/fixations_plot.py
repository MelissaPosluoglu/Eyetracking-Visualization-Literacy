import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np

# ----------------------------------------------------
# Non-interactive backend
# ----------------------------------------------------
matplotlib.use("Agg")

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
PARTICIPANT = "Participant2"
QUESTION_ID = 2

DATA_FILE = os.path.join("..", "..", "data", "testA", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..", "..", "data", "testA", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..", "..", "results", "testA", PARTICIPANT.lower(), "debug_clean")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Load TSV
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# Find time window for this Question (via Event value)
# ----------------------------------------------------
events = df[df["Event value"].str.contains("Question", na=False)]
events = events.sort_values("Recording timestamp")

current_event = events[
    events["Event value"].str.contains(f"Question {QUESTION_ID}", na=False)
]

if current_event.empty:
    print(f"⚠️ Question {QUESTION_ID} not found in Event value")
    exit()

start_time = current_event["Recording timestamp"].iloc[0]

# end = start of next Question (if exists)
current_idx = current_event.index[0]
next_events = events[events.index > current_idx]

if not next_events.empty:
    end_time = next_events["Recording timestamp"].iloc[0]
else:
    end_time = df["Recording timestamp"].max()

# ----------------------------------------------------
# Filter fixations in this time window
# ----------------------------------------------------
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp"] >= start_time) &
    (df["Recording timestamp"] <= end_time)
    ].copy()

print(f"Fixations for Question {QUESTION_ID}: {len(fix)}")
print(fix["Eye movement type index"].nunique())


if fix.empty:
    print("⚠️ No fixations in this time window")
    exit()

# ----------------------------------------------------
# OPTIONAL: limit for visualization
# ----------------------------------------------------
MAX_FIX = 6000
if len(fix) > MAX_FIX:
    fix = fix.sort_values("Recording timestamp").iloc[:MAX_FIX]

# ----------------------------------------------------
# Load stimulus
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

fix["X_px"] = fix["Fixation point X (MCSnorm)"] * w
fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * h

# ----------------------------------------------------
# Normalize time & duration
# ----------------------------------------------------
t = fix["Recording timestamp"].to_numpy()
t_norm = (t - t.min()) / (t.max() - t.min())

dur = fix["Gaze event duration"].to_numpy()
dur_clipped = np.clip(dur, 40, 400)
size = dur_clipped / 12

# ----------------------------------------------------
# Plot
# ----------------------------------------------------
# ----------------------------------------------------
# Plot (visual-quality version)
# ----------------------------------------------------
# ----------------------------------------------------
# Plot (FINAL, CLEAN, READABLE)
# ----------------------------------------------------

# Figure size proportional to image
dpi = 100
fig_w = w / dpi
fig_h = h / dpi

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

ax.imshow(img)
ax.axis("off")

sc = ax.scatter(
    fix["X_px"],
    fix["Y_px"],
    s=size * 10.0,
    color = "#b22222",   # firebrick
alpha=0.5
)


ax.set_title(
    f"{PARTICIPANT} – Question {QUESTION_ID}",
    fontsize=18,
    pad=20
)

plt.tight_layout()


# ----------------------------------------------------
# Save
# ----------------------------------------------------
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_FIXATIONS.png"
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ Fixation plot saved successfully.")

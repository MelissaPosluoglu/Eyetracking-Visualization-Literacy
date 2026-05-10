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
PARTICIPANT = "Participant61"
QUESTION_ID = 1

DATA_FILE = os.path.join("..", "..", "data", "testD", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..", "..", "data", "testD", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..", "..", "results", "testD", PARTICIPANT.lower(), "fixations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_FIX_DURATION = 80
MAX_FIX_DURATION = 1000
TOP_TEXT_THRESHOLD = 0.28

TOP_TEXT_SHIFTS = {
    "Participant61": 0.00,
    "Participant62": 0.00,
    "Participant63": 0.00,
    "Participant64": 0.00,
    "Participant65": 0.00,
}

# ----------------------------------------------------
# Same shift logic as scanpath
# ----------------------------------------------------
def apply_top_text_shift(fix_df, participant):
    fix_df = fix_df.copy()

    fix_df["X_shifted"] = fix_df["Fixation point X [MCS norm]"].copy()
    fix_df["Y_shifted"] = fix_df["Fixation point Y [MCS norm]"].copy()

    top_text_y_shift = TOP_TEXT_SHIFTS.get(participant, 0.00)

    mask_top = fix_df["Y_shifted"] < TOP_TEXT_THRESHOLD
    fix_df.loc[mask_top, "Y_shifted"] = (
            fix_df.loc[mask_top, "Y_shifted"] + top_text_y_shift
    )

    fix_df["X_shifted"] = fix_df["X_shifted"].clip(0, 1)
    fix_df["Y_shifted"] = fix_df["Y_shifted"].clip(0, 1)

    return fix_df

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
    print(f"Question {QUESTION_ID} not found")
    raise SystemExit

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
    (fix["Gaze event duration [ms]"] >= MIN_FIX_DURATION) &
    (fix["Gaze event duration [ms]"] <= MAX_FIX_DURATION)
    ]

# Remove duplicates
if "Eye movement type index" in fix.columns:
    fix = fix.drop_duplicates(subset="Eye movement type index")

# Nur gültige normierte Punkte behalten
fix = fix[
    (fix["Fixation point X [MCS norm]"].between(0, 1)) &
    (fix["Fixation point Y [MCS norm]"].between(0, 1))
    ].copy()

print(f"Clean Fixations for Question {QUESTION_ID}: {len(fix)}")

if fix.empty:
    print("No valid fixations after cleaning")
    raise SystemExit

# ----------------------------------------------------
# Load stimulus image
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

# ----------------------------------------------------
# Apply same shift logic as scanpath
# ----------------------------------------------------
fix = apply_top_text_shift(fix, PARTICIPANT)

fix["X_px"] = fix["X_shifted"] * w
fix["Y_px"] = fix["Y_shifted"] * h

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
    f"{PARTICIPANT}_Question{QUESTION_ID}_FIXATIONS_CLEAN_SHIFTED.png"
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Clean fixation plot with shifted top-text region saved.")
print("Saved to:", out_path)
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np

# ============================================================
# NON-INTERACTIVE BACKEND
# ============================================================
matplotlib.use("Agg")

# ============================================================
# CONFIG
# ============================================================
PARTICIPANT = "Participant53"
QUESTION_ID =3

DATA_FILE = os.path.join("..", "..", "data", "testC", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..", "..", "data", "testA", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..", "..", "results", "testA", PARTICIPANT.lower(), "fixations_clean")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_FIX_DURATION = 80
MAX_FIX_DURATION = 1000
TOP_TEXT_THRESHOLD = 0.28

TOP_TEXT_SHIFTS = {
    "Participant1": 0.04,
    "Participant4": -0.02,
    "Participant5": -0.04,
    "Participant8": -0.06,
    "Participant10": 0.00,
    "Participant12": -0.08,
    "Participant13": -0.04,
    "Participant14": -0.10,
    "Participant15": -0.07,
    "Participant16": 0.03,
    "Participant20": -0.05,
    "Participant21": 0.00,
}

# ============================================================
# SAME SHIFT LOGIC AS SCANPATH
# ============================================================
def apply_top_text_shift(fix_df, participant):
    fix_df = fix_df.copy()

    fix_df["X_shifted"] = fix_df["Fixation point X (MCSnorm)"].copy()
    fix_df["Y_shifted"] = fix_df["Fixation point Y (MCSnorm)"].copy()

    top_text_y_shift = TOP_TEXT_SHIFTS.get(participant, 0.00)

    mask_top = fix_df["Y_shifted"] < TOP_TEXT_THRESHOLD
    fix_df.loc[mask_top, "Y_shifted"] = (
            fix_df.loc[mask_top, "Y_shifted"] + top_text_y_shift
    )

    fix_df["X_shifted"] = fix_df["X_shifted"].clip(0, 1)
    fix_df["Y_shifted"] = fix_df["Y_shifted"].clip(0, 1)

    return fix_df

# ============================================================
# LOAD TSV
# ============================================================
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ============================================================
# DETECT QUESTION TIME WINDOW
# ============================================================
events = df[
    (df["Event"] == "URLStart") &
    (df["Event value"].astype(str).str.contains("Question", na=False))
    ].sort_values("Recording timestamp")

current_event = events[
    events["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False)
]

if current_event.empty:
    raise RuntimeError(f"Question {QUESTION_ID} not found")

start_time = current_event["Recording timestamp"].iloc[0]

next_events = events[events["Recording timestamp"] > start_time]

if not next_events.empty:
    end_time = next_events["Recording timestamp"].iloc[0]
else:
    end_time = df["Recording timestamp"].max()

# ============================================================
# EXTRACT + CLEAN FIXATIONS
# ============================================================
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp"] >= start_time) &
    (df["Recording timestamp"] < end_time)
    ].copy()

if "Eye movement type index" in fix.columns:
    fix = fix.drop_duplicates(subset="Eye movement type index")

fix = fix[
    (fix["Gaze event duration"] >= MIN_FIX_DURATION) &
    (fix["Gaze event duration"] <= MAX_FIX_DURATION)
    ]

fix = fix[
    (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
    (fix["Fixation point Y (MCSnorm)"].between(0, 1))
    ].copy()

print(f"Clean Fixations for Question {QUESTION_ID}: {len(fix)}")

if fix.empty:
    raise RuntimeError("No valid fixations after cleaning")

# ============================================================
# LOAD IMAGE
# ============================================================
img = Image.open(IMAGE_PATH)
w, h = img.size

# ============================================================
# APPLY SAME SHIFT LOGIC AS SCANPATH
# ============================================================
fix = apply_top_text_shift(fix, PARTICIPANT)

fix["X_px"] = fix["X_shifted"] * w
fix["Y_px"] = fix["Y_shifted"] * h

# ============================================================
# SCALE FIXATION SIZE BY DURATION
# ============================================================
dur = fix["Gaze event duration"].to_numpy()
dur_scaled = np.clip(dur, 80, 600)
size = dur_scaled / 6

# ============================================================
# PLOT
# ============================================================
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

# ============================================================
# SAVE
# ============================================================
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_FIXATIONS_CLEAN_SHIFTED.png"
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Clean fixation plot with same shift logic as scanpath saved.")
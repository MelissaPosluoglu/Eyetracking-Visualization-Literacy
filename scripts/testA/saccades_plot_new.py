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
QUESTION_ID = 7  # <<< hier wechseln (z.B. 1 oder 12)

DATA_FILE = os.path.join(
    "..", "..", "data", "testA", f"{PARTICIPANT}.tsv"
)

IMAGE_PATH = os.path.join(
    "..", "..", "data", "testA", "stimuli", f"Question{QUESTION_ID}.png"
)

OUTPUT_DIR = os.path.join(
    "..", "..", "results", "testA", PARTICIPANT.lower(), "saccades_only"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Load TSV
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# Find URL window for this question
# ----------------------------------------------------
url_events = df[
    (df["Event"].isin(["URLStart", "URLEnd"])) &
    (df["Event value"].str.contains(
        f"Question {QUESTION_ID}", na=False
    ))
]

if url_events.empty:
    raise RuntimeError(f"No URL events found for Question {QUESTION_ID}")

t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

# ----------------------------------------------------
# Select FIXATIONS inside URL window
# ----------------------------------------------------
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp"].between(t_start, t_end))
].copy()

# Keep only valid on-screen fixations
fix = fix[
    (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
    (fix["Fixation point Y (MCSnorm)"].between(0, 1))
]

# Sort temporally
fix = fix.sort_values("Recording timestamp").reset_index(drop=True)

if len(fix) < 2:
    raise RuntimeError("Not enough fixations to compute saccades")

# ----------------------------------------------------
# Load stimulus image
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

fix["X_px"] = fix["Fixation point X (MCSnorm)"] * w
fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * h

# ----------------------------------------------------
# Visualization (SACCADES ONLY – red & clean)
# ----------------------------------------------------
plt.figure(figsize=(5.5, 9))
plt.imshow(img)
plt.gca().invert_yaxis()

# ---- Parameters (für Lesbarkeit)
MIN_SACCADE_LEN = 60   # px – filtert Mikrosakkaden
ALPHA = 0.25           # Transparenz
LINEWIDTH = 1.0        # Linienbreite

for i in range(len(fix) - 1):
    x1, y1 = fix.iloc[i]["X_px"], fix.iloc[i]["Y_px"]
    x2, y2 = fix.iloc[i + 1]["X_px"], fix.iloc[i + 1]["Y_px"]

    # Sakkadenlänge
    dist = np.hypot(x2 - x1, y2 - y1)

    if dist < MIN_SACCADE_LEN:
        continue

    plt.plot(
        [x1, x2],
        [y1, y2],
        color="#d32f2f",   # <<< dunkles, angenehmes Rot
        alpha=ALPHA,
        linewidth=LINEWIDTH,
        zorder=1
    )

plt.title(
    f"{PARTICIPANT} – Saccades only\nQuestion {QUESTION_ID}",
    fontsize=13
)

plt.axis("off")
plt.tight_layout(pad=0)

plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.xlim(0, w)
plt.ylim(h, 0)

plt.tight_layout()


# ----------------------------------------------------
# Save
# ----------------------------------------------------
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_SaccadesOnly.png"
)

plt.savefig(out_path, dpi=300)
plt.close()

print("✅ Saccades-only visualization saved.")

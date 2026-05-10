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
PARTICIPANT = "Participant51"
QUESTION_ID = 2

DATA_FILE = os.path.join("..", "..", "data", "testC", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..", "..", "data", "testC", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..", "..", "results", "testC", PARTICIPANT.lower(), "saccades")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
ANALYSIS_MIN_PX = 20
VISUAL_MIN_PX = 60
ALPHA = 0.25
LINEWIDTH = 1.0

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# URL / Zeitfenster bestimmen
# ----------------------------------------------------
question_events = df[
    (df["Event"].isin(["URLStart", "URLEnd"])) &
    (df["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False))
    ].copy()

if question_events.empty:
    raise RuntimeError(f"No URL events found for Question {QUESTION_ID}")

# Startzeit
start_rows = question_events[question_events["Event"] == "URLStart"]
if start_rows.empty:
    raise RuntimeError(f"No URLStart found for Question {QUESTION_ID}")

t_start = start_rows["Recording timestamp [ms]"].min()

# Endzeit bevorzugt über URLEnd
end_rows = question_events[
    (question_events["Event"] == "URLEnd") &
    (question_events["Recording timestamp [ms]"] > t_start)
    ]

if not end_rows.empty:
    t_end = end_rows["Recording timestamp [ms]"].max()
else:
    # Fallback: nächster URLStart irgendeiner anderen Frage
    all_urlstarts = df[
        (df["Event"] == "URLStart") &
        (df["Event value"].astype(str).str.contains("Question", na=False))
        ].sort_values("Recording timestamp [ms]")

    next_starts = all_urlstarts[all_urlstarts["Recording timestamp [ms]"] > t_start]

    if not next_starts.empty:
        t_end = next_starts["Recording timestamp [ms]"].iloc[0]
    else:
        t_end = df["Recording timestamp [ms]"].max()

# ----------------------------------------------------
# Fixationen im Zeitfenster
# ----------------------------------------------------
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp [ms]"] >= t_start) &
    (df["Recording timestamp [ms]"] < t_end)
    ].copy()

fix = fix[
    (fix["Fixation point X [MCS norm]"].between(0, 1)) &
    (fix["Fixation point Y [MCS norm]"].between(0, 1))
    ].copy()

# Duplikate entfernen
if "Eye movement type index" in fix.columns:
    fix = fix.drop_duplicates(subset="Eye movement type index")

fix = fix.sort_values("Recording timestamp [ms]").reset_index(drop=True)

if len(fix) < 2:
    raise RuntimeError("Not enough fixations to compute saccades")

# ----------------------------------------------------
# Stimulus laden
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

fix["X_px"] = fix["Fixation point X [MCS norm]"] * w
fix["Y_px"] = fix["Fixation point Y [MCS norm]"] * h

# ----------------------------------------------------
# Sakkaden berechnen
# ----------------------------------------------------
saccades = []

for i in range(len(fix) - 1):
    x1, y1 = fix.loc[i, ["X_px", "Y_px"]]
    x2, y2 = fix.loc[i + 1, ["X_px", "Y_px"]]

    dist = np.hypot(x2 - x1, y2 - y1)

    if dist >= ANALYSIS_MIN_PX:
        saccades.append((x1, y1, x2, y2, dist))

if len(saccades) == 0:
    raise RuntimeError("No valid saccades after filtering.")

# ----------------------------------------------------
# Analysekennwerte
# ----------------------------------------------------
lengths = [s[4] for s in saccades]

print("---- Saccade Statistics ----")
print(f"Number of saccades: {len(lengths)}")
print(f"Mean length (px): {np.mean(lengths):.2f}")
print(f"Median length (px): {np.median(lengths):.2f}")
print(f"Std (px): {np.std(lengths):.2f}")
print("----------------------------")

# ----------------------------------------------------
# Visualisierung
# ----------------------------------------------------
plt.figure(figsize=(5.5, 9))
plt.imshow(img)

for x1, y1, x2, y2, dist in saccades:
    if dist < VISUAL_MIN_PX:
        continue

    plt.plot(
        [x1, x2],
        [y1, y2],
        color="#d32f2f",
        alpha=ALPHA,
        linewidth=LINEWIDTH,
        zorder=1
    )

plt.title(
    f"{PARTICIPANT} – Saccades (cleaned)\nQuestion {QUESTION_ID}",
    fontsize=13
)

plt.axis("off")
plt.tight_layout(pad=0)

# ----------------------------------------------------
# Save
# ----------------------------------------------------
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_SaccadesClean.png"
)

plt.savefig(out_path, dpi=300)
plt.close()

print("Clean, filtered saccade visualization saved.")
print("Saved to:", out_path)
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PATHS
# ============================================================

# Define project, data, stimulus, and output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testA")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTICIPANTS = ["Participant28"]

# ============================================================
# AOIs FOR QUESTION 8
# ============================================================

def get_aois_q8():
    return [

        
        {"name": "question", "x1": 0.31, "y1": 0.03, "x2": 0.68, "y2": 0.165, "type": "relevant"},
        {"name": "price_50", "x1": 0.395, "y1": 0.23, "x2": 0.414, "y2": 0.27, "type": "relevant"},
        {"name": "month_jan", "x1": 0.415, "y1": 0.17, "x2": 0.425, "y2": 0.6, "type": "relevant"},
        {"name": "month_feb", "x1": 0.427, "y1": 0.17, "x2": 0.44, "y2": 0.6, "type": "relevant"},
        {"name": "month_rest", "x1": 0.445, "y1": 0.17, "x2": 0.65, "y2": 0.6, "type": "relevant"},
        {"name": "answers", "x1": 0.28, "y1": 0.64, "x2": 0.72, "y2": 0.92, "type": "relevant"},
        {"name": "background", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "type": "irrelevant"},
    ]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_question_id(event_value):
    m = re.search(r"Question\s+(\d+)", str(event_value))
    return int(m.group(1)) if m else None


def get_fixations_for_question(df, question_label):

    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]

    # Skip if either URLStart or URLEnd is missing
    if len(url_events) < 2:
        return None, None

    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
    t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

    # Fallback for missing or unrealistic trial end times
    if pd.isna(t_end) or (t_end - t_start) > 30000:
        t_end = t_start + 25000

    duration = (t_end - t_start)

    # Select fixation events within the trial time window
    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp"].between(t_start, t_end))
        ].copy()

    # Keep only fixations with valid normalized screen coordinates
    fix = fix[
        (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
        (fix["Fixation point Y (MCSnorm)"].between(0, 1))
        ]

    return fix.sort_values("Recording timestamp").reset_index(drop=True), duration

# ============================================================
# AOI MAPPING
# ============================================================

def map_aois(fix, aois):

    aois_sorted = sorted(
        aois,
        key=lambda a: (a["name"] == "background", (a["x2"]-a["x1"]) * (a["y2"]-a["y1"]))
    )

    aoi_names = []
    aoi_types = []

    for _, row in fix.iterrows():
        x = row["Fixation point X (MCSnorm)"]
        y = row["Fixation point Y (MCSnorm)"]

        assigned = False

        for aoi in aois_sorted:
            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                aoi_names.append(aoi["name"])
                aoi_types.append(aoi["type"])
                assigned = True
                break

        # Fallback if no AOI matches
        if not assigned:
            aoi_names.append("background")
            aoi_types.append("irrelevant")

    fix["AOI"] = aoi_names
    fix["AOI_type"] = aoi_types

    return fix

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fix, duration):

    # Sum dwell time per AOI
    dwell = fix.groupby("AOI")["Gaze event duration"].sum()
    total_dwell = dwell.sum()

    # Normalize dwell time if total fixation duration exceeds trial duration
    if total_dwell > duration and total_dwell > 0:
        scale = duration / total_dwell
        dwell = dwell * scale
        total_dwell = dwell.sum()

    # Compute relative dwell time per AOI
    dwell_ratio = {k: v / total_dwell for k, v in dwell.items()} if total_dwell > 0 else {}

    def ttff(target):
        subset = fix[fix["AOI"] == target]
        if len(subset) == 0:
            return np.nan
        return (subset["Recording timestamp"].iloc[0] - fix["Recording timestamp"].iloc[0])  # ms

    # Create AOI sequence and remove consecutive duplicates
    seq = fix["AOI"].tolist()
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i-1]]

    # Build transition matrix from cleaned scanpath sequence
    if len(seq_clean) >= 2:
        transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
        trans_df = pd.DataFrame(transitions, columns=["from", "to"])
        matrix = pd.crosstab(trans_df["from"], trans_df["to"])
    else:
        matrix = pd.DataFrame()

    transitions_count = max(len(seq_clean) - 1, 0)
 
    # Background dwell ratio after normalization
    irrelevant = dwell.get("background", 0)
    irrelevant_ratio = irrelevant / total_dwell if total_dwell > 0 else 0

    return {
        "TTFF_feb": ttff("month_feb"),
        "TTFF_answers": ttff("answers"),

        "Dwell_feb": dwell.get("month_feb", 0),
        "Dwell_answers": dwell.get("answers", 0),

        "Dwell_ratio_feb": dwell_ratio.get("month_feb", 0),

        "Transitions": transitions_count,
        "Transitions_per_sec": transitions_count / (duration / 1000) if duration > 0 else 0,
        "Sequence_length": len(seq_clean),
        "First_AOI": next((a for a in seq_clean if a != "background"), None),

        "Irrelevant_Ratio": irrelevant_ratio,

        "Transition_Matrix": matrix
    }

# ============================================================
# AOI OVERLAY PNG
# ============================================================

def plot_aoi_overlay():

    img_path = os.path.join(STIM_PATH, "Question8.png")

    if not os.path.exists(img_path):
        print("Stimulus fehlt:", img_path)
        return

    img = Image.open(img_path)
    w, h = img.size

    aois = get_aois_q8()

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    for aoi in aois:

        # Highlight the target AOI in red
        if aoi["name"] == "month_feb":
            color = "#ff2d2d"
            lw = 0.9
        else:
            color = "#1f4aff" if aoi["type"] == "relevant" else "#999999"
            lw = 0.9

        x1 = aoi["x1"] * w
        y1 = aoi["y1"] * h
        width = (aoi["x2"] - aoi["x1"]) * w
        height = (aoi["y2"] - aoi["y1"]) * h

        rect = plt.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
            linestyle=(0, (3, 3))
        )

        plt.gca().add_patch(rect)

        # Position labels individually to keep the overlay readable
        if aoi["name"] == "month_rest":
           
            text_x = x1 + 3
            text_y = y1 + 8
            ha = "left"
            va = "top"

        elif aoi["name"] in ["month_jan", "price_50"]:
            text_x = x1 - 25
            text_y = y1 + height / 2
            ha = "right"
            va = "center"

        elif aoi["name"] == "month_feb":
            text_x = x1 + width / 2
            text_y = y1 + height + 8
            ha = "center"
            va = "top"

        elif aoi["name"] in ["question", "answers", "background"]:
            text_x = x1
            text_y = y1 - 20
            ha = "left"
            va = "top"

        else:
            # fallback
            text_x = x1 + 2
            text_y = y1 - 3
            ha = "left"
            va = "top"

        plt.text(
            text_x,
            text_y,
            aoi["name"],
            color=color,
            fontsize=4,
            alpha=0.7,
            ha=ha,
            va=va
        )

    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, "aoi_overlay_q8.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ AOI Overlay saved:", save_path)

# ============================================================
# PARTICIPANT ANALYSIS
# ============================================================

def run_analysis(participant):

    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results = []
    matrices = []

    for q_label in question_labels:

        qid = extract_question_id(q_label)
        if qid != 8:
            continue

        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        aois = get_aois_q8()
        fix = map_aois(fix, aois)

        print("\n==============================")
        print(f"AOI Counts für {participant}")
        print(fix["AOI"].value_counts())

        print("\nDwell Time pro AOI:")
        print(fix.groupby("AOI")["Gaze event duration"].sum())
        print("==============================\n")

        metrics = compute_metrics(fix, duration)

        matrix = metrics.pop("Transition_Matrix")
        matrix["Participant"] = participant
        matrix["Question"] = qid
        matrices.append(matrix)

        row = {"Participant": participant, "Question": qid}
        row.update(metrics)
        results.append(row)

    return pd.DataFrame(results), pd.concat(matrices)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    # Create AOI overlay for visual inspection
    plot_aoi_overlay()

    all_results = []
    all_matrices = []

    for p in PARTICIPANTS:
        df_res, df_mat = run_analysis(p)
        all_results.append(df_res)
        all_matrices.append(df_mat)

    pd.concat(all_results).to_csv(os.path.join(OUTPUT_DIR, "aoi_metrics_q8.csv"), index=False)
    pd.concat(all_matrices).to_csv(os.path.join(OUTPUT_DIR, "transition_matrix_q8.csv"), index=False)

    print("\n✔ Done: CSV files and AOI PNG saved")
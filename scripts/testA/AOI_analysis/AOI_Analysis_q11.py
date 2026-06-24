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

# Participants included in the analysis
PARTICIPANTS = ["Participant28"]

# ============================================================
# AOIs FOR QUESTION 11
# ============================================================

def get_aois_q11():
    return [
        {"name": "question", "x1": 0.30, "y1": 0.03, "x2": 0.70, "y2": 0.16, "type": "relevant"},

        {"name": "year_2012", "x1": 0.515, "x2": 0.545, "y1": 0.24, "y2": 0.60, "type": "relevant"},
        {"name": "Rest",      "x1": 0.415, "x2": 0.515, "y1": 0.22, "y2": 0.60, "type": "relevant"},
        {"name": "Rest1",     "x1": 0.545, "x2": 0.602, "y1": 0.22, "y2": 0.60, "type": "relevant"},

        {"name": "answers", "x1": 0.28, "y1": 0.67, "x2": 0.72, "y2": 0.95, "type": "relevant"},

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

    # Skip if no relevant timing event is available
    if len(url_events) < 1:
        return None, None

    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
    t_end = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

    
    if pd.isna(t_end) or (t_end - t_start) > 30000:
        t_end = t_start + 25000

    duration = (t_end - t_start) 

    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp"].between(t_start, t_end))
        ].copy()

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

    names, types = [], []

    for _, row in fix.iterrows():
        x = row["Fixation point X (MCSnorm)"]
        y = row["Fixation point Y (MCSnorm)"]

        assigned = False

        for aoi in aois_sorted:
            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                names.append(aoi["name"])
                types.append(aoi["type"])
                assigned = True
                break

        # Fallback if no AOI matches
        if not assigned:
            names.append("background")
            types.append("irrelevant")

    fix["AOI"] = names
    fix["AOI_type"] = types

    return fix

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fix, duration):
    """
    Compute AOI-based eye-tracking metrics for Question 11.

    The duration is expected to be in milliseconds.
    """

    # Sum dwell time per AOI
    dwell = fix.groupby("AOI")["Gaze event duration"].sum()
    total_dwell = dwell.sum()

     # Compute irrelevant dwell time before scaling
    irrelevant = fix[fix["AOI_type"] == "irrelevant"]["Gaze event duration"].sum()

    # Normalize dwell time if total fixation duration exceeds trial duration
    if total_dwell > duration and duration > 0:
        scale = duration / total_dwell

        dwell = dwell * scale
        irrelevant = irrelevant * scale  

        total_dwell = dwell.sum()

    dwell_ratio = {
        k: (v / total_dwell if total_dwell > 0 else 0)
        for k, v in dwell.items()
    }


    def ttff(target):
        subset = fix[fix["AOI"] == target]
        if subset.empty:
            return np.nan
        return (subset["Recording timestamp"].iloc[0] - fix["Recording timestamp"].iloc[0]) / 1000

    # Create AOI sequence and remove consecutive duplicates
    seq = fix["AOI"].tolist()

    seq_clean = [
        seq[i] for i in range(len(seq))
        if i == 0 or seq[i] != seq[i - 1]
    ]

    # Build transition matrix from cleaned scanpath sequence
    if len(seq_clean) >= 2:
        transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
        trans_df = pd.DataFrame(transitions, columns=["from", "to"])
        matrix = pd.crosstab(trans_df["from"], trans_df["to"])
    else:
        matrix = pd.DataFrame()

    transitions_count = max(len(seq_clean) - 1, 0)

    transitions_per_sec = (
        transitions_count / (duration / 1000)
        if duration > 0 else 0
    )

    # Background dwell ratio after normalization
    irrelevant_ratio = (
        irrelevant / total_dwell if total_dwell > 0 else 0
    )

    # =========================================================
    # OUTPUT
    # =========================================================
    return {
        "TTFF_2012": ttff("year_2012"),
        "TTFF_answers": ttff("answers"),

        "Dwell_2012": dwell.get("year_2012", 0),
        "Dwell_answers": dwell.get("answers", 0),

        "Dwell_ratio_2012": dwell_ratio.get("year_2012", 0),
        "Dwell_ratio_answers": dwell_ratio.get("answers", 0),

        "Transitions": transitions_count,
        "Transitions_per_sec": transitions_per_sec,
        "Sequence_length": len(seq_clean),

        "First_AOI": next((a for a in seq_clean if a != "background"), None),

        "Irrelevant_Ratio": irrelevant_ratio,

        "Transition_Matrix": matrix
    }

# ============================================================
# OVERLAY 
# ============================================================

def plot_aoi_overlay():

    img_path = os.path.join(STIM_PATH, "Question11.png")

    if not os.path.exists(img_path):
        print("Stimulus fehlt:", img_path)
        return

    img = Image.open(img_path)
    w, h = img.size

    aois = get_aois_q11()

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    for aoi in aois:

         # Highlight the target AOI in red
        color = "#ff2d2d" if aoi["name"] == "year_2012" else ("#1f4aff" if aoi["type"] == "relevant" else "#999999")

        x1 = aoi["x1"] * w
        y1 = aoi["y1"] * h
        width = (aoi["x2"] - aoi["x1"]) * w
        height = (aoi["y2"] - aoi["y1"]) * h

        rect = plt.Rectangle((x1, y1), width, height,
                             linewidth=1,
                             edgecolor=color,
                             facecolor="none",
                             linestyle=(0, (3, 3)))

        plt.gca().add_patch(rect)

    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, "aoi_overlay_q11.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ AOI Overlay gespeichert:", save_path)

# ============================================================
# PARTICIPANT ANALYSIS
# ============================================================

def run_analysis(participant):

    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results, matrices = [], []

    for q_label in question_labels:

        qid = extract_question_id(q_label)
        if qid != 11:
            continue

        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        fix = map_aois(fix, get_aois_q11())

        # Debug output to inspect AOI assignment and dwell times
        print("\n==============================")
        print("AOI Counts fuer", participant)
        print(fix["AOI"].value_counts())

        print("\nDwell Time pro AOI:")
        print(fix.groupby("AOI")["Gaze event duration"].sum())
        print("==============================\n")

        metrics = compute_metrics(fix, duration)

        matrix = metrics.pop("Transition_Matrix")
        if not matrix.empty:
            matrix["Participant"] = participant
            matrix["Question"] = qid
            matrices.append(matrix)

        row = {"Participant": participant, "Question": qid}
        row.update(metrics)
        results.append(row)

    return pd.DataFrame(results), pd.concat(matrices) if matrices else pd.DataFrame()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    # Create AOI overlay for visual inspection
    plot_aoi_overlay()

    all_results, all_matrices = [], []

    for p in PARTICIPANTS:
        df_res, df_mat = run_analysis(p)

        if not df_res.empty:
            all_results.append(df_res)

        if not df_mat.empty:
            all_matrices.append(df_mat)

    if all_results:
        pd.concat(all_results).to_csv(os.path.join(OUTPUT_DIR, "aoi_metrics_q11.csv"), index=False)

    if all_matrices:
        pd.concat(all_matrices).to_csv(os.path.join(OUTPUT_DIR, "transition_matrix_q11.csv"), index=False)

    print("\n✔ Final: Q11 analysis completed successfully")
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PATHS
# ============================================================

# Define input data, stimulus image, and output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testC")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testC")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of participants to analyze
PARTICIPANTS = ["Participant51"]

# ============================================================
# AOIs
# ============================================================

def get_aois_q1():
    return [
        {"name": "question", "x1": 0.26, "y1": 0.03, "x2": 0.74, "y2": 0.17, "type": "relevant"},
        {"name": "search_portal", "x1": 0.30, "y1": 0.18, "x2": 0.52, "y2": 0.53, "type": "relevant"},
        {"name": "software", "x1": 0.52, "y1": 0.18, "x2": 0.70, "y2": 0.33, "type": "relevant"},
        {"name": "retail", "x1": 0.52, "y1": 0.33, "x2": 0.70, "y2": 0.53, "type": "relevant"},
        {"name": "social_network", "x1": 0.30, "y1": 0.53, "x2": 0.52, "y2": 0.64, "type": "relevant"},
        {"name": "computer", "x1": 0.52, "y1": 0.53, "x2": 0.70, "y2": 0.64, "type": "relevant"},
        {"name": "answers", "x1": 0.28, "y1": 0.64, "x2": 0.72, "y2": 0.85, "type": "relevant"},
        {"name": "background", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "type": "irrelevant"},
    ]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_question_id(event_value):
    m = re.search(r"Question\s+(\d+)", str(event_value))
    return int(m.group(1)) if m else None


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {}
    if "Recording timestamp [ms]" in df.columns:
        rename_map["Recording timestamp [ms]"] = "Recording timestamp"
    if "Gaze event duration [ms]" in df.columns:
        rename_map["Gaze event duration [ms]"] = "Gaze event duration"
    if "Fixation point X [MCS norm]" in df.columns:
        rename_map["Fixation point X [MCS norm]"] = "Fixation point X (MCSnorm)"
    if "Fixation point Y [MCS norm]" in df.columns:
        rename_map["Fixation point Y [MCS norm]"] = "Fixation point Y (MCSnorm)"

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df


def get_fixations_for_question(df, question_label):

    # Find all URLStart events that match the current question label
    url_events = df[
        (df["Event"] == "URLStart") &
        (df["Event value"] == question_label)
        ]
    # If no URLStart event exists for this question, skip it
    if len(url_events) == 0:
        return None, None

    t_start_raw = url_events["Recording timestamp"].min()

    BUFFER_BEFORE = 2000
    MAX_DURATION = 25000

    t_start = t_start_raw - BUFFER_BEFORE
    t_end_real = t_start_raw + MAX_DURATION

    duration = (t_end_real - t_start)

    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp"].between(t_start, t_end_real))
        ].copy()

    fix = fix[
        (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
        (fix["Fixation point Y (MCSnorm)"].between(0, 1))
        ]

    if len(fix) == 0:
        return None, None
    # Sort fixations chronologically
    return fix.sort_values("Recording timestamp").reset_index(drop=True), duration

# ============================================================
# AOI MAPPING
# ============================================================

def map_aois(fix, aois):

    # Check specific AOIs before the background AOI
    aois_sorted = sorted(
        aois,
        key=lambda a: (a["name"] == "background", (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    )

    aoi_names = []
    aoi_types = []

    # Assign each fixation to an AOI
    for _, row in fix.iterrows():
        x = row["Fixation point X (MCSnorm)"]
        y = row["Fixation point Y (MCSnorm)"]

        assigned = False
        # Assign each fixation to an AOI
        for aoi in aois_sorted:
            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                aoi_names.append(aoi["name"])
                aoi_types.append(aoi["type"])
                assigned = True
                break

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

    if fix is None or len(fix) == 0:
        return None

    # Sum dwell time per AOI
    dwell = fix.groupby("AOI")["Gaze event duration"].sum()
    total_dwell = dwell.sum()

    if total_dwell > 0 and total_dwell > duration:
        scale = duration / total_dwell
        dwell = dwell * scale
        total_dwell = dwell.sum()

    # Calculate dwell ratios
    dwell_ratio = {k: v / total_dwell for k, v in dwell.items()} if total_dwell > 0 else {}

    def ttff(target):

        # Time to first fixation for one AOI
        subset = fix[fix["AOI"] == target]
        if len(subset) == 0:
            return np.nan
        return (subset["Recording timestamp"].iloc[0] - fix["Recording timestamp"].iloc[0])

    # Create AOI sequence
    seq = fix["AOI"].tolist()

    # Remove repeated consecutive AOIs
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i - 1]]

    # Build transition matrix
    transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
    trans_df = pd.DataFrame(transitions, columns=["from", "to"])
    matrix = pd.crosstab(trans_df["from"], trans_df["to"]) if len(trans_df) > 0 else pd.DataFrame()

    # Transitions normalized by duration
    transitions_per_sec = len(seq_clean) / (duration / 1000) if duration > 0 else 0

    # Irrelevant dwell time ratio
    irrelevant = dwell.get("background", 0)
    irrelevant_ratio = irrelevant / total_dwell if total_dwell > 0 else 0

    return {
        "TTFF_search_portal": ttff("search_portal"),
        "TTFF_retail": ttff("retail"),
        "TTFF_answers": ttff("answers"),

        "Dwell_search": dwell.get("search_portal", 0),
        "Dwell_retail": dwell.get("retail", 0),
        "Dwell_answers": dwell.get("answers", 0),

        "Dwell_ratio_search": dwell_ratio.get("search_portal", 0),
        "Dwell_ratio_retail": dwell_ratio.get("retail", 0),

        "Transitions": len(seq_clean),
        "Transitions_per_sec": transitions_per_sec,
        "Sequence_length": len(seq_clean),
        "First_AOI": next((a for a in seq_clean if a != "background"), None),

        "Irrelevant_Ratio": irrelevant_ratio,

        "Transition_Matrix": matrix
    }

# ============================================================
# AOI OVERLAY PLOT
# ============================================================

def plot_aoi_overlay():
    img_path = os.path.join(STIM_PATH, "Question1.png")

    # Stop if stimulus image is missing
    if not os.path.exists(img_path):
        print("Stimulus fehlt:", img_path)
        return

    img = Image.open(img_path)
    w, h = img.size

    aois = get_aois_q1()

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    # Draw all AOI rectangles
    for aoi in aois:
        x1 = aoi["x1"] * w
        y1 = aoi["y1"] * h
        width = (aoi["x2"] - aoi["x1"]) * w
        height = (aoi["y2"] - aoi["y1"]) * h

        rect = plt.Rectangle(
            (x1, y1), width, height,
            linewidth=1.5,
            edgecolor="dodgerblue" if aoi["type"] == "relevant" else "gray",
            facecolor="none"
        )

        plt.gca().add_patch(rect)
        # Add AOI label
        plt.text(x1 + 5, y1 + 15, aoi["name"], fontsize=6)

    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, "aoi_overlay_q1.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ AOI PNG gespeichert:", save_path)

# ============================================================
# MAIN
# ============================================================

def run_analysis(participant):

    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    df = normalize_columns(df)

    # Get all question labels
    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results = []
    matrices = []

    for q_label in question_labels:

        # Analyze only Question 1
        qid = extract_question_id(q_label)
        if qid != 1:
            continue

        # Extract fixations for this question
        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        # Map fixations to AOIs
        aois = get_aois_q1()
        fix = map_aois(fix, aois)

        # ============================================================
        #  DEBUG OUTPUT
        # ============================================================
        print("\n===== DEBUG TIME CHECK =====")
        print("Fixations count:", len(fix))
        print("Total fixation time (ms):", fix["Gaze event duration"].sum())
        print("Task duration (ms):", duration)
        print("============================\n")

        metrics = compute_metrics(fix, duration)
        if metrics is None:
            continue

        matrix = metrics.pop("Transition_Matrix")

        if not matrix.empty:
            matrix["Participant"] = participant
            matrix["Question"] = qid
            matrices.append(matrix)

        row = {"Participant": participant, "Question": qid}
        row.update(metrics)
        results.append(row)

    if len(results) == 0:
        return pd.DataFrame(), pd.DataFrame()

    return pd.DataFrame(results), pd.concat(matrices) if len(matrices) > 0 else pd.DataFrame()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    # Create AOI overlay image
    plot_aoi_overlay()

    all_results = []
    all_matrices = []

    # Run analysis for all participants
    for p in PARTICIPANTS:
        df_res, df_mat = run_analysis(p)
        all_results.append(df_res)
        all_matrices.append(df_mat)

    # Save metric results
    pd.concat(all_results).to_csv(os.path.join(OUTPUT_DIR, "aoi_metrics_q1.csv"), index=False)

    # Save transition matrices
    pd.concat(all_matrices).to_csv(os.path.join(OUTPUT_DIR, "transition_matrix_q1.csv"), index=False)

    print("\n✔ Analysis complete: CSV + AOI PNG")
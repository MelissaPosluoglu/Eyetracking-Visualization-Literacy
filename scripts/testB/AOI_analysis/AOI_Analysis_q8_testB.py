import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testB")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testB","Participant41","AOI","q8")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTICIPANTS = ["Participant41"]

# ============================================================
# AOIs (Q8 FINAL )
# ============================================================

def get_aois_q8():
    # AOI coordinates are normalized between 0 and 1
    return [

        # QUESTION
        {"name": "question", "x1": 0.31, "y1": 0.03, "x2": 0.68, "y2": 0.165, "type": "relevant"},

        # Price label around 50
        {"name": "price_50", "x1": 0.395, "y1": 0.23, "x2": 0.414, "y2": 0.27, "type": "relevant"},

        # January area in the plot
        {"name": "month_jan", "x1": 0.415, "y1": 0.17, "x2": 0.425, "y2": 0.6, "type": "relevant"},

        # February area, main target AOI
        {"name": "month_feb", "x1": 0.427, "y1": 0.17, "x2": 0.44, "y2": 0.6, "type": "relevant"},

        # Remaining months in the plot
        {"name": "month_rest", "x1": 0.445, "y1": 0.17, "x2": 0.65, "y2": 0.6, "type": "relevant"},

        # Answer options area
        {"name": "answers", "x1": 0.28, "y1": 0.64, "x2": 0.72, "y2": 0.92, "type": "relevant"},

        # Full-screen fallback AOI
        {"name": "background", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "type": "irrelevant"},
    ]

# ============================================================
# HELPERS
# ============================================================

def extract_question_id(event_value):
    # Extract question number from event label
    m = re.search(r"Question\s+(\d+)", str(event_value))
    return int(m.group(1)) if m else None


def get_fixations_for_question(df, question_label):
    # Get URL start and end events for this question
    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]

    if len(url_events) < 2:
        return None, None

    # Define question time window
    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp [ms]"].min()
    t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp [ms]"].max()

    # Convert duration to seconds
    duration = (t_end - t_start) / 1000

    # Select fixation data within this time window
    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp [ms]"].between(t_start, t_end))
        ].copy()

    # Keep only valid normalized coordinates
    fix = fix[
        (fix["Fixation point X [MCS norm]"].between(0, 1)) &
        (fix["Fixation point Y [MCS norm]"].between(0, 1))
        ]

    # Sort fixations by timestamp
    return fix.sort_values("Recording timestamp [ms]").reset_index(drop=True), duration

# ============================================================
# AOI MAPPING
# ============================================================

def map_aois(fix, aois):

    # Check smaller AOIs before the background AOI
    aois_sorted = sorted(
        aois,
        key=lambda a: (a["name"] == "background", (a["x2"]-a["x1"]) * (a["y2"]-a["y1"]))
    )

    aoi_names = []
    aoi_types = []

    # Assign each fixation to one AOI
    for _, row in fix.iterrows():
        x = row["Fixation point X [MCS norm]"]
        y = row["Fixation point Y [MCS norm]"]

        for aoi in aois_sorted:
            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                aoi_names.append(aoi["name"])
                aoi_types.append(aoi["type"])
                break

    # Add AOI information to fixation data
    fix["AOI"] = aoi_names
    fix["AOI_type"] = aoi_types

    return fix

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fix, duration):
    # Sum dwell time per AOI
    dwell = fix.groupby("AOI")["Gaze event duration [ms]"].sum()
    total_dwell = dwell.sum()

    # Calculate dwell ratios
    dwell_ratio = {k: v / total_dwell for k, v in dwell.items()}

    def ttff(target):
        # Time to first fixation for one AOI
        subset = fix[fix["AOI"] == target]
        if len(subset) == 0:
            return np.nan
        return (subset["Recording timestamp [ms]"].iloc[0] - fix["Recording timestamp [ms]"].iloc[0]) / 1000

    # Create AOI sequence
    seq = fix["AOI"].tolist()
    # Remove consecutive duplicate AOIs
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i-1]]

    # Build transition matrix
    transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
    trans_df = pd.DataFrame(transitions, columns=["from", "to"])
    matrix = pd.crosstab(trans_df["from"], trans_df["to"])


    # Calculate transitions per second
    transitions_per_sec = len(seq_clean) / duration if duration > 0 else 0

    # Calculate irrelevant dwell ratio
    irrelevant = fix[fix["AOI_type"] == "irrelevant"]["Gaze event duration [ms]"].sum()
    irrelevant_ratio = irrelevant / total_dwell if total_dwell > 0 else 0

    return {
        # Time to first fixation
        "TTFF_feb": ttff("month_feb"),
        "TTFF_answers": ttff("answers"),

        # Dwell times
        "Dwell_feb": dwell.get("month_feb", 0),
        "Dwell_answers": dwell.get("answers", 0),

        # Dwell ratio
        "Dwell_ratio_feb": dwell_ratio.get("month_feb", 0),

        # Sequence metrics
        "Transitions": len(seq_clean),
        "Transitions_per_sec": transitions_per_sec,
        "Sequence_length": len(seq_clean),
        "First_AOI": next((a for a in seq_clean if a != "background"), None),

        # Irrelevant gaze metric
        "Irrelevant_Ratio": irrelevant_ratio,

        # Transition matrix
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

    # Draw all AOI rectangles
    for aoi in aois:
        # Highlight February AOI
        if aoi["name"] == "month_feb":
            color = "#ff2d2d"
            lw = 0.9
        else:
            color = "#1f4aff" if aoi["type"] == "relevant" else "#999999"
            lw = 0.9

        # Convert normalized coordinates to pixels
        x1 = aoi["x1"] * w
        y1 = aoi["y1"] * h
        width = (aoi["x2"] - aoi["x1"]) * w
        height = (aoi["y2"] - aoi["y1"]) * h

        # Draw AOI rectangle
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

        # Set label position for each AOI
        if aoi["name"] == "month_rest":
            # 👉 oben links IM KASTEN
            text_x = x1 + 3
            text_y = y1 + 8
            ha = "left"
            va = "top"

        elif aoi["name"] in ["month_jan", "price_50"]:
            # 👉 deutlich weiter links außerhalb
            text_x = x1 - 25
            text_y = y1 + height / 2
            ha = "right"
            va = "center"

        elif aoi["name"] == "month_feb":
            # 👉 UNTER dem Kasten
            text_x = x1 + width / 2
            text_y = y1 + height + 8
            ha = "center"
            va = "top"

        elif aoi["name"] in ["question", "answers", "background"]:
            # 👉 links oberhalb des Kastens
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

    print("✔ AOI Overlay gespeichert:", save_path)

# ============================================================
# MAIN
# ============================================================

def run_analysis(participant):

    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)


    # Get all question labels
    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results = []
    matrices = []

    for q_label in question_labels:
        # Analyze only Question 8
        qid = extract_question_id(q_label)
        if qid != 8:
            continue

        # Extract fixations for this question
        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        # Map fixations to AOIs
        aois = get_aois_q8()
        fix = map_aois(fix, aois)

        # Print debug information
        print("\n==============================")
        print(f"AOI Counts für {participant}")
        print(fix["AOI"].value_counts())

        print("\nDwell Time pro AOI:")
        print(fix.groupby("AOI")["Gaze event duration [ms]"].sum())
        print("==============================\n")

        # Calculate metrics
        metrics = compute_metrics(fix, duration)

        # Store transition matrix separately
        matrix = metrics.pop("Transition_Matrix")
        matrix["Participant"] = participant
        matrix["Question"] = qid
        matrices.append(matrix)

        # Store metric row
        row = {"Participant": participant, "Question": qid}
        row.update(metrics)
        results.append(row)

    return pd.DataFrame(results), pd.concat(matrices)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    plot_aoi_overlay()

    all_results = []
    all_matrices = []

    # Run analysis for all participants
    for p in PARTICIPANTS:
        df_res, df_mat = run_analysis(p)
        all_results.append(df_res)
        all_matrices.append(df_mat)

    # Save metric results
    pd.concat(all_results).to_csv(os.path.join(OUTPUT_DIR, "aoi_metrics_q8.csv"), index=False)

    # Save transition matrix
    pd.concat(all_matrices).to_csv(os.path.join(OUTPUT_DIR, "transition_matrix_q8.csv"), index=False)

    print("\n✔ Alles fertig: CSV + AOI PNG")
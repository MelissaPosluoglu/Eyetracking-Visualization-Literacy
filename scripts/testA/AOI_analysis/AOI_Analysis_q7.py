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
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")
RESULTS_BASE = os.path.join(BASE_DIR, "results", "testA")

PARTICIPANTS = ["Participant20"]

def get_output_dir(participant):
    participant_dir = os.path.join(RESULTS_BASE, participant.lower())
    output_dir = os.path.join(participant_dir, "AOI", "q7_stackedbar")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
# ============================================================
# AOIs (Q7 FINAL - STACKED BAR)
# ============================================================

def get_aois_q7():
    return [
        {"name": "question",        "x1": 0.27,  "y1": 0.01,  "x2": 0.7,  "y2": 0.165,  "type": "relevant"},

        # gesamter Diagrammbereich
        #{"name": "chart",           "x1": 0.35,  "y1": 0.18,  "x2": 0.58,  "y2": 0.60,  "type": "relevant"},

        # y-Achse links vom Plot
        {"name": "y_axis", "x1": 0.37, "y1": 0.18, "x2": 0.4, "y2": 0.6,"type": "relevant"},



        {"name": "city_right",      "x1": 0.456, "y1": 0.18,  "x2": 0.58, "y2": 0.6,  "type": "relevant"},
        {"name": "city_left",       "x1": 0.4, "y1": 0.18,  "x2": 0.430, "y2": 0.6,  "type": "relevant"},
        {"name": "seoul",           "x1": 0.432, "y1": 0.18,  "x2": 0.455, "y2": 0.6,  "type": "relevant"},
        #{"name": "peanut",          "x1": 0.430, "y1":0.28,  "x2": 0.455, "y2": 0.38,  "type": "relevant"},




        # Legende rechts neben dem Chart
        {"name": "legend",          "x1": 0.585,  "y1": 0.2,  "x2": 0.65,  "y2": 0.35,  "type": "relevant"},

        {
            "name": "answers",
            "x1": 0.314,
            "y1": 0.64,
            "x2": 0.685,
            "y2": 0.94,
            "type": "relevant"
        },

        {"name": "background",      "x1": 0.0,   "y1": 0.0,   "x2": 1.0,   "y2": 1.0,   "type": "irrelevant"},
    ]
# ============================================================
# HELPERS
# ============================================================

def extract_question_id(event_value):
    m = re.search(r"Question\s+(\d+)", str(event_value))
    return int(m.group(1)) if m else None


def get_fixations_for_question(df, question_label):
    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]

    if len(url_events) < 2:
        return None, None

    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
    t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

    duration = (t_end - t_start) / 1000

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
        key=lambda a: (a["name"] == "background", (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
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
    dwell = fix.groupby("AOI")["Gaze event duration"].sum()
    total_dwell = dwell.sum()

    dwell_ratio = {k: v / total_dwell for k, v in dwell.items()} if total_dwell > 0 else {}

    def ttff(target):
        subset = fix[fix["AOI"] == target]
        if len(subset) == 0:
            return np.nan
        return (subset["Recording timestamp"].iloc[0] - fix["Recording timestamp"].iloc[0]) / 1000

    seq = fix["AOI"].tolist()
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i - 1]]

    if len(seq_clean) >= 2:
        transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
        trans_df = pd.DataFrame(transitions, columns=["from", "to"])
        matrix = pd.crosstab(trans_df["from"], trans_df["to"])
    else:
        matrix = pd.DataFrame()

    irrelevant = fix[fix["AOI_type"] == "irrelevant"]["Gaze event duration"].sum()
    irrelevant_ratio = irrelevant / total_dwell if total_dwell > 0 else 0

    return {
        "TTFF_seoul": ttff("seoul"),
        "TTFF_y_axis": ttff("y_axis"),
        "TTFF_legend": ttff("legend"),
        "TTFF_answers": ttff("answers"),
        "TTFF_city_left": ttff("city_left"),
        "TTFF_city_right": ttff("city_right"),
        "TTFF_question": ttff("question"),

        "Dwell_seoul": dwell.get("seoul", 0),
        "Dwell_y_axis": dwell.get("y_axis", 0),
        "Dwell_legend": dwell.get("legend", 0),
        "Dwell_answers": dwell.get("answers", 0),
        "Dwell_city_left": dwell.get("city_left", 0),
        "Dwell_city_right": dwell.get("city_right", 0),
        "Dwell_question": dwell.get("question", 0),

        "Dwell_ratio_seoul": dwell_ratio.get("seoul", 0),
        "Dwell_ratio_y_axis": dwell_ratio.get("y_axis", 0),
        "Dwell_ratio_legend": dwell_ratio.get("legend", 0),
        "Dwell_ratio_answers": dwell_ratio.get("answers", 0),
        "Dwell_ratio_city_left": dwell_ratio.get("city_left", 0),
        "Dwell_ratio_city_right": dwell_ratio.get("city_right", 0),
        "Dwell_ratio_question": dwell_ratio.get("question", 0),

        "Transitions": max(len(seq_clean) - 1, 0),
        "Transitions_per_sec": max(len(seq_clean) - 1, 0) / duration if duration > 0 else 0,
        "Sequence_length": len(seq_clean),
        "First_AOI": next((a for a in seq_clean if a != "background"), None),

        "Irrelevant_Ratio": irrelevant_ratio,
        "Transition_Matrix": matrix
    }

# ============================================================
# AOI OVERLAY PNG
# ============================================================
def plot_aoi_overlay(output_dir):

    img_path = os.path.join(STIM_PATH, "Question7.png")

    if not os.path.exists(img_path):
        print("Stimulus fehlt:", img_path)
        return

    img = Image.open(img_path)
    w, h = img.size

    aois = get_aois_q7()

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    for aoi in aois:

        if aoi["name"] == "seoul":
            color = "#ff2d2d"   # rot highlight wie month_feb
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

        # Labels
        if aoi["name"] == "question":
            tx = x1 + 6
            ty = y1 + 12
            ha = "left"
            va = "top"

        elif aoi["name"] == "chart":
            tx = x1 + 4
            ty = y1 + height - 4
            ha = "left"
            va = "bottom"

        elif aoi["name"] == "y_axis":
            tx = x1 - 6
            ty = y1 + height / 2
            ha = "right"
            va = "center"

        elif aoi["name"] == "seoul":
            tx = x1 + width / 2
            ty = y1 + 27
            ha = "center"
            va = "bottom"

        elif aoi["name"] == "city_left":
            tx = x1 - 6
            ty = y1 + 12
            ha = "right"
            va = "top"

        elif aoi["name"] == "city_right":
            tx = x1 + width - 6
            ty = y1 + 12
            ha = "right"
            va = "top"

        elif aoi["name"] == "legend":
            tx = x1 + 12
            ty = y1 + 10
            ha = "left"
            va = "top"

        elif aoi["name"] == "answers":
            tx = x1 + 6
            ty = y1 + 12
            ha = "left"
            va = "top"

        else:
            tx = x1 + 6
            ty = y1 + 12
            ha = "left"
            va = "top"

        plt.text(
            tx,
            ty,
            aoi["name"],
            fontsize=4,
            alpha=0.7,
            color=color,
            ha=ha,
            va=va
        )

    plt.axis("off")

    save_path = os.path.join(output_dir, "aoi_overlay_q7.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ AOI PNG gespeichert:", save_path)
# ============================================================
# MAIN
# ============================================================

def run_analysis(participant):
    output_dir = get_output_dir(participant)

    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results = []
    matrices = []

    for q_label in question_labels:
        qid = extract_question_id(q_label)
        if qid != 7:
            continue

        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        aois = get_aois_q7()
        fix = map_aois(fix, aois)

        print("\n==============================")
        print(f"AOI Counts für {participant}")
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

    results_df = pd.DataFrame(results)
    matrices_df = pd.concat(matrices) if len(matrices) > 0 else pd.DataFrame()

    results_df.to_csv(os.path.join(output_dir, "aoi_metrics_q7.csv"), index=False)

    if not matrices_df.empty:
        matrices_df.to_csv(os.path.join(output_dir, "transition_matrix_q7.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(output_dir, "transition_matrix_q7.csv"), index=False)

    return results_df, matrices_df

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    all_results = []
    all_matrices = []

    for p in PARTICIPANTS:
        output_dir = get_output_dir(p)
        plot_aoi_overlay(output_dir)

        df_res, df_mat = run_analysis(p)
        all_results.append(df_res)

        if not df_mat.empty:
            all_matrices.append(df_mat)

    print("\n✔ Alles fertig: AOI PNG + CSVs in jeweiligem participant/AOI/q7_stackedbar")
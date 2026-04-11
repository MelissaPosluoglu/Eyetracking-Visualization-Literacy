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
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testA")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTICIPANTS = ["Participant27"]

# ============================================================
# AOIs (Q1 FINAL)
# ============================================================

def get_aois_q1():
    return [
        {"name": "question", "x1": 0.26, "y1": 0.03, "x2": 0.74, "y2": 0.17, "type": "relevant"},

        {"name": "search_portal", "x1": 0.30, "y1": 0.18, "x2": 0.52, "y2": 0.53, "type": "relevant"},
        {"name": "software",      "x1": 0.52, "y1": 0.18, "x2": 0.70, "y2": 0.33, "type": "relevant"},
        {"name": "retail",        "x1": 0.52, "y1": 0.33, "x2": 0.70, "y2": 0.53, "type": "relevant"},
        {"name": "social_network","x1": 0.30, "y1": 0.53, "x2": 0.52, "y2": 0.64, "type": "relevant"},
        {"name": "computer",      "x1": 0.52, "y1": 0.53, "x2": 0.70, "y2": 0.64, "type": "relevant"},

        {"name": "answers", "x1": 0.28, "y1": 0.64, "x2": 0.72, "y2": 0.85, "type": "relevant"},

        {"name": "background", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "type": "irrelevant"},
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

    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp [ms]"].min()
    t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp [ms]"].max()

    duration = (t_end - t_start) / 1000

    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp [ms]"].between(t_start, t_end))
        ].copy()

    fix = fix[
        (fix["Fixation point X [MCS norm]"].between(0, 1)) &
        (fix["Fixation point Y [MCS norm]"].between(0, 1))
        ]

    return fix.sort_values("Recording timestamp [ms]").reset_index(drop=True), duration

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
        x = row["Fixation point X [MCS norm]"]
        y = row["Fixation point Y [MCS norm]"]

        for aoi in aois_sorted:
            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                aoi_names.append(aoi["name"])
                aoi_types.append(aoi["type"])
                break

    fix["AOI"] = aoi_names
    fix["AOI_type"] = aoi_types

    return fix

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fix, duration):

    dwell = fix.groupby("AOI")["Gaze event duration [ms]"].sum()
    total_dwell = dwell.sum()

    dwell_ratio = {k: v / total_dwell for k, v in dwell.items()}

    def ttff(target):
        subset = fix[fix["AOI"] == target]
        if len(subset) == 0:
            return np.nan
        return (subset["Recording timestamp [ms]"].iloc[0] - fix["Recording timestamp [ms]"].iloc[0]) / 1000

    seq = fix["AOI"].tolist()
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i-1]]

    transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
    trans_df = pd.DataFrame(transitions, columns=["from", "to"])
    matrix = pd.crosstab(trans_df["from"], trans_df["to"])

    transitions_per_sec = len(seq_clean) / duration if duration > 0 else 0

    irrelevant = fix[fix["AOI_type"] == "irrelevant"]["Gaze event duration [ms]"].sum()
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
# AOI OVERLAY PNG
# ============================================================

def plot_aoi_overlay():

    img_path = os.path.join(STIM_PATH, "Question1.png")

    if not os.path.exists(img_path):
        print("Stimulus fehlt:", img_path)
        return

    img = Image.open(img_path)
    w, h = img.size

    aois = get_aois_q1()

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    for aoi in aois:

        x1 = aoi["x1"] * w
        y1 = aoi["y1"] * h
        width = (aoi["x2"] - aoi["x1"]) * w
        height = (aoi["y2"] - aoi["y1"]) * h

        color = "dodgerblue" if aoi["type"] == "relevant" else "gray"
        alpha = 0.15 if aoi["type"] == "relevant" else 0.05

        rect = plt.Rectangle((x1, y1), width, height,
                             linewidth=1.5,
                             edgecolor=color,
                             facecolor=color,
                             alpha=alpha)

        plt.gca().add_patch(rect)

        # saubere kleine Labels oben links
        plt.text(
            x1 + 5,
            y1 + 15,
            aoi["name"],
            fontsize=6,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
        )

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

    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results = []
    matrices = []

    for q_label in question_labels:

        qid = extract_question_id(q_label)
        if qid != 1:
            continue

        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        aois = get_aois_q1()
        fix = map_aois(fix, aois)

        # ============================================================
        # 🔥 DEBUG OUTPUT (NEU)
        # ============================================================
        print("\n==============================")
        print(f"AOI Counts für {participant}")
        print(fix["AOI"].value_counts())

        print("\nDwell Time pro AOI:")
        print(fix.groupby("AOI")["Gaze event duration [ms]"].sum())
        print("==============================\n")

        # ============================================================

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

    # 👉 PNG erzeugen
    plot_aoi_overlay()

    all_results = []
    all_matrices = []

    for p in PARTICIPANTS:
        df_res, df_mat = run_analysis(p)
        all_results.append(df_res)
        all_matrices.append(df_mat)

    pd.concat(all_results).to_csv(os.path.join(OUTPUT_DIR, "aoi_metrics_q1.csv"), index=False)
    pd.concat(all_matrices).to_csv(os.path.join(OUTPUT_DIR, "transition_matrix_q1.csv"), index=False)

    print("\n✔ Alles fertig: CSV + AOI PNG")
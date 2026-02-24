import os
import glob
import pandas as pd

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import styles
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import inch


# ---------------------------------------------------
# 1️⃣ Alle TSV-Dateien finden
# ---------------------------------------------------

folder_path = "../../data/testA/"
files = glob.glob(os.path.join(folder_path, "*.tsv"))
# Scores laden
scores_df = pd.read_csv("../../data/testA//scores.csv")
answers_df = pd.read_csv("../../data/testA/answers.csv")


all_results = []

print("Found participants:", len(files))


# ---------------------------------------------------
# 2️⃣ Jede Datei verarbeiten
# ---------------------------------------------------

for file in files:

    participant_name = os.path.basename(file).replace(".tsv", "")
    df = pd.read_csv(file, sep="\t", low_memory=False)

    # Question Events
    question_events = df[
        (df["Event"] == "URLStart") &
        (df["Event value"].str.contains("Question", na=False))
        ].copy()

    question_events = question_events.sort_values("Recording timestamp")

    if question_events.empty:
        continue

    # Fixationen vorbereiten
    fix = df[df["Eye movement type"] == "Fixation"].copy()

    fix = fix[
        (fix["Gaze event duration"] >= 80) &
        (fix["Gaze event duration"] <= 1000)
        ]

    first_question_time = question_events.iloc[0]["Recording timestamp"]
    fix = fix[fix["Recording timestamp"] >= first_question_time]

    # Segmentierung
    for i in range(len(question_events)):

        start_time = question_events.iloc[i]["Recording timestamp"]

        if i < len(question_events) - 1:
            end_time = question_events.iloc[i+1]["Recording timestamp"]
        else:
            end_time = df["Recording timestamp"].max()

        question_name = question_events.iloc[i]["Event value"]

        question_fix = fix[
            (fix["Recording timestamp"] >= start_time) &
            (fix["Recording timestamp"] < end_time)
            ]

        unique_fix = question_fix.drop_duplicates(
            subset="Eye movement type index"
        )

        fixation_count = unique_fix["Eye movement type index"].nunique()
        mean_fix = unique_fix["Gaze event duration"].mean()
        total_dwell = unique_fix["Gaze event duration"].sum()

        all_results.append([
            participant_name,
            question_name.split(" (")[0],
            fixation_count,
            round(mean_fix, 2) if pd.notna(mean_fix) else 0,
            total_dwell
        ])


# ---------------------------------------------------
# 3️⃣ Strukturierte PDF erstellen
# ---------------------------------------------------

from reportlab.platypus import PageBreak
from reportlab.lib.pagesizes import A4

pdf_file = "All_Participants_Report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=A4)
elements = []

styles = getSampleStyleSheet()
title_style = styles["Heading1"]
participant_style = styles["Heading2"]

# Gruppiere Ergebnisse nach Participant
result_df = pd.DataFrame(all_results, columns=[
    "Participant",
    "Question",
    "Fix_Count",
    "Mean_Fix_ms",
    "Total_Dwell_ms"
])
result_df["Visualization"] = result_df["Question"].str.replace(
    r"Question \d+ – ", "", regex=True
)

participants = result_df["Participant"].unique()

elements.append(Paragraph("Eye Tracking Report", title_style))
elements.append(Spacer(1, 0.4 * inch))


for p in participants:
    # Score holen
    participant_answers = answers_df[
        answers_df["Participant"] == p
        ]

    score_row = scores_df[scores_df["Participant"] == p]

    if not score_row.empty:
        participant_score = int(score_row["Score"].values[0])
    else:
        participant_score = "N/A"

    # Total Duration holen (aus answers.csv)
    duration_row = answers_df[answers_df["Participant"] == p]

    if not duration_row.empty and "TotalDuration" in answers_df.columns:
        participant_duration = duration_row["TotalDuration"].iloc[0]
    else:
        participant_duration = "N/A"


    participant_data = result_df[result_df["Participant"] == p]

    elements.append(Paragraph(f"Participant: {p}", participant_style))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(
        Paragraph(f"<b>Total Score:</b> {participant_score}", styles["Normal"])
    )
    elements.append(
        Paragraph(f"<b>Total Duration:</b> {participant_duration} s", styles["Normal"])
    )

    elements.append(Spacer(1, 0.3 * inch))


    table_data = [
        ["Question", "Fix Count", "Mean Fix (ms)", "Total Dwell (ms)"]
    ]

    for _, row in participant_data.iterrows():
        table_data.append([
            row["Question"],
            row["Fix_Count"],
            row["Mean_Fix_ms"],
            row["Total_Dwell_ms"]
        ])

    table = Table(
        table_data,
        colWidths=[150, 80, 100, 120],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # -----------------------------
    # Answer Overview (PRO Participant!)
    # -----------------------------
    elements.append(Paragraph("Answer Overview", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))

    answer_table_data = [
        ["Question", "Seconds", "Correct"]
    ]

    for _, row in participant_answers.iterrows():
        result_symbol = "✓" if row["Correct"] == 1 else "✗"

        answer_table_data.append([
            row["Question"],
            row["Seconds"],
            result_symbol
        ])

    answer_table = Table(
        answer_table_data,
        colWidths=[120, 80, 60],
        repeatRows=1
    )

    answer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))

    elements.append(answer_table)
    elements.append(PageBreak())


doc.build(elements)

print("Structured PDF created:", pdf_file)

# ---------------------------------------------------
# 4️⃣ Repeated Measures ANOVA
# ---------------------------------------------------

import pingouin as pg

# Sicherstellen, dass jede Person jede Visualisierung hat
check = result_df.groupby(["Participant", "Visualization"]).size()
print("\nCheck counts per cell:\n", check)

# ANOVA berechnen
anova = pg.rm_anova(
    data=result_df,
    dv="Mean_Fix_ms",
    within="Visualization",
    subject="Participant",
    detailed=True
)

print("\nRepeated Measures ANOVA Result:\n")
print(anova)

# Optional: Post-hoc Tests
posthoc = pg.pairwise_tests(
    data=result_df,
    dv="Mean_Fix_ms",
    within="Visualization",
    subject="Participant",
    padjust="bonf"
)

print("\nPost-hoc comparisons:\n")
print(posthoc)
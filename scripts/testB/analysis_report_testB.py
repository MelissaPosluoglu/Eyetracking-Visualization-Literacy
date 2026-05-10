import os
import glob
import pandas as pd

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# ---------------------------------------------------
# KONSTANTEN (verhindert Fehler!)
# ---------------------------------------------------

TIMESTAMP = "Recording timestamp [ms]"
DURATION = "Gaze event duration [ms]"
INDEX = "Eye movement type index"

# ---------------------------------------------------
# 1️⃣ Dateien laden
# ---------------------------------------------------

folder_path = "../../data/testB/"
files = glob.glob(os.path.join(folder_path, "*.tsv"))

scores_df = pd.read_csv("../../data/testB/scores.csv")
answers_df = pd.read_csv(
    "../../data/testB/answers.csv",
    sep=",",              # oder ";" wenn nötig
    on_bad_lines="skip"
)

print(answers_df.columns.tolist())

all_results = []

print("Found participants:", len(files))

# ---------------------------------------------------
# 2️⃣ Verarbeitung
# ---------------------------------------------------

for file in files:

    participant_name = os.path.basename(file).replace(".tsv", "")
    df = pd.read_csv(file, sep="\t", low_memory=False)

    # -------------------------------
    # Timestamp finden
    # -------------------------------
    timestamp_col = None
    for col in df.columns:
        if "recording" in col.lower() and "timestamp" in col.lower():
            timestamp_col = col
            break

    if timestamp_col is None:
        print(f"⚠️ Keine Timestamp-Spalte in {file}")
        continue

    df = df.rename(columns={timestamp_col: TIMESTAMP})

    # -------------------------------
    # Duration finden
    # -------------------------------
    duration_col = None
    for col in df.columns:
        if "duration" in col.lower() and ("gaze" in col.lower() or "fixation" in col.lower()):
            duration_col = col
            break

    if duration_col is None:
        print(f"⚠️ Keine Duration-Spalte in {file}")
        continue

    df = df.rename(columns={duration_col: DURATION})

    # -------------------------------
    # Index finden
    # -------------------------------
    index_col = None
    for col in df.columns:
        if "index" in col.lower() and "movement" in col.lower():
            index_col = col
            break

    if index_col is None:
        print(f"⚠️ Keine Index-Spalte in {file}")
        continue

    df = df.rename(columns={index_col: INDEX})

    # -------------------------------
    # Question Events
    # -------------------------------
    question_events = df[
        (df["Event"] == "URLStart") &
        (df["Event value"].astype(str).str.contains("Question", na=False))
        ].copy()

    if question_events.empty:
        continue

    question_events = question_events.sort_values(TIMESTAMP)

    # -------------------------------
    # Fixationen
    # -------------------------------
    fix = df[df["Eye movement type"] == "Fixation"].copy()

    fix = fix[
        (fix[DURATION] >= 80) &
        (fix[DURATION] <= 1000)
        ]

    if fix.empty:
        continue

    first_question_time = question_events.iloc[0][TIMESTAMP]
    fix = fix[fix[TIMESTAMP] >= first_question_time]

    # -------------------------------
    # Segmentierung
    # -------------------------------
    for i in range(len(question_events)):

        start_time = question_events.iloc[i][TIMESTAMP]

        if i < len(question_events) - 1:
            end_time = question_events.iloc[i + 1][TIMESTAMP]
        else:
            end_time = df[TIMESTAMP].max()

        question_name = question_events.iloc[i]["Event value"]

        question_fix = fix[
            (fix[TIMESTAMP] >= start_time) &
            (fix[TIMESTAMP] < end_time)
            ]

        if question_fix.empty:
            continue

        unique_fix = question_fix.drop_duplicates(subset=INDEX)

        fixation_count = unique_fix[INDEX].nunique()
        mean_fix = unique_fix[DURATION].mean()
        total_dwell = unique_fix[DURATION].sum()

        all_results.append([
            participant_name,
            question_name.split(" (")[0],
            fixation_count,
            round(mean_fix, 2) if pd.notna(mean_fix) else 0,
            total_dwell
        ])

# ---------------------------------------------------
# 3️⃣ DataFrame + Filter (22–30)
# ---------------------------------------------------

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

# Participant ID extrahieren
result_df["Participant_ID"] = result_df["Participant"].str.extract(r'(\d+)').astype(int)

# 🔥 Filter: nur 22–30
result_df = result_df[
    (result_df["Participant_ID"] >= 29) &
    (result_df["Participant_ID"] <= 41)
    ]

participants = result_df["Participant"].unique()

# ---------------------------------------------------
# 4️⃣ PDF
# ---------------------------------------------------

pdf_file = "All_Participants_Report_TestB.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=A4)
elements = []

styles = getSampleStyleSheet()

elements.append(Paragraph("Eye Tracking Report", styles["Heading1"]))
elements.append(Spacer(1, 0.4 * inch))

for p in participants:

    participant_answers = answers_df[answers_df["Participant"] == p]
    score_row = answers_df[answers_df["Participant"] == p]
    participant_score = int(score_row["Score"].iloc[0]) if not score_row.empty else "N/A"

    duration_row = answers_df[answers_df["Participant"] == p]
    participant_duration = duration_row["TotalDuration"].iloc[0] if "TotalDuration" in answers_df.columns else "N/A"

    participant_data = result_df[result_df["Participant"] == p]

    elements.append(Paragraph(f"Participant: {p}", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"<b>Total Score:</b> {participant_score}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Total Duration:</b> {participant_duration} s", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    table_data = [["Question", "Fix Count", "Mean Fix (ms)", "Total Dwell (ms)"]]

    for _, row in participant_data.iterrows():
        table_data.append([
            row["Question"],
            row["Fix_Count"],
            row["Mean_Fix_ms"],
            row["Total_Dwell_ms"]
        ])

    table = Table(table_data, colWidths=[150, 80, 100, 120], repeatRows=1)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    answer_table_data = [["Question", "Seconds", "Correct"]]

    for _, row in participant_answers.iterrows():
        result_symbol = "✓" if row["Correct"] == 1 else "✗"
        answer_table_data.append([row["Question"], row["Seconds"], result_symbol])

    answer_table = Table(answer_table_data, colWidths=[120, 80, 60], repeatRows=1)

    answer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
    ]))

    elements.append(answer_table)
    elements.append(PageBreak())

doc.build(elements)

print("✅ PDF erstellt:", pdf_file)

# ---------------------------------------------------
# 5️⃣ ANOVA
# ---------------------------------------------------

try:
    import pingouin as pg

    print("\nCheck counts:\n", result_df.groupby(["Participant", "Visualization"]).size())

    anova = pg.rm_anova(
        data=result_df,
        dv="Mean_Fix_ms",
        within="Visualization",
        subject="Participant",
        detailed=True
    )

    print("\nANOVA:\n", anova)

    posthoc = pg.pairwise_tests(
        data=result_df,
        dv="Mean_Fix_ms",
        within="Visualization",
        subject="Participant",
        padjust="bonf"
    )

    print("\nPosthoc:\n", posthoc)

except ImportError:
    print("\n⚠️ pingouin nicht installiert → ANOVA übersprungen")
    print("👉 pip install pingouin")
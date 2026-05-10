import os
import glob
import pandas as pd

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------

# Standardized column names used throughout the script
TIMESTAMP = "Recording timestamp [ms]"
DURATION = "Gaze event duration [ms]"
INDEX = "Eye movement type index"

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------

# Folder containing all participant TSV files for Test B
folder_path = "../../data/testB/"
files = glob.glob(os.path.join(folder_path, "*.tsv"))

# Load score and answer data
scores_df = pd.read_csv("../../data/testB/scores.csv")

answers_df = pd.read_csv(
    "../../data/testB/answers.csv",
    sep=",",              # Change to ";" if the file uses semicolons
    on_bad_lines="skip"
)

# Print columns for debugging
print(answers_df.columns.tolist())

# Store all extracted fixation metrics
all_results = []

print("Found participants:", len(files))

# ---------------------------------------------------
# PROCESS PARTICIPANT FILES
# ---------------------------------------------------

for file in files:

    # Extract participant name from the filename
    participant_name = os.path.basename(file).replace(".tsv", "")

    # Load participant eye-tracking data
    df = pd.read_csv(file, sep="\t", low_memory=False)

    # -------------------------------
    # Detect timestamp column
    # -------------------------------

    timestamp_col = None

    for col in df.columns:
        if "recording" in col.lower() and "timestamp" in col.lower():
            timestamp_col = col
            break

    if timestamp_col is None:
        print(f"⚠️ No timestamp column found in {file}")
        continue

    # Rename timestamp column to a standardized name
    df = df.rename(columns={timestamp_col: TIMESTAMP})

    # -------------------------------
    # Detect fixation duration column
    # -------------------------------

    duration_col = None

    for col in df.columns:
        if "duration" in col.lower() and ("gaze" in col.lower() or "fixation" in col.lower()):
            duration_col = col
            break

    if duration_col is None:
        print(f"⚠️ No duration column found in {file}")
        continue

    # Rename duration column to a standardized name
    df = df.rename(columns={duration_col: DURATION})

    # -------------------------------
    # Detect fixation index column
    # -------------------------------

    index_col = None

    for col in df.columns:
        if "index" in col.lower() and "movement" in col.lower():
            index_col = col
            break

    if index_col is None:
        print(f"⚠️ No fixation index column found in {file}")
        continue

    # Rename index column to a standardized name
    df = df.rename(columns={index_col: INDEX})

    # -------------------------------
    # Detect question start events
    # -------------------------------

    question_events = df[
        (df["Event"] == "URLStart") &
        (df["Event value"].astype(str).str.contains("Question", na=False))
    ].copy()

    if question_events.empty:
        continue

    # Sort question events chronologically
    question_events = question_events.sort_values(TIMESTAMP)

    # -------------------------------
    # Extract and clean fixations
    # -------------------------------

    # Keep only fixation events
    fix = df[df["Eye movement type"] == "Fixation"].copy()

    # Keep only fixations within a reasonable duration range
    fix = fix[
        (fix[DURATION] >= 80) &
        (fix[DURATION] <= 1000)
    ]

    if fix.empty:
        continue

    # Ignore fixations before the first question starts
    first_question_time = question_events.iloc[0][TIMESTAMP]
    fix = fix[fix[TIMESTAMP] >= first_question_time]

    # -------------------------------
    # Segment fixations by question
    # -------------------------------

    for i in range(len(question_events)):

        start_time = question_events.iloc[i][TIMESTAMP]

        # Use the next question start as the current question end
        if i < len(question_events) - 1:
            end_time = question_events.iloc[i + 1][TIMESTAMP]
        else:
            end_time = df[TIMESTAMP].max()

        question_name = question_events.iloc[i]["Event value"]

        # Select fixations within the current question time window
        question_fix = fix[
            (fix[TIMESTAMP] >= start_time) &
            (fix[TIMESTAMP] < end_time)
        ]

        if question_fix.empty:
            continue

        # Remove duplicate fixation events
        unique_fix = question_fix.drop_duplicates(subset=INDEX)

        # Compute fixation metrics
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
# CREATE RESULT DATAFRAME AND FILTER PARTICIPANTS
# ---------------------------------------------------

result_df = pd.DataFrame(all_results, columns=[
    "Participant",
    "Question",
    "Fix_Count",
    "Mean_Fix_ms",
    "Total_Dwell_ms"
])

# Extract visualization name from the question label
result_df["Visualization"] = result_df["Question"].str.replace(
    r"Question \d+ – ",
    "",
    regex=True
)

# Extract numeric participant ID
result_df["Participant_ID"] = result_df["Participant"].str.extract(r'(\d+)').astype(int)

# Keep only participants with IDs from 29 to 41
result_df = result_df[
    (result_df["Participant_ID"] >= 29) &
    (result_df["Participant_ID"] <= 41)
]

participants = result_df["Participant"].unique()

# ---------------------------------------------------
# CREATE PDF REPORT
# ---------------------------------------------------

pdf_file = "All_Participants_Report_TestB.pdf"

# Set up PDF document
doc = SimpleDocTemplate(pdf_file, pagesize=A4)
elements = []

styles = getSampleStyleSheet()

# Report title
elements.append(Paragraph("Eye Tracking Report", styles["Heading1"]))
elements.append(Spacer(1, 0.4 * inch))

# Create one report section per participant
for p in participants:

    participant_answers = answers_df[answers_df["Participant"] == p]

    # Extract participant score from the answers file
    score_row = answers_df[answers_df["Participant"] == p]
    participant_score = int(score_row["Score"].iloc[0]) if not score_row.empty else "N/A"

    # Extract total task duration
    duration_row = answers_df[answers_df["Participant"] == p]
    participant_duration = (
        duration_row["TotalDuration"].iloc[0]
        if "TotalDuration" in answers_df.columns
        else "N/A"
    )

    # Select fixation metrics for the current participant
    participant_data = result_df[result_df["Participant"] == p]

    # Participant header
    elements.append(Paragraph(f"Participant: {p}", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"<b>Total Score:</b> {participant_score}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Total Duration:</b> {participant_duration} s", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # -------------------------------
    # Fixation metrics table
    # -------------------------------

    table_data = [["Question", "Fix Count", "Mean Fix (ms)", "Total Dwell (ms)"]]

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
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # -------------------------------
    # Answer correctness table
    # -------------------------------

    answer_table_data = [["Question", "Seconds", "Correct"]]

    for _, row in participant_answers.iterrows():

        # Use a checkmark for correct answers and a cross for incorrect answers
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
        ("BACKGROUND", (0, 0), (-1, 0), colors.green),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ]))

    elements.append(answer_table)

    # Start a new page for the next participant
    elements.append(PageBreak())

# Build and save PDF report
doc.build(elements)

print("✅ PDF created:", pdf_file)

# ---------------------------------------------------
# REPEATED-MEASURES ANOVA
# ---------------------------------------------------

try:
    import pingouin as pg

    # Check how many observations exist per participant and visualization
    print("\nCheck counts:\n", result_df.groupby(["Participant", "Visualization"]).size())

    # Run repeated-measures ANOVA on mean fixation duration
    anova = pg.rm_anova(
        data=result_df,
        dv="Mean_Fix_ms",
        within="Visualization",
        subject="Participant",
        detailed=True
    )

    print("\nANOVA:\n", anova)

    # Run Bonferroni-corrected pairwise post-hoc tests
    posthoc = pg.pairwise_tests(
        data=result_df,
        dv="Mean_Fix_ms",
        within="Visualization",
        subject="Participant",
        padjust="bonf"
    )

    print("\nPosthoc:\n", posthoc)

except ImportError:
    print("\n⚠️ pingouin is not installed → ANOVA skipped")
    print("👉 Install it with: pip install pingouin")
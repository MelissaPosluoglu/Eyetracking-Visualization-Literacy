# Mini-VLAT Eye-Tracking – Visualization Literacy

Dieses Repository enthält den Analysecode für eine Eye-Tracking-Studie zur Untersuchung der Visualisierungskompetenz (Visualization Literacy) mithilfe des **Mini-VLAT-Tests**.  
Der Fokus liegt auf der zeitlich präzisen Segmentierung einzelner Testfragen sowie auf der Analyse und Visualisierung von Fixationen auf stimulusbasierter Ebene.

---

## 📌 Projektkontext

Das Projekt wurde im Rahmen eines universitären Forschungsprojekts durchgeführt und untersucht den Zusammenhang zwischen:

- Visualisierungskompetenz (Mini-VLAT)
- Blickverhalten (Eye Tracking)
- Zeitdruck und Aufgabenbearbeitung

Die Eye-Tracking-Daten wurden mit **Tobii** aufgezeichnet und als `.tsv` exportiert.  
Die Segmentierung der Aufgaben erfolgt anhand von **URL Start / URL End Events**, die während der webbasierten Durchführung des Mini-VLAT-Tests geloggt wurden.

---

## 🐍 Python-Umgebung

Dieses Projekt verwendet **Python 3.11**.

### Voraussetzungen

- Python **3.11.x**
- Git
- Windows / macOS / Linux

---

## ⚙️ Setup (virtuelle Umgebung)

### 1️⃣ Repository klonen

```bash
git clone https://github.com/MelissaPosluoglu/mini-vlat-eyetracking.git
cd mini-vlat-eyetracking
```

### 2️⃣ Virtuelle Umgebung erstellen
```bash
python -m venv .venv
```

### 3️⃣ Virtuelle Umgebung aktivieren

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source .venv/bin/activate
```

Nach der Aktivierung sollte im Terminal Folgendes erscheinen:
```
(.venv)
```

### 4️⃣ Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

---

## ▶️ Ausführen der Fixationsanalyse

Die Fixationsvisualisierung erzeugt für jede der 12 Mini-VLAT-Fragen eine eigene Abbildung, 
in der Fixationen über dem jeweiligen Stimulusbild dargestellt werden.

Skript starten
```bash
cd analysis
python fixation_all_questions.py
```


## 📊 Output

- Für jede Frage wird ein Fixationsplot erzeugt
- Die Ergebnisse werden automatisch im Ordner `output_fixations/` gespeichert

Jede Abbildung zeigt:

- Fixationspositionen
- Fixationsdauer (Punktgröße)
- stimulus-spezifische Blickverteilung

---

## 🧠 Methodik (kurz)

- Die Segmentierung der Eye-Tracking-Daten erfolgt über **URL Start / URL End Events**
- Jede Fixation wird eindeutig einer Mini-VLAT-Frage zugeordnet
- Die Analyse erfolgt stimulusbasiert (eine Frage = ein Bild)
- Die Fixationsdauer wird visuell skaliert dargestellt

---

## 📄 Lizenz

Dieses Projekt steht unter der MIT License.
Eine freie Nutzung für Forschungs- und Lehrzwecke ist ausdrücklich erlaubt.
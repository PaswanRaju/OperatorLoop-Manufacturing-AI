from pathlib import Path
import pandas as pd
from operatorloop import analyze_run, log_feedback

ROOT = Path(__file__).parent
data = pd.read_csv(ROOT / "data" / "synthetic_runs.csv")
row, label, retrieved, prompt, rec, violations = analyze_run(data, "R005")
log_feedback("R005", "edit", rec, "Inspect the machine before making any parameter change.")
print("Logged one example human edit for R005.")

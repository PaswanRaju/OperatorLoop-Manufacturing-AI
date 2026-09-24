from pathlib import Path
import pandas as pd
from operatorloop import classify_snapshot, validate_recommendation, retrieve_guidance, baseline_recommendation

ROOT = Path(__file__).parent

df = pd.read_csv(ROOT / "data" / "synthetic_runs.csv")
df["predicted"] = df.apply(classify_snapshot, axis=1)
true_anom = df["injected_anomaly"] != "normal"
pred_anom = df["predicted"] != "normal_operation"
coverage = ((true_anom & pred_anom).sum() / true_anom.sum()) if true_anom.sum() else 0
false_alarm = ((~true_anom & pred_anom).sum() / (~true_anom).sum()) if (~true_anom).sum() else 0

violations = 0
for _, row in df.iterrows():
    label = classify_snapshot(row)
    retrieved = retrieve_guidance(f"{label} nozzle temperature vibration layer time defect score")
    rec = baseline_recommendation(label, retrieved)
    violations += len(validate_recommendation(row, rec))

print(f"Synthetic runs: {len(df)}")
print(f"Injected anomalies: {true_anom.sum()}")
print(f"Anomaly coverage: {coverage:.1%}")
print(f"False-alarm rate: {false_alarm:.1%}")
print(f"Constraint violations from baseline recommendations: {violations}")

fb = ROOT / "data" / "operator_feedback.csv"
if fb.exists():
    feedback = pd.read_csv(fb)
    approved = (feedback["decision"] == "approve").mean() if len(feedback) else 0
    print(f"Operator feedback records: {len(feedback)}")
    print(f"Approval rate: {approved:.1%}")
else:
    print("Operator feedback records: 0 (run the CLI and log feedback to populate this metric)")

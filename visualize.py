from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "synthetic_runs.csv"

df = pd.read_csv(DATA_PATH)

# Only highlight vibration-related injected anomalies
vibration_anomalies = df[
    df["injected_anomaly"] == "vibration_instability"
]

plt.figure(figsize=(11, 5))

plt.plot(
    range(len(df)),
    df["vibration_g"],
    label="Vibration"
)

plt.scatter(
    vibration_anomalies.index,
    vibration_anomalies["vibration_g"],
    label="Vibration anomaly"
)

plt.axhline(
    y=1.2,
    linestyle="--",
    label="Vibration threshold (1.2 g)"
)

plt.xlabel("Manufacturing Run")
plt.ylabel("Vibration (g)")
plt.title("Manufacturing Vibration Monitoring")

# Show only every 10th run label
step = 10
plt.xticks(
    range(0, len(df), step),
    df["run_id"].iloc[::step],
    rotation=45
)

plt.legend()
plt.tight_layout()

plt.savefig("vibration_monitoring.png", dpi=200)
plt.show()
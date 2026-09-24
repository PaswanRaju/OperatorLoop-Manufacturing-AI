from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data" / "synthetic_runs.csv"
rng = np.random.default_rng(42)
n = 160

df = pd.DataFrame({
    "run_id": [f"R{i:03d}" for i in range(1, n + 1)],
    "nozzle_temp_c": rng.normal(215, 5, n),
    "vibration_g": np.clip(rng.normal(0.72, 0.14, n), 0.25, None),
    "layer_time_s": rng.normal(12.5, 1.0, n),
    "defect_score": np.clip(rng.normal(0.18, 0.07, n), 0, 1),
})

df["injected_anomaly"] = "normal"
# Inject four interpretable manufacturing anomalies.
idx = rng.choice(n, size=28, replace=False)
for j, i in enumerate(idx):
    mode = j % 4
    if mode == 0:
        df.loc[i, "nozzle_temp_c"] += rng.uniform(22, 34)
        df.loc[i, "defect_score"] += rng.uniform(0.28, 0.5)
        df.loc[i, "injected_anomaly"] = "thermal_overheating"
    elif mode == 1:
        df.loc[i, "vibration_g"] += rng.uniform(0.65, 1.0)
        df.loc[i, "defect_score"] += rng.uniform(0.18, 0.38)
        df.loc[i, "injected_anomaly"] = "vibration_instability"
    elif mode == 2:
        df.loc[i, "layer_time_s"] += rng.uniform(4.5, 7.0)
        df.loc[i, "injected_anomaly"] = "slow_layer_cycle"
    else:
        df.loc[i, "defect_score"] += rng.uniform(0.45, 0.65)
        df.loc[i, "injected_anomaly"] = "quality_drift"

df["defect_score"] = df["defect_score"].clip(0, 1)
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} synthetic runs to {OUT}")

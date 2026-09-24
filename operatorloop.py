from pathlib import Path
import csv
import json
from datetime import datetime, timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
KB_PATH = ROOT / "data" / "process_knowledge.csv"
FEEDBACK_PATH = ROOT / "data" / "operator_feedback.csv"


def classify_snapshot(row: pd.Series) -> str:
    if row["nozzle_temp_c"] > 235 and row["defect_score"] > 0.40:
        return "thermal_overheating"
    if row["vibration_g"] > 1.20:
        return "vibration_instability"
    if row["defect_score"] > 0.55:
        return "quality_drift"
    if row["layer_time_s"] > 16.0:
        return "slow_layer_cycle"
    return "normal_operation"


def snapshot_text(row: pd.Series) -> str:
    return (
        f"nozzle temperature {row['nozzle_temp_c']:.1f} C, "
        f"vibration {row['vibration_g']:.2f} g, "
        f"layer time {row['layer_time_s']:.1f} s, "
        f"defect score {row['defect_score']:.2f}"
    )



def retrieve_guidance(query: str, top_k: int = 2):
    kb = pd.read_csv(KB_PATH)

    # 1. Prioritize an exact issue match
    issue_query = query.split()[0]
    exact_matches = kb[kb["issue"] == issue_query]

    results = []

    if not exact_matches.empty:
        exact_row = exact_matches.iloc[0].to_dict()
        exact_row["similarity"] = 1.0
        results.append(exact_row)

    # 2. Use TF-IDF for additional related guidance
    remaining_kb = kb[kb["issue"] != issue_query]

    if len(results) < top_k and not remaining_kb.empty:
        docs = (
            remaining_kb["issue"] + " "
            + remaining_kb["signals"] + " "
            + remaining_kb["guidance"]
        ).tolist()

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        )

        matrix = vec.fit_transform(docs + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

        needed = top_k - len(results)
        top = scores.argsort()[::-1][:needed]

        for i in top:
            row = remaining_kb.iloc[i].to_dict()
            row["similarity"] = float(scores[i])
            results.append(row)

    return results


def build_grounded_prompt(row: pd.Series, retrieved: list[dict]) -> str:
    context = "\n".join(
        f"- {r['issue']}: {r['guidance']} Constraint: {r['constraint']}"
        for r in retrieved
    )
    return f"""You are assisting a manufacturing operator. Treat the operator as the final decision-maker.
Use only the supplied process guidance and sensor snapshot. Recommend at most one conservative action,
explain the reason briefly, and state when human inspection is required. Do not invent machine limits.

SENSOR SNAPSHOT
{snapshot_text(row)}

RETRIEVED PROCESS GUIDANCE
{context}

Return fields: observed_issue, recommended_action, rationale, human_check, cited_guidance.
"""


def baseline_recommendation(label: str, retrieved: list[dict]) -> dict:
    best = next((r for r in retrieved if r["issue"] == label), retrieved[0])
    return {
        "observed_issue": label,
        "recommended_action": best["guidance"],
        "rationale": "Recommendation is grounded in the closest process guidance for the detected condition.",
        "human_check": "Operator must approve, edit, or reject the recommendation before any process change.",
        "cited_guidance": best["issue"],
    }


def validate_recommendation(row: pd.Series, rec: dict) -> list[str]:
    text = rec["recommended_action"].lower()
    violations = []

    # Do not allow speed increases during excessive vibration
    if (
        row["vibration_g"] > 1.2
        and "increase" in text
        and "speed" in text
    ):
        violations.append(
            "Speed increase suggested while vibration is above 1.2 g."
        )

    # Flag aggressive changes only when they are actually being recommended,
    # not when the guidance says to avoid or hold them.
    safe_aggressive_phrases = [
        "hold aggressive",
        "avoid aggressive",
        "do not make aggressive",
        "do not recommend aggressive",
    ]

    if (
        row["defect_score"] > 0.55
        and "aggressive" in text
        and not any(phrase in text for phrase in safe_aggressive_phrases)
    ):
        violations.append(
            "Aggressive change suggested while defect score is high."
        )

    return violations


def log_feedback(run_id: str, decision: str, original: dict, edited_action: str = ""):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "run_id", "decision", "original_action", "edited_action"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "decision": decision,
            "original_action": original["recommended_action"],
            "edited_action": edited_action,
        })


def analyze_run(df: pd.DataFrame, run_id: str):
    match = df[df["run_id"] == run_id]
    if match.empty:
        raise ValueError(f"Unknown run_id: {run_id}")
    row = match.iloc[0]
    label = classify_snapshot(row)
    query = f"{label} {snapshot_text(row)}"
    retrieved = retrieve_guidance(query)
    prompt = build_grounded_prompt(row, retrieved)
    rec = baseline_recommendation(label, retrieved)
    violations = validate_recommendation(row, rec)
    return row, label, retrieved, prompt, rec, violations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OperatorLoop human-in-the-loop manufacturing AI sandbox")
    parser.add_argument("run_id", nargs="?", default="R001")
    args = parser.parse_args()

    data = pd.read_csv(ROOT / "data" / "synthetic_runs.csv")
    row, label, retrieved, prompt, rec, violations = analyze_run(data, args.run_id)
    print("Snapshot:", snapshot_text(row))
    print("Detected condition:", label)
    print("\nRetrieved guidance:")
    for item in retrieved:
        print(f"  - {item['issue']} (similarity={item['similarity']:.3f})")
    print("\nLLM-ready grounded prompt:\n")
    print(prompt)
    print("Baseline recommendation:\n", json.dumps(rec, indent=2))
    print("Constraint violations:", violations or "none")

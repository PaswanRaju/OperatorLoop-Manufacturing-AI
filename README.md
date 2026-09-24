# OperatorLoop

**Human-in-the-Loop Manufacturing Optimization Sandbox**

OperatorLoop is a small Python research prototype for exploring how an LLM-style decision-support workflow could combine manufacturing sensor data, retrieved process guidance, safety constraints, and operator feedback.

## What is implemented

- Synthetic additive-manufacturing-style sensor data (temperature, vibration, layer time, defect score)
- Interpretable anomaly classification
- TF-IDF retrieval over a small manufacturing process knowledge base
- Construction of a grounded, LLM-ready prompt from sensor context + retrieved guidance
- A deterministic baseline recommendation so the prototype runs without an external API
- A constraint checker for unsafe/inconsistent recommendations
- Operator approve/edit/reject feedback logging
- Simple evaluation metrics for anomaly coverage, false alarms, and constraint violations

## Important limitation

The current version **does not call a live LLM**. It intentionally separates the research workflow from the model provider. The next step is to plug the generated prompt into an LLM API or local model and compare LLM recommendations against the deterministic baseline and human feedback.

## Run

```bash
pip install -r requirements.txt
python generate_data.py
python operatorloop.py R010
python evaluate.py
```

## Research questions this prototype can support

1. Does retrieval-grounded context reduce unsupported manufacturing recommendations?
2. How often do human operators approve, edit, or reject model suggestions?
3. Can simple process constraints catch recommendations that should not be executed?
4. How should operator feedback be stored and reused in later recommendations?

This is an educational prototype using synthetic data, not a production manufacturing control system.

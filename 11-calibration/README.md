# 11 · Calibration

**Level 3 · Reliability · no API key**

## What it is
A model is **calibrated** when its confidence matches reality: of the predictions
it makes at 90% confidence, about 90% should actually be correct. A **reliability
diagram** plots accuracy against confidence per bin; **Expected Calibration
Error (ECE)** summarizes the average gap between the two. Overconfident models sit
above the diagonal — they claim more certainty than they earn.

## Why it matters in production
The moment you gate a decision on confidence — "auto-approve if the model is >0.9
sure, else route to a human" — calibration becomes load-bearing. An
overconfident model floods the auto-approve path with wrong answers. Calibration
metrics tell you whether a confidence threshold means what you think, and
techniques like temperature scaling fix it.

## Where it's used
Selective prediction / human-in-the-loop routing, risk-based automation
thresholds, model monitoring dashboards, and anywhere probabilities feed
downstream decisions.

## Run it
```bash
python 11-calibration/demo.py
```
Builds a numpy reliability diagram and ECE for a well-calibrated vs an
overconfident model, showing accuracy per confidence band (with "—" for bands the
overconfident model never emits) and ECE for both.

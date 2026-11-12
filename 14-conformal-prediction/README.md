# 14 · Conformal Prediction

**Level 4 · Reliability · no API key**

## What it is
Conformal prediction wraps any trained classifier so that, instead of a single
label, it outputs a **prediction set** that contains the true label with a chosen
frequency (here 90%). Split conformal: hold out a calibration set, score each
point's **nonconformity** (`1 - p(true class)`), take the `1-alpha` quantile as a
threshold, and at test time include every class whose score falls under it. The
only assumption is **exchangeability** of calibration and test data — no
distributional assumptions about the model.

## Why it matters in production
It gives a principled, model-agnostic notion of uncertainty. Big sets flag hard
inputs that deserve human review; singletons are confident calls you can
automate. The coverage property is a **finite-sample expectation over
exchangeable data**, not a per-prediction promise — but it's far more honest than
an uncalibrated softmax score.

## Where it's used
Selective prediction and human-in-the-loop routing, risk-controlled automation,
medical/finance decision support, and uncertainty quantification for LLM
classification.

## Run it
```bash
python 14-conformal-prediction/demo.py
```
Split conformal classification on iris (sklearn): train/calibrate/test split,
nonconformity = 1 − p(true), quantile threshold at alpha=0.10, prediction sets,
then empirical coverage, average set size, and abstention rate.

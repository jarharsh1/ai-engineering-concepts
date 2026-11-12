# 15 · Differential Privacy

**Level 4 · Privacy · no API key**

## What it is
Differential privacy (DP) adds **calibrated random noise** to a computation so
that its output is almost identical whether or not any single individual is in
the dataset. The privacy budget **epsilon** controls the trade-off: small epsilon
means more noise and stronger privacy; large epsilon means less noise and weaker
privacy. The **Laplace mechanism** achieves this for numeric queries by adding
noise scaled to the query's *sensitivity* (how much one record can change the
result) divided by epsilon.

## Why it matters in production
DP gives a *provable*, worst-case guarantee — unlike ad-hoc anonymization, which
repeatedly fails to real re-identification attacks. It lets you publish
aggregates, train models, or expose analytics over sensitive data while bounding
what any one person's inclusion reveals. The cost is accuracy: you explicitly
spend privacy budget for utility.

## Where it's used
Census and analytics releases, private model training (DP-SGD), telemetry
aggregation, and federated learning.

## Run it
```bash
python 15-differential-privacy/demo.py
```
Applies the Laplace mechanism to a private mean over ~500 values. Over many
independent noise draws it reports the distribution (mean ± std) of the released
value with vs without one outlier ("Alice") at epsilon = 10, 1, 0.1 — showing the
noise std grow until it dwarfs Alice's effect and the two distributions overlap
(she goes from "exposed" at large epsilon to "hidden" at small epsilon).

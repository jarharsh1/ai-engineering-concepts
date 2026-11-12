# 10 · Canary Deployments for Prompts

**Level 3 · MLOps · key optional**

## What it is
A canary deployment routes a small slice of traffic (e.g. 5%) to a new version
while the rest stays on the proven one, measures both, and promotes only if the
canary wins. Applied to **prompts**: a prompt edit is a production change with
unpredictable effects, so you treat `v2.3` of a prompt exactly like a code
rollout — small exposure, measured signal, automated promote/rollback gate.

## Why it matters in production
Prompt changes can silently regress quality, cost, or latency on real traffic in
ways offline evals miss. Canarying catches regressions on a fraction of users
instead of all of them, and gives you a data-driven promotion decision instead of
"it looked fine in testing."

## Where it's used
LLM app platforms with prompt versioning, A/B and online-eval systems, feature
flags for prompts/models, and progressive-delivery tooling.

## Run it
```bash
python 10-canary-prompts/demo.py
```
Simulates a 95/5 traffic split between stable v2.2 and canary v2.3, measures a
per-request quality signal, and prints a promote/rollback gate decision.

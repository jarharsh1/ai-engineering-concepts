# 08 · CQRS

**Level 3 · Architecture · no API key**

## What it is
**Command Query Responsibility Segregation** splits the write model (commands
that change state) from the read model (queries that return state), so each side
is optimized on its own terms. Writes can be normalized and transactional; reads
can serve from denormalized, pre-computed, independently-scaled views.

## Why it matters in production
Read and write workloads have opposite needs. Mixing them means heavy ingestion
contends with queries — locks, hot tables, and stalled dashboards. Separating
them lets you scale reads horizontally, cache aggressively, and keep query
latency flat under write pressure. The cost is **eventual consistency**: the read
view lags the write side by some small delay, which most read-heavy systems can
tolerate.

## Where it's used
Event-sourced systems, high-read dashboards and search indexes, materialized
read replicas, and feature stores that separate ingestion from serving.

## Run it
```bash
python 08-cqrs/demo.py
```
A threaded demo: a naive single-lock store where ingestion blocks queries, vs a
split read-view where queries stay fast during heavy ingestion. Prints query
latency during ingestion for both.

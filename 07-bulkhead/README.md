# 07 · Bulkhead

**Level 2 · Reliability · no API key**

## What it is
A bulkhead isolates resources per dependency, like the watertight compartments
of a ship. Each downstream service gets its own pool of threads / connections /
concurrency slots, so exhausting one pool (because that dependency is slow or
down) cannot starve calls to the others.

## Why it matters in production
The classic cascading failure: one dependency hangs, every worker thread blocks
waiting on it, and now your *entire* service is unresponsive — even for requests
that never needed the broken dependency. Bulkheads cap how many resources any one
dependency can tie up, containing the blast radius. Pair with timeouts and
circuit breakers for full isolation.

## Where it's used
Resilience libraries (Resilience4j, Polly), service meshes, connection-pool
sizing per backend, and LLM gateways that call multiple model providers.

## Run it
```bash
python 07-bulkhead/demo.py
```
Compares a shared semaphore (the failing service contends with all) against
per-service semaphores (only the failing service degrades), printing average
latency per service in each case.

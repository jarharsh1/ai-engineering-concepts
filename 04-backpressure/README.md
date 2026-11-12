# 04 · Backpressure

**Level 2 · Reliability · no API key**

## What it is
Backpressure is the practice of **admitting only as much work as you can serve
and rejecting the rest immediately** (typically HTTP 429). It's the opposite of
unbounded queuing. By pushing back on the sender, you keep the requests you did
admit fast and healthy instead of letting everyone slow down together.

## Why it matters in production
LLM serving is expensive and bounded by GPU memory (see
[01-kv-cache-effect](../01-kv-cache-effect)). Without admission control, a
traffic spike fills queues, inflates tail latency, and eventually OOMs the
server — taking down the requests that were already in flight. A bounded
semaphore (or token bucket, or queue with a max depth) converts a cascading
outage into a clean, retryable rejection.

## Where it's used
API gateways, inference servers (`max_num_seqs`, concurrency limits), rate
limiters, and any backend fronting a scarce resource.

## Run it
```bash
python 04-backpressure/demo.py
```
Compares "accept all" (serves 500, no shedding) against a bounded semaphore
(CAPACITY=200 → serve 200, shed 300 as 429).

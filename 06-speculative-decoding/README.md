# 06 · Speculative Decoding

**Level 2 · Serving · key optional**

## What it is
Normal decoding emits one token per forward pass of the large model — pure
latency. **Speculative decoding** uses a small, fast *draft* model to guess the
next K tokens, then the large *target* model verifies all K in a single pass.
Correct guesses are accepted; at the first wrong guess you discard the rest and
re-draft. Because the big model verifies every token, **the output is bit-for-bit
identical** to standard decoding — it's a pure latency optimization, not a
quality trade-off.

## Why it matters in production
Decode latency dominates user-perceived speed for chat and streaming. If the
draft model is right most of the time, you produce several tokens per big-model
pass, cutting the number of expensive forward passes and raising throughput with
no change in output distribution.

## Where it's used
vLLM / TensorRT-LLM speculative decoding, Medusa / EAGLE draft heads, and
n-gram/prompt-lookup speculation for latency-sensitive serving.

## Run it
```bash
python 06-speculative-decoding/demo.py
```
Simulates draft(K=4)+verify over a fixed sentence (draft ~70% correct), counts
big-model calls vs the autoregressive baseline, and reports the speed-up — noting
the output is identical.

# 05 · Prefix Caching

**Level 2 · Serving · no API key**

## What it is
Prefix caching stores the KV cache for a **shared prompt prefix** (system prompt,
tool schemas, few-shot examples) once and reuses it across every request that
begins with the same prefix. Each request then only pays to prefill its unique
suffix instead of re-encoding thousands of identical tokens.

## Why it matters in production
Prefill (processing the input prompt) is often the dominant cost for
short-output LLM calls, and in agent/RAG systems the prefix is enormous and
nearly identical across calls. Caching it cuts prefill compute dramatically —
frequently 90%+ — lowering latency and cost at the same time. It's one of the
cheapest high-impact serving optimizations.

## Where it's used
vLLM automatic prefix caching, Anthropic/OpenAI prompt caching, and any serving
stack that hashes and reuses KV blocks for common prefixes.

## Run it
```bash
python 05-prefix-caching/demo.py
```
A token-accounting estimator: prefix=2000, unique=50, 1000 req/h. Prints tokens
encoded per hour with vs without prefix caching and the % saved.

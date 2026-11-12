# 01 · KV Cache Effect

**Level 1 · Serving · no API key**

## What it is
During autoregressive generation a transformer caches the **Key** and **Value**
tensors for every token it has already processed, so it never recomputes them.
The size of this cache is:

```
kv_bytes_per_token = 2 * layers * hidden * dtype_bytes
```

It grows **linearly with sequence length** and **linearly with the number of
concurrent requests**.

## Why it matters in production
On most inference servers the KV cache — not raw compute — is what limits how
many users a single GPU can serve. A 40 GB GPU running an 8B model in fp16 has
~24 GB left for KV after weights; at 2048 tokens/user that caps you at a few
dozen concurrent sessions. Push past it and the server OOMs. This is the number
behind "why can't we just raise the concurrency limit?"

## Where it's used
Capacity planning for vLLM / TensorRT-LLM / TGI deployments, choosing
`max_num_seqs` and `max_model_len`, and deciding when you need
PagedAttention, quantized KV cache, or more GPUs.

## Run it
```bash
python 01-kv-cache-effect/demo.py
```
Prints KV bytes/token, max concurrent users, and an OK→OOM crash table.

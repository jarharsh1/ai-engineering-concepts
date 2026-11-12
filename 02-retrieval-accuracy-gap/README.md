# 02 · Retrieval Accuracy Gap

**Level 1 · Retrieval · key optional**

## What it is
A single embedding search (a **bi-encoder**) is fast because it embeds the query
and documents independently and compares vectors — but that independence costs
precision. A **cross-encoder reranker** reads the query and a candidate document
*together* and scores their actual relevance, which is far more accurate but too
slow to run over the whole corpus. The standard fix is two stages: recall a
shortlist cheaply, then rerank it precisely. The "accuracy gap" is what you lose
by stopping after stage one.

## Why it matters in production
Most "our RAG retrieves irrelevant chunks" problems are a missing reranker. The
right document is frequently in the bi-encoder's top-10 but not its top-1; a
reranker promotes it. Adding stage two is usually the single highest-leverage
retrieval-quality improvement.

## Where it's used
RAG pipelines, search, and any system with `retrieve -> rerank -> generate`
(e.g. bi-encoder recall + a cross-encoder or ColBERT-style reranker).

## Run it
```bash
python 02-retrieval-accuracy-gap/demo.py
```
Prints the top-4 after stage 1 and the top-2 after reranking. Set
`NVIDIA_API_KEY` to use real NIM embeddings for stage 1; otherwise it falls back
to lexical overlap.

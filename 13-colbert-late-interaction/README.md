# 13 · ColBERT / Late Interaction

**Level 4 · Retrieval · key optional**

## What it is
Standard dense retrieval compresses a whole document into a **single** vector,
which averages away fine-grained signal. **ColBERT** keeps a vector **per token**
and scores a query/document pair with **MaxSim**: for each query token, take its
maximum similarity against all document token vectors, then sum. This "late
interaction" happens at scoring time (not by pre-pooling), so a precise match on a
rare, important term survives instead of being averaged out.

## Why it matters in production
Single-vector retrieval struggles with queries that hinge on specific terms
(error codes, names, jargon) — exactly the queries that matter in technical
search. Late interaction recovers much of a cross-encoder's accuracy while
staying efficient enough to index at scale, making it a strong first-stage or
reranking retriever.

## Where it's used
ColBERT / ColBERTv2 / PLAID retrievers, late-interaction rerankers, and
multi-vector search in modern RAG stacks.

## Run it
```bash
python 13-colbert-late-interaction/demo.py
```
Compares MaxSim over per-token vectors against single averaged-vector cosine on a
query + 2 docs. Uses deterministic toy per-token vectors as a fallback, or NIM
per-token embeddings if a key is set — showing MaxSim rewards token-level matches.

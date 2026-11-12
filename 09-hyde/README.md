# 09 · HyDE (Hypothetical Document Embeddings)

**Level 3 · Retrieval · key recommended**

## What it is
HyDE improves retrieval for short or keyword-poor queries. Instead of embedding
the question directly, you first ask an LLM to **write a hypothetical answer** to
it, then embed *that* and search with it. A plausible fake answer shares far more
vocabulary and semantic structure with the true passage than the terse question
does, so the correct document ranks higher.

## Why it matters in production
Real user queries are short and under-specified ("refund time?"), which embed
poorly against full-sentence passages. HyDE closes that gap without fine-tuning
anything — one extra cheap LLM call per query. It's especially effective for
zero-shot retrieval over a new corpus.

## Where it's used
RAG query expansion, zero-shot dense retrieval, and search over domains where no
labeled query/passage pairs exist to train a better embedder.

## Run it
```bash
python 09-hyde/demo.py
```
Generates a hypothetical answer with NIM (canned fallback with no key), embeds
it, and compares retrieval ranking by the raw-question embedding vs the
HyDE-answer embedding over toy docs.

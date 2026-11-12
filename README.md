# AI Engineering Concepts

Fifteen self-contained topics across LLM serving, retrieval, reliability,
architecture, security, MLOps, and privacy — each with a short concept write-up
and a small, **runnable** Python demo you can execute in isolation. LLM,
embedding, and rerank demos target **NVIDIA NIM** via its OpenAI-compatible API
and degrade gracefully (offline fallback) when no key is set, so every demo runs
to completion with or without credentials.

## The 15 concepts

| # | Concept | Level | Category | What it is | Where it's used |
|---|---|---|---|---|---|
| 01 | [KV Cache Effect](01-kv-cache-effect) | L1 | Serving | Per-token KV memory that caps GPU concurrency | Inference capacity planning (vLLM/TRT-LLM) |
| 02 | [Retrieval Accuracy Gap](02-retrieval-accuracy-gap) | L1 | Retrieval | Bi-encoder recall then cross-encoder rerank | RAG pipelines, search |
| 03 | [4 Types of Prompt Injection](03-prompt-injection-types) | L1 | Security | Classifying direct/indirect/cross/multi-modal attacks | LLM guardrails, agent safety |
| 04 | [Backpressure](04-backpressure) | L2 | Reliability | Shed load instead of collapsing under it | API gateways, inference servers |
| 05 | [Prefix Caching](05-prefix-caching) | L2 | Serving | Reuse KV for a shared prompt prefix | Prompt caching, agent/RAG serving |
| 06 | [Speculative Decoding](06-speculative-decoding) | L2 | Serving | Small model drafts, big model verifies | Low-latency LLM serving |
| 07 | [Bulkhead](07-bulkhead) | L2 | Reliability | Isolate resource pools per dependency | Resilient microservices, LLM gateways |
| 08 | [CQRS](08-cqrs) | L3 | Architecture | Split the write path from the read path | High-read dashboards, event sourcing |
| 09 | [HyDE](09-hyde) | L3 | Retrieval | Embed a hypothetical answer, not the question | RAG query expansion |
| 10 | [Canary Prompts](10-canary-prompts) | L3 | MLOps | Roll out prompt changes to a traffic slice | Prompt versioning, online eval |
| 11 | [Calibration](11-calibration) | L3 | Reliability | Does stated confidence match accuracy? | Confidence-gated automation |
| 12 | [Saga](12-saga) | L4 | Architecture | Distributed transaction via compensations | Payment/booking workflows, agents |
| 13 | [ColBERT / Late Interaction](13-colbert-late-interaction) | L4 | Retrieval | Token-level MaxSim beats single-vector cosine | Late-interaction retrieval/reranking |
| 14 | [Conformal Prediction](14-conformal-prediction) | L4 | Reliability | Prediction sets with coverage in expectation | Selective prediction, HITL routing |
| 15 | [Differential Privacy](15-differential-privacy) | L4 | Privacy | Noise that bounds one person's data leakage | Private analytics, DP-SGD |

*Level 1 = foundational · Level 4 = advanced.*

## Quick start

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then paste your NVIDIA_API_KEY (optional)
```

Run any demo on its own:

```bash
python 01-kv-cache-effect/demo.py
python 09-hyde/demo.py
```

Every demo prints labeled output and finishes on its own. The LLM-backed ones
print a clear `[no NVIDIA_API_KEY] ...` note and use a fallback when no key is
configured.

## Which demos use the NIM key?

| Demo | NIM key | Behavior without a key |
|---|---|---|
| 01 KV Cache Effect | — | pure arithmetic |
| 02 Retrieval Accuracy Gap | optional | lexical fallback for stage 1 |
| 03 Prompt Injection | optional | regex scanner only (skips LLM judge) |
| 04 Backpressure | — | asyncio simulation |
| 05 Prefix Caching | — | token accounting |
| 06 Speculative Decoding | optional | deterministic simulation |
| 07 Bulkhead | — | asyncio simulation |
| 08 CQRS | — | threaded simulation |
| 09 HyDE | recommended | canned hypothesis + lexical fallback |
| 10 Canary Prompts | optional | self-contained simulation |
| 11 Calibration | — | numpy |
| 12 Saga | — | in-process orchestrator |
| 13 ColBERT / Late Interaction | optional | deterministic toy per-token vectors |
| 14 Conformal Prediction | — | sklearn (iris) |
| 15 Differential Privacy | — | numpy |

## Configuration

NVIDIA NIM is OpenAI-compatible. Defaults (see `.env.example`):

- `NVIDIA_BASE_URL` = `https://integrate.api.nvidia.com/v1`
- `NIM_CHAT_MODEL` = `meta/llama-3.1-8b-instruct`
- `NIM_EMBED_MODEL` = `nvidia/nv-embedqa-e5-v5`

Shared client logic lives in [`common/nim.py`](common/nim.py)
(`available()`, `chat(...)`, `embed(...)`).

## Requirements

Python 3.10+ and four packages: `openai`, `numpy`, `scikit-learn`,
`python-dotenv`. No GPU required — infra concepts are simulated in plain
Python/numpy.

## License

MIT — see [LICENSE](LICENSE).

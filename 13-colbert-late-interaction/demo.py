"""ColBERT / Late Interaction — match at the token level, not the sentence level.

A normal dense retriever squashes a whole document into ONE vector, so a strong
match on a single rare term gets averaged away. ColBERT keeps a vector per token
and scores with MaxSim: for each query token, take its best match among the
document's token vectors, then sum. This "late interaction" rewards documents
that contain the exact things the query asks about.

This demo compares, on a query + 2 docs:
  single-vector cosine : average-pooled document vector vs averaged query vector
  MaxSim (late interaction): per-token max-similarity, summed

Fallback uses deterministic toy per-token vectors; with NVIDIA_API_KEY it uses
NIM per-token embeddings.
Key optional.
"""
import os
import sys
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import nim  # noqa: E402

import numpy as np

QUERY = "gpu memory oom"
# Doc A contains the EXACT query terms. Doc B is a near-miss: it is topically
# about the same area (hardware / resource errors) so its *averaged* vector
# looks similar to the query, but it never uses the exact query tokens. This is
# the case where single-vector cosine is fooled and only MaxSim gets it right.
DOCS = {
    "A": "the gpu ran out of memory and threw an oom error while the model loaded",
    "B": "the hardware hit a resource crash when the compute device overloaded",
}

# Interpretable toy embeddings: each token is placed in a small concept space so
# that topically-related tokens have HIGH cosine but are NOT identical. Exact
# token identity still gives cosine 1.0 (a token matches itself). This lets us
# construct an honest near-miss instead of relying on opaque hash vectors.
CONCEPTS = ["hardware", "memory_res", "failure", "filler"]

# token -> weights over CONCEPTS (unnormalized; _unit() normalizes later).
TOKEN_CONCEPTS = {
    # query tokens
    "gpu":      {"hardware": 1.0, "memory_res": 0.2},
    "memory":   {"memory_res": 1.0, "hardware": 0.2},
    "oom":      {"failure": 1.0, "memory_res": 0.6},
    # Doc A exact matches reuse the same vectors as the query tokens above.
    # Doc B near-miss tokens: close to the query concepts but never identical.
    "hardware": {"hardware": 1.0, "failure": 0.3},
    "resource": {"memory_res": 0.7, "hardware": 0.4},
    "crash":    {"failure": 0.9, "hardware": 0.35},
    "compute":  {"hardware": 0.9, "memory_res": 0.3},
    "device":   {"hardware": 0.85, "memory_res": 0.25},
    "overloaded": {"failure": 0.7, "memory_res": 0.3, "hardware": 0.3},
    "hit":      {"failure": 0.5, "filler": 0.6},
    # shared/filler words (push averages toward a neutral 'filler' axis)
    "the": {"filler": 1.0}, "and": {"filler": 1.0}, "an": {"filler": 1.0},
    "of": {"filler": 1.0}, "out": {"filler": 1.0}, "ran": {"filler": 1.0},
    "threw": {"failure": 0.4, "filler": 0.7}, "error": {"failure": 0.7, "filler": 0.3},
    "while": {"filler": 1.0}, "model": {"filler": 0.8, "hardware": 0.2},
    "loaded": {"filler": 0.8, "hardware": 0.2}, "a": {"filler": 1.0},
    "when": {"filler": 1.0},
}


def toy_vec(token, dim=len(CONCEPTS)):
    """Interpretable concept-space embedding for a token (offline fallback).

    Unknown tokens fall back to a deterministic hashed vector so the demo never
    crashes on vocabulary it hasn't been told about.
    """
    weights = TOKEN_CONCEPTS.get(token)
    if weights is not None:
        v = np.array([weights.get(c, 0.0) for c in CONCEPTS], dtype=float)
    else:
        h = hashlib.sha256(token.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8)[:dim].astype(float)
        v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def token_vectors(text):
    toks = text.lower().split()
    if nim.available():
        try:
            # One embedding per token (asymmetric: passages as passages).
            vecs = nim.embed(toks, input_type="passage")
            return toks, np.array([_unit(v) for v in vecs])
        except Exception as e:
            print(f"[NIM embed failed: {e}] toy vectors")
    return toks, np.array([toy_vec(t) for t in toks])


def _unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def single_vector_score(q_vecs, d_vecs):
    q = _unit(q_vecs.mean(axis=0))
    d = _unit(d_vecs.mean(axis=0))
    return float(q @ d)


def maxsim_score(q_vecs, d_vecs):
    # For each query token, best similarity to any doc token; sum.
    sims = q_vecs @ d_vecs.T          # (|q|, |d|)
    return float(sims.max(axis=1).sum())


def main():
    print("=== ColBERT / Late Interaction (MaxSim) ===")
    print(f"Query: {QUERY!r}\n")
    if not nim.available():
        print("[no NVIDIA_API_KEY] using deterministic toy per-token vectors\n")

    _, q_vecs = token_vectors(QUERY)

    print(f"  {'doc':<4} | {'single-vector cos':>18} | {'MaxSim (late-int)':>18}")
    print(f"  {'-'*4}-+-{'-'*18}-+-{'-'*18}")
    results = {}
    for name, text in DOCS.items():
        _, d_vecs = token_vectors(text)
        sv = single_vector_score(q_vecs, d_vecs)
        ms = maxsim_score(q_vecs, d_vecs)
        results[name] = (sv, ms)
        print(f"  {name:<4} | {sv:>18.3f} | {ms:>18.3f}")

    sv_winner = max(results, key=lambda k: results[k][0])
    ms_winner = max(results, key=lambda k: results[k][1])
    print(f"\n  single-vector picks: Doc {sv_winner}")
    print(f"  MaxSim picks       : Doc {ms_winner}  (the relevant one)")

    print(
        "\nTakeaway: averaging into one vector dilutes the exact 'gpu/oom/memory' "
        "matches. MaxSim scores each query token against its best doc token, so it "
        "rewards the document that actually contains those terms."
    )


if __name__ == "__main__":
    main()

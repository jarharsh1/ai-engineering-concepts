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
DOCS = {
    "A": "the server crashed with a gpu out of memory oom error under load",
    "B": "our quarterly marketing report covers social media engagement metrics",
}


def toy_vec(token, dim=16):
    """Deterministic pseudo-embedding for a token (offline fallback)."""
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

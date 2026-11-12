"""Retrieval Accuracy Gap — why one-stage retrieval under-delivers.

A single bi-encoder (embedding + cosine) is fast but imprecise: it ranks by
coarse semantic similarity. A second-stage reranker (cross-encoder) looks at the
query and document *together* and reorders the shortlist much more accurately.
The "gap" is the quality you leave on the table by skipping stage two.

This demo runs the two stages over ~6 toy docs:
  stage 1: bi-encoder recall -> top-4   (NIM embeddings + cosine; lexical fallback)
  stage 2: cross-encoder-style rerank -> top-2 (lexical joint-overlap stand-in)

Key optional: with NVIDIA_API_KEY set, stage 1 uses real embeddings.
"""
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import nim  # noqa: E402

DOCS = [
    "Our refund policy allows returns within 30 days of purchase with a receipt.",
    "To reset your password, click 'Forgot password' on the login screen.",
    "The annual developer conference takes place in Berlin every September.",
    "Refunds are processed to the original payment method within 5 business days.",
    "Our office is closed on public holidays and reopens the next business day.",
    "You can update your billing address in Account Settings under Payment.",
]
QUERY = "how long do refunds take to show up on my card"


def _tok(s: str):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def cosine(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def bi_encoder_rank(query, docs):
    """Stage 1: return list of (idx, score) sorted desc."""
    if nim.available():
        try:
            q = nim.embed([query], input_type="query")[0]
            ds = nim.embed(docs, input_type="passage")
            scored = [(i, cosine(q, d)) for i, d in enumerate(ds)]
            return sorted(scored, key=lambda x: -x[1]), "NIM embeddings + cosine"
        except Exception as e:
            print(f"[NIM embed failed: {e}] falling back to lexical")
    # Lexical fallback: Jaccard overlap.
    qt = _tok(query)
    scored = [(i, len(qt & _tok(d)) / (len(qt | _tok(d)) or 1)) for i, d in enumerate(docs)]
    return sorted(scored, key=lambda x: -x[1]), "lexical Jaccard (fallback)"


def rerank(query, candidate_idxs):
    """Stage 2 stand-in: joint overlap weighted by rare-term match."""
    qt = _tok(query)
    scored = []
    for i in candidate_idxs:
        dt = _tok(DOCS[i])
        overlap = qt & dt
        # Reward matching the salient terms (refund/card/show).
        weight = sum(2 if t in {"refund", "refunds", "card", "payment"} else 1 for t in overlap)
        scored.append((i, weight))
    return sorted(scored, key=lambda x: -x[1])


def main():
    if not nim.available():
        print("[no NVIDIA_API_KEY] using lexical fallback for stage 1\n")

    print("=== Retrieval Accuracy Gap ===")
    print(f"Query: {QUERY!r}\n")

    ranked, method = bi_encoder_rank(QUERY, DOCS)
    top4 = [i for i, _ in ranked[:4]]
    print(f"Stage 1 — bi-encoder recall ({method}) -> top 4:")
    for rank, (i, s) in enumerate(ranked[:4], 1):
        print(f"  {rank}. [{s:.3f}] {DOCS[i]}")

    reranked = rerank(QUERY, top4)
    top2 = [i for i, _ in reranked[:2]]
    print("\nStage 2 — cross-encoder-style rerank -> top 2:")
    for rank, (i, s) in enumerate(reranked[:2], 1):
        print(f"  {rank}. [score {s}] {DOCS[i]}")

    print(
        "\nTakeaway: stage 1 casts a wide, cheap net; stage 2 precisely reorders. "
        "The best answer often sits at rank 3-4 after stage 1 and is promoted by rerank."
    )


if __name__ == "__main__":
    main()

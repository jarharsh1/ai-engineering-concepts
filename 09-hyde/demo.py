"""HyDE — Hypothetical Document Embeddings.

Short questions embed poorly: "how long do refunds take?" shares few words and
little semantic surface with the answer passage. HyDE's trick: ask the LLM to
*write a hypothetical answer* to the question, then embed THAT and search with it.
A fake-but-plausible answer looks much more like the real passage than the bare
question does, so retrieval improves.

This demo:
  1. generates a hypothetical answer with NIM (canned fallback if no key)
  2. embeds the raw question AND the hypothetical answer
  3. compares retrieval ranking for each over toy docs

Key recommended (works offline with a canned hypothesis + lexical fallback).
"""
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import nim  # noqa: E402

QUESTION = "how long do refunds take?"
CANNED_HYPOTHESIS = (
    "Refunds are typically processed back to your original payment method within "
    "five business days after the return is approved."
)
DOCS = [
    "Refunds are issued to the original payment method within 5 business days of approval.",
    "Our developer conference happens every September in Berlin.",
    "Reset your password from the login screen using 'Forgot password'.",
    "Standard shipping takes 3 to 7 business days depending on destination.",
]
GOLD = 0  # index of the correct doc


def _tok(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def cosine(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def rank_by_embedding(text):
    if nim.available():
        try:
            q = nim.embed([text], input_type="query")[0]
            ds = nim.embed(DOCS, input_type="passage")
            scored = [(i, cosine(q, d)) for i, d in enumerate(ds)]
            return sorted(scored, key=lambda x: -x[1])
        except Exception as e:
            print(f"[NIM embed failed: {e}] lexical fallback")
    qt = _tok(text)
    scored = [(i, len(qt & _tok(d)) / (len(qt | _tok(d)) or 1)) for i, d in enumerate(DOCS)]
    return sorted(scored, key=lambda x: -x[1])


def gold_rank(ranking):
    for pos, (i, _) in enumerate(ranking, 1):
        if i == GOLD:
            return pos
    return len(ranking)


def main():
    print("=== HyDE (Hypothetical Document Embeddings) ===")
    print(f"Question: {QUESTION!r}\n")

    if nim.available():
        try:
            hypo = nim.chat(
                f"Write a one-sentence plausible answer to: {QUESTION}",
                system="You answer concisely as if from a help-center article.",
                max_tokens=80,
            )
        except Exception as e:
            print(f"[NIM chat failed: {e}] using canned hypothesis")
            hypo = CANNED_HYPOTHESIS
    else:
        print("[no NVIDIA_API_KEY] using canned hypothetical answer + lexical fallback\n")
        hypo = CANNED_HYPOTHESIS

    print(f"Hypothetical answer: {hypo}\n")

    r_q = rank_by_embedding(QUESTION)
    r_h = rank_by_embedding(hypo)

    print("Ranking by RAW QUESTION embedding:")
    for pos, (i, s) in enumerate(r_q, 1):
        mark = " <-- correct" if i == GOLD else ""
        print(f"  {pos}. [{s:.3f}] {DOCS[i]}{mark}")
    print(f"  gold doc rank: {gold_rank(r_q)}")

    print("\nRanking by HyDE (hypothetical-answer) embedding:")
    for pos, (i, s) in enumerate(r_h, 1):
        mark = " <-- correct" if i == GOLD else ""
        print(f"  {pos}. [{s:.3f}] {DOCS[i]}{mark}")
    print(f"  gold doc rank: {gold_rank(r_h)}")

    print(
        "\nTakeaway: searching with a hypothetical answer (which resembles real "
        "passages) usually ranks the correct document at or above where the bare "
        "question does — especially for short, keyword-poor queries."
    )


if __name__ == "__main__":
    main()

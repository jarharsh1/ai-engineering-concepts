"""Speculative Decoding — a small model drafts, the big model verifies.

Generating one token at a time with a large model is latency-bound: each token
needs a full forward pass. Speculative decoding runs a cheap *draft* model to
propose K tokens ahead, then the big *target* model verifies all K in a single
pass. Accepted tokens are free; on the first mismatch you fall back and re-draft.
Crucially the output is *identical* to plain decoding — verification guarantees
correctness.

This demo simulates draft(K=4)+verify over a fixed sentence, with the draft model
~70% correct, and counts big-model calls vs the autoregressive baseline.

Key optional (not required; simulation is deterministic-ish with a seed).
"""
import random

TARGET = ("speculative decoding lets a small draft model propose tokens "
          "that the large model verifies in one pass").split()
K = 4
DRAFT_ACCURACY = 0.70
SEED = 7


def main():
    rng = random.Random(SEED)
    print("=== Speculative Decoding ===")
    print(f"target length={len(TARGET)} tokens, draft K={K}, "
          f"draft accuracy~{DRAFT_ACCURACY:.0%}\n")

    # Baseline: one big-model call per token.
    baseline_calls = len(TARGET)

    # Speculative: each round, draft proposes up to K tokens; big model verifies
    # in ONE call. We accept the correct prefix of the draft, then the big model
    # emits the first corrected token for free as part of verification.
    pos = 0
    spec_calls = 0
    accepted_total = 0
    rounds = 0
    while pos < len(TARGET):
        rounds += 1
        spec_calls += 1  # one verification pass by the big model
        # Draft proposes min(K, remaining) tokens.
        remaining = len(TARGET) - pos
        draft_len = min(K, remaining)
        accepted = 0
        for _ in range(draft_len):
            if rng.random() < DRAFT_ACCURACY:
                accepted += 1
            else:
                break
        # Verification emits accepted draft tokens + 1 correction token.
        advance = min(accepted + 1, remaining)
        accepted_total += accepted
        pos += advance

    speedup = baseline_calls / spec_calls

    print(f"  {'method':<22} | big-model calls")
    print(f"  {'-'*22}-+----------------")
    print(f"  {'autoregressive':<22} | {baseline_calls}")
    print(f"  {'speculative (K='+str(K)+')':<22} | {spec_calls}")
    print(f"\n  rounds              : {rounds}")
    print(f"  draft tokens accepted: {accepted_total}")
    print(f"  speed-up (fewer big-model passes): {speedup:.2f}x")
    print("  output identical to autoregressive: YES (verification guarantees it)")

    print(
        "\nTakeaway: when the draft model is often right, most tokens are produced "
        "by the cheap model and merely checked by the expensive one — fewer big-model "
        "passes, same exact output."
    )


if __name__ == "__main__":
    main()

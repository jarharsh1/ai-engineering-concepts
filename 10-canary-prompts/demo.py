"""Canary Deployments for Prompts — ship prompt changes like code.

A new prompt version is a production change with unknown blast radius. Rather than
flip 100% of traffic, route a small slice (say 5%) to the canary, measure a
quality signal on both arms, and let a gate decide: promote the canary to 100%,
or roll back.

This demo simulates a 95/5 split between stable v2.2 and canary v2.3, measures a
per-request quality score, and prints a promote/rollback decision.

Key optional (simulation is self-contained).
"""
import random

N_REQUESTS = 2000
CANARY_SHARE = 0.05
SEED = 42

# Simulated per-request quality signal (e.g. thumbs-up rate / eval score 0..1).
STABLE_MEAN = 0.78
CANARY_MEAN = 0.83          # the canary is actually a bit better
NOISE = 0.10
MIN_CANARY_N = 30           # need enough samples to trust the signal
PROMOTE_MARGIN = 0.01       # canary must beat stable by this much


def quality(mean, rng):
    return max(0.0, min(1.0, rng.gauss(mean, NOISE)))


def main():
    rng = random.Random(SEED)
    print("=== Canary Deployments for Prompts ===")
    print(f"stable=v2.2, canary=v2.3, split={1-CANARY_SHARE:.0%}/{CANARY_SHARE:.0%}, "
          f"{N_REQUESTS} requests\n")

    stable_scores, canary_scores = [], []
    for _ in range(N_REQUESTS):
        if rng.random() < CANARY_SHARE:
            canary_scores.append(quality(CANARY_MEAN, rng))
        else:
            stable_scores.append(quality(STABLE_MEAN, rng))

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    s_avg, c_avg = avg(stable_scores), avg(canary_scores)
    print(f"  stable v2.2 : n={len(stable_scores):4d}  avg quality={s_avg:.3f}")
    print(f"  canary v2.3 : n={len(canary_scores):4d}  avg quality={c_avg:.3f}")
    print(f"  delta       : {c_avg - s_avg:+.3f}\n")

    # Gate.
    if len(canary_scores) < MIN_CANARY_N:
        decision = "HOLD — not enough canary samples yet"
    elif c_avg >= s_avg + PROMOTE_MARGIN:
        decision = "PROMOTE canary v2.3 to 100% traffic"
    elif c_avg < s_avg - PROMOTE_MARGIN:
        decision = "ROLLBACK — canary underperforms"
    else:
        decision = "HOLD — within noise, keep observing"

    print(f"Gate decision: {decision}")
    print(
        "\nTakeaway: a canary bounds the blast radius of a prompt change. You only "
        "promote on a measured quality win with enough samples; otherwise you roll "
        "back automatically."
    )


if __name__ == "__main__":
    main()

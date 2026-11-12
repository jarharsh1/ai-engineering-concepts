"""Differential Privacy — a provable bound on what one person's data reveals.

DP adds calibrated noise so that the output of a query is almost unchanged
whether or not any single individual is in the dataset. Formally, the presence of
one person can shift the output distribution by at most a factor controlled by
epsilon: small epsilon = strong privacy (more noise), large epsilon = weak
privacy (less noise).

This demo uses the Laplace mechanism on a private MEAN over ~500 values and shows
the released mean barely changes whether or not one outlier ("Alice") is
included, across epsilon = 10, 1, 0.1.

No API key needed.
"""
import numpy as np

SEED = 1
N = 500
LOW, HIGH = 0.0, 100.0          # value range -> sensitivity of the mean


def private_mean(values, epsilon, rng):
    """Laplace mechanism for a bounded mean.

    Sensitivity of a mean over n values each in [LOW, HIGH] to adding/removing
    one record is (HIGH - LOW) / n.
    """
    n = len(values)
    true_mean = float(np.mean(values))
    sensitivity = (HIGH - LOW) / n
    scale = sensitivity / epsilon
    noise = rng.laplace(0.0, scale)
    return true_mean, true_mean + noise


def main():
    rng = np.random.default_rng(SEED)
    base = rng.uniform(LOW, HIGH, size=N - 1)
    alice = np.array([HIGH])                       # an extreme outlier
    with_alice = np.concatenate([base, alice])
    without_alice = base

    print("=== Differential Privacy (Laplace mechanism on a mean) ===")
    print(f"n={N} values in [{LOW:.0f},{HIGH:.0f}], Alice is an outlier at {HIGH:.0f}\n")

    print(f"  {'epsilon':>7} | {'released (with Alice)':>21} | "
          f"{'released (without)':>19} | {'|diff|':>7}")
    print(f"  {'-'*7}-+-{'-'*21}-+-{'-'*19}-+-{'-'*7}")
    for eps in [10.0, 1.0, 0.1]:
        # Use the same noise draw conceptually by reseeding per epsilon for a
        # fair side-by-side.
        r1 = np.random.default_rng(100)
        r2 = np.random.default_rng(100)
        _, out_with = private_mean(with_alice, eps, r1)
        _, out_without = private_mean(without_alice, eps, r2)
        diff = abs(out_with - out_without)
        print(f"  {eps:>7.1f} | {out_with:>21.4f} | {out_without:>19.4f} | {diff:>7.4f}")

    true_with = float(np.mean(with_alice))
    true_without = float(np.mean(without_alice))
    print(f"\n  (true means: with Alice={true_with:.4f}, without={true_without:.4f})")

    print(
        "\nTakeaway: the released mean is nearly the same whether or not Alice is in "
        "the data, so an observer can't confidently tell. Smaller epsilon adds more "
        "noise -> stronger privacy but noisier answers; larger epsilon is the reverse."
    )


if __name__ == "__main__":
    main()

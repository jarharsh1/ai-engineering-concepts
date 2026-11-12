"""Differential Privacy — a provable bound on what one person's data reveals.

DP adds calibrated noise so that the output of a query is almost unchanged
whether or not any single individual is in the dataset. Formally, the presence of
one person can shift the output distribution by at most a factor controlled by
epsilon: small epsilon = strong privacy (more noise), large epsilon = weak
privacy (less noise).

This demo uses the Laplace mechanism on a private MEAN over ~500 values. Over many
independent noise draws it shows how the DISTRIBUTION of the released mean (with vs
without one outlier, "Alice") overlaps as epsilon shrinks: at small epsilon the
noise spread dwarfs Alice's effect, so the two are indistinguishable. Across
epsilon = 10, 1, 0.1 you can watch the noise std grow and Alice go from exposed
to hidden.

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

    true_with = float(np.mean(with_alice))
    true_without = float(np.mean(without_alice))
    alice_effect = abs(true_with - true_without)

    print("=== Differential Privacy (Laplace mechanism on a mean) ===")
    print(f"n={N} values in [{LOW:.0f},{HIGH:.0f}], Alice is an outlier at {HIGH:.0f}")
    print(f"true mean with Alice={true_with:.4f}, without={true_without:.4f} "
          f"(Alice shifts it by only {alice_effect:.4f})\n")

    # The honest demonstration: draw INDEPENDENT noise many times for each
    # dataset and look at the DISTRIBUTION of released values. DP works when the
    # noise spread swamps Alice's effect, so the two output distributions overlap
    # and an attacker can't tell which dataset produced a given release.
    TRIALS = 20000
    print(f"Each cell is the mean +/- std of the released value over {TRIALS} "
          f"independent noise draws.\n")
    print(f"  {'epsilon':>7} | {'noise std':>9} | {'released (with Alice)':>22} | "
          f"{'released (without)':>22} | overlap?")
    print(f"  {'-'*7}-+-{'-'*9}-+-{'-'*22}-+-{'-'*22}-+---------")
    for eps in [10.0, 1.0, 0.1]:
        r1 = np.random.default_rng(100 + int(eps * 10))
        r2 = np.random.default_rng(900 + int(eps * 10))
        outs_with = np.array([private_mean(with_alice, eps, r1)[1] for _ in range(TRIALS)])
        outs_without = np.array([private_mean(without_alice, eps, r2)[1] for _ in range(TRIALS)])
        # Laplace std = sqrt(2) * scale = sqrt(2) * sensitivity / eps.
        noise_std = np.sqrt(2) * ((HIGH - LOW) / N) / eps
        # Alice is "hidden" once the noise spread dwarfs her effect on the mean.
        hidden = noise_std > alice_effect
        tag = "YES (hidden)" if hidden else "no (exposed)"
        print(f"  {eps:>7.1f} | {noise_std:>9.4f} | "
              f"{outs_with.mean():>8.4f} +/- {outs_with.std():>7.4f} | "
              f"{outs_without.mean():>8.4f} +/- {outs_without.std():>7.4f} | {tag}")

    print(
        "\nTakeaway: smaller epsilon -> larger Laplace noise std. Once the noise "
        "spread dwarfs Alice's tiny effect on the mean, the with/without output "
        "distributions overlap almost entirely, so an observer cannot confidently "
        "tell whether Alice was in the data. That overlap IS the privacy guarantee; "
        "the cost is a noisier (less accurate) released answer."
    )


if __name__ == "__main__":
    main()

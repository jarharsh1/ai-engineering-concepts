"""Calibration — does "90% confident" actually mean right 90% of the time?

A model is *calibrated* when its stated confidence matches its empirical
accuracy: among predictions it makes at 0.9 confidence, ~90% should be correct.
Overconfident models say 0.95 and are right 0.70 of the time — dangerous when you
gate actions on confidence.

This demo builds a numpy reliability diagram and computes ECE (Expected
Calibration Error) for a well-calibrated vs an overconfident model.

No API key needed.
"""
import numpy as np

N = 4000
BINS = 10
SEED = 0


def make_data(overconfident: bool, rng):
    """Return (confidences, correct_flags)."""
    # True probability of being correct for each sample.
    p_true = rng.uniform(0.0, 1.0, size=N)
    correct = (rng.uniform(size=N) < p_true).astype(int)
    if overconfident:
        # Stated confidence is pushed toward 1 — claims more than it delivers.
        conf = np.clip(p_true + 0.25, 0.0, 1.0)
    else:
        conf = p_true  # perfectly calibrated: stated == true
    return conf, correct


def reliability(conf, correct):
    """Per-bin (count, avg_confidence, accuracy)."""
    edges = np.linspace(0, 1, BINS + 1)
    rows = []
    ece = 0.0
    for b in range(BINS):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, hi, 0, None, None))
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (n / len(conf)) * abs(acc - avg_conf)
        rows.append((lo, hi, n, avg_conf, acc))
    return rows, ece


def print_diagram(name, rows, ece):
    print(f"\n{name}")
    print(f"  {'band':<12} | {'n':>5} | {'avg_conf':>8} | {'accuracy':>8}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}")
    for lo, hi, n, avg_conf, acc in rows:
        if n == 0:
            print(f"  {lo:.1f}-{hi:.1f}      | {n:>5} | {'—':>8} | {'—':>8}")
        else:
            print(f"  {lo:.1f}-{hi:.1f}      | {n:>5} | {avg_conf:>8.3f} | {acc:>8.3f}")
    print(f"  ECE = {ece:.4f}")


def main():
    rng = np.random.default_rng(SEED)
    print("=== Calibration (reliability diagram + ECE) ===")

    conf_c, corr_c = make_data(overconfident=False, rng=np.random.default_rng(SEED))
    rows_c, ece_c = reliability(conf_c, corr_c)
    print_diagram("Well-calibrated model:", rows_c, ece_c)

    conf_o, corr_o = make_data(overconfident=True, rng=np.random.default_rng(SEED))
    rows_o, ece_o = reliability(conf_o, corr_o)
    print_diagram("Overconfident model:", rows_o, ece_o)

    print(
        "\nTakeaway: the calibrated model's accuracy tracks its confidence (low ECE). "
        "The overconfident model's accuracy lags its confidence and it never emits low "
        "confidence (— bands), inflating ECE. Calibrate before gating on confidence."
    )


if __name__ == "__main__":
    main()

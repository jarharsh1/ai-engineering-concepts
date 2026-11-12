"""Conformal Prediction — prediction SETS with a coverage guarantee.

Instead of a single class guess, split conformal prediction outputs a *set* of
classes that contains the true label with a user-chosen frequency (here 90%).
It's distribution-free: you only assume the calibration and test data are
exchangeable. The recipe:

  1. split data into train / calibrate / test
  2. train a classifier
  3. nonconformity score on calibration = 1 - p(true class)
  4. threshold q = the (1-alpha) empirical quantile of those scores
  5. prediction set = { classes whose 1 - p(class) <= q }  (i.e. p(class) >= 1-q)

Over many test points the sets cover the truth ~1-alpha of the time *in
expectation* — a finite-sample property, not an absolute per-point guarantee.

No API key needed (uses sklearn's iris).
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ALPHA = 0.10
SEED = 0


def main():
    print("=== Conformal Prediction (split conformal) ===")
    X, y = load_iris(return_X_y=True)
    # Use only the two sepal features: the classes overlap heavily there, so the
    # classifier is genuinely uncertain near boundaries. That makes conformal
    # sets interesting (some have >1 class) instead of trivially perfect.
    X = X[:, :2]

    # train (~1/3) / calibrate (~1/3) / test (~1/3)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.67, random_state=1, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=SEED, stratify=y_rest
    )

    # A softmax classifier gives smooth, non-saturated probabilities, so the
    # nonconformity scores spread out and the conformal threshold is meaningful.
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)

    # Nonconformity on calibration set = 1 - p(true class).
    cal_probs = clf.predict_proba(X_cal)
    cal_scores = 1.0 - cal_probs[np.arange(len(y_cal)), y_cal]

    # Conformal quantile with finite-sample correction.
    n = len(cal_scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    q = np.quantile(cal_scores, q_level, method="higher")

    print(f"alpha={ALPHA} (target coverage {1-ALPHA:.0%}), "
          f"calib n={n}, threshold q={q:.3f}\n")

    # Build prediction sets on the test set: include class c if 1 - p(c) <= q.
    test_probs = clf.predict_proba(X_test)
    include = (1.0 - test_probs) <= q           # (n_test, n_classes) boolean

    covered = include[np.arange(len(y_test)), y_test]
    set_sizes = include.sum(axis=1)

    coverage = covered.mean()
    avg_size = set_sizes.mean()
    abstain = (set_sizes == 0).mean()       # empty set = "abstain / unsure"

    print(f"  test points        : {len(y_test)}")
    print(f"  empirical coverage : {coverage:.3f}  (target {1-ALPHA:.2f})")
    print(f"  avg set size       : {avg_size:.3f}")
    print(f"  abstention rate    : {abstain:.3f}  (empty sets)\n")

    # Show a few example prediction sets.
    classes = [str(c) for c in load_iris().target_names]
    print("  sample prediction sets:")
    for i in range(min(8, len(y_test))):
        s = [classes[c] for c in range(include.shape[1]) if include[i, c]]
        print(f"    true={classes[y_test[i]]:<11} set={s}")

    print(
        "\nTakeaway: conformal prediction turns any classifier into one that outputs "
        "calibrated sets. Coverage near 1-alpha holds as a finite-sample expectation "
        "over exchangeable data, not as an absolute guarantee for every single point."
    )


if __name__ == "__main__":
    main()

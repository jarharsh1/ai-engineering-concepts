"""Saga — distributed transactions without a distributed lock.

You can't hold an ACID transaction across independent services (search, payments,
inventory...). A Saga instead runs a sequence of local steps, each with a
*compensating action* that undoes it. If step N fails, you run the compensations
for steps N-1 ... 1 in reverse order, leaving the system consistent.

This demo runs a booking saga:
  search -> reserve -> charge -> confirm -> update
Each forward step has a compensation. We show the happy path, then a failure at
step 4 (confirm) that triggers reverse-order compensation.

No API key needed.
"""


class Step:
    def __init__(self, name, do, undo):
        self.name = name
        self.do = do
        self.undo = undo


class Saga:
    def __init__(self, steps):
        self.steps = steps

    def run(self, fail_at=None):
        completed = []
        print("  forward:")
        for idx, step in enumerate(self.steps, 1):
            if fail_at == idx:
                print(f"    {idx}. {step.name} ... FAILED")
                print("  compensating (reverse order):")
                for done in reversed(completed):
                    done.undo()
                    print(f"    undo {done.name}")
                print("  -> system left consistent (all prior steps rolled back)")
                return False
            step.do()
            completed.append(step)
            print(f"    {idx}. {step.name} ... ok")
        print("  -> saga committed (all steps succeeded)")
        return True


def build_saga():
    state = {}
    steps = [
        Step("search",  lambda: state.update(search="done"),   lambda: state.pop("search", None)),
        Step("reserve", lambda: state.update(reserve="held"),  lambda: state.pop("reserve", None)),
        Step("charge",  lambda: state.update(charge="paid"),   lambda: state.pop("charge", None)),
        Step("confirm", lambda: state.update(confirm="sent"),  lambda: state.pop("confirm", None)),
        Step("update",  lambda: state.update(update="written"), lambda: state.pop("update", None)),
    ]
    return Saga(steps)


def main():
    print("=== Saga ===")
    print("Booking workflow: search -> reserve -> charge -> confirm -> update\n")

    print("Happy path:")
    build_saga().run()

    print("\nFailure path (confirm fails at step 4):")
    build_saga().run(fail_at=4)

    print(
        "\nTakeaway: a Saga swaps one big distributed transaction for a chain of "
        "local ones plus compensations. On failure it unwinds in reverse, so no "
        "money stays charged for a booking that never confirmed."
    )


if __name__ == "__main__":
    main()

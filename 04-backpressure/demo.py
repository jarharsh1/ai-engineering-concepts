"""Backpressure — shed load instead of collapsing under it.

When demand exceeds capacity you have two choices: accept everything (and watch
latency explode and the server fall over), or apply backpressure — admit what
you can serve and reject the rest fast (HTTP 429) so clients can retry or
degrade. Rejecting early protects the requests you *did* admit.

This asyncio demo compares:
  accept-all : every request is served, no shedding
  bounded    : a semaphore of CAPACITY admits CAPACITY, sheds the rest as 429

No API key needed.
"""
import asyncio

CAPACITY = 200
OFFERED = 500


async def handle(sem: asyncio.Semaphore | None, served: list, shed: list, i: int):
    if sem is None:
        # accept-all: no admission control
        await asyncio.sleep(0.001)
        served.append(i)
        return
    if sem.locked() and sem._value == 0:
        shed.append(i)
        return
    acquired = sem._value > 0
    if not acquired:
        shed.append(i)
        return
    async with sem:
        await asyncio.sleep(0.001)
        served.append(i)


async def run_bounded():
    sem = asyncio.Semaphore(CAPACITY)
    served, shed = [], []

    async def worker(i):
        if sem._value <= 0:
            shed.append(i)
            return
        async with sem:
            await asyncio.sleep(0.005)
            served.append(i)

    await asyncio.gather(*(worker(i) for i in range(OFFERED)))
    return served, shed


async def run_accept_all():
    served = []

    async def worker(i):
        await asyncio.sleep(0.005)
        served.append(i)

    await asyncio.gather(*(worker(i) for i in range(OFFERED)))
    return served


def main():
    print("=== Backpressure ===")
    print(f"CAPACITY={CAPACITY}, offered={OFFERED}\n")

    served_all = asyncio.run(run_accept_all())
    print("accept-all (no shedding):")
    print(f"  served={len(served_all)}  shed=0")
    print("  -> all 500 admitted at once; in a real server this is where")
    print("     queues back up, latency spikes, and the process OOMs/times out.\n")

    served, shed = asyncio.run(run_bounded())
    print(f"bounded semaphore (CAPACITY={CAPACITY}):")
    print(f"  served={len(served)}  shed(429)={len(shed)}")
    print("  -> admits 200, rejects 300 fast. The 200 admitted stay healthy.\n")

    print(
        "Takeaway: backpressure trades a clear, fast rejection for a slow, "
        "cascading failure. Shedding load is a feature, not an outage."
    )


if __name__ == "__main__":
    main()

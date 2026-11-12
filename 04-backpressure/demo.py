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


async def run_bounded():
    """Admit up to CAPACITY concurrent requests; shed the rest immediately.

    Each offered request tries to grab a slot. `Semaphore.locked()` is True when
    no slots are free, so we reject (429) without waiting — that fast rejection
    is the backpressure. Admitted requests hold a slot while they do their work.
    """
    sem = asyncio.Semaphore(CAPACITY)
    served, shed = [], []

    async def worker(i):
        if sem.locked():          # all CAPACITY slots currently taken
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

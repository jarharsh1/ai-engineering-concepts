"""Bulkhead — isolate failures so one bad dependency can't sink the ship.

Named after a ship's watertight compartments: if one floods, the others keep the
vessel afloat. In software, a bulkhead gives each downstream dependency its own
resource pool (threads, connections, concurrency slots). When one dependency
hangs, only its pool is exhausted — callers to healthy dependencies are
unaffected.

This asyncio demo contrasts:
  shared pool      : one semaphore for all services; a slow service starves others
  per-service pools : each service has its own semaphore; the slow one is contained

No API key needed.
"""
import asyncio
import time

SERVICES = ["search", "payments", "profile"]
SLOW = "payments"            # this dependency is hanging
POOL = 4                     # concurrency slots
CALLS_PER_SERVICE = 6


async def call_service(name, delay):
    await asyncio.sleep(delay)
    return name


async def run_shared():
    sem = asyncio.Semaphore(POOL)
    latencies = {s: [] for s in SERVICES}

    async def do(name):
        t0 = time.perf_counter()
        async with sem:  # all services share POOL slots
            delay = 0.30 if name == SLOW else 0.01
            await call_service(name, delay)
        latencies[name].append(time.perf_counter() - t0)

    tasks = [do(s) for s in SERVICES for _ in range(CALLS_PER_SERVICE)]
    await asyncio.gather(*tasks)
    return latencies


async def run_bulkhead():
    sems = {s: asyncio.Semaphore(POOL) for s in SERVICES}
    latencies = {s: [] for s in SERVICES}

    async def do(name):
        t0 = time.perf_counter()
        async with sems[name]:  # each service has its own pool
            delay = 0.30 if name == SLOW else 0.01
            await call_service(name, delay)
        latencies[name].append(time.perf_counter() - t0)

    tasks = [do(s) for s in SERVICES for _ in range(CALLS_PER_SERVICE)]
    await asyncio.gather(*tasks)
    return latencies


def summarize(latencies):
    return {s: 1000 * sum(v) / len(v) for s, v in latencies.items()}


def main():
    print("=== Bulkhead ===")
    print(f"'{SLOW}' is the failing/slow dependency (300ms); others are 10ms.\n")

    shared = summarize(asyncio.run(run_shared()))
    print("Shared pool (one semaphore for all):")
    for s in SERVICES:
        flag = "  <-- healthy service dragged down" if s != SLOW and shared[s] > 50 else ""
        print(f"  {s:<9} avg latency: {shared[s]:6.1f} ms{flag}")

    bulk = summarize(asyncio.run(run_bulkhead()))
    print("\nPer-service bulkheads (one semaphore each):")
    for s in SERVICES:
        flag = "  <-- stays fast" if s != SLOW else "  <-- only this one is slow"
        print(f"  {s:<9} avg latency: {bulk[s]:6.1f} ms{flag}")

    print(
        "\nTakeaway: with a shared pool the slow dependency's queue backs up every "
        "caller. With bulkheads the blast radius is contained to the failing service."
    )


if __name__ == "__main__":
    main()

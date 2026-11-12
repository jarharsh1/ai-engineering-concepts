"""CQRS — separate the write path from the read path.

Command Query Responsibility Segregation splits a system's writes (commands) from
its reads (queries) so each can be optimized independently. A common failure
without it: heavy ingestion holds a lock (or the same hot table) and queries
stall behind it. With a dedicated read view, queries serve from a structure that
writers don't block.

This threaded demo contrasts:
  naive  : single lock; a long ingestion batch blocks every query
  cqrs   : writes land in a staging buffer; queries read an immutable snapshot

No API key needed.
"""
import threading
import time

N_QUERIES = 20


def run_naive():
    lock = threading.Lock()
    store = {"rows": 0}
    latencies = []

    def ingest():
        with lock:  # holds the lock for the whole batch
            for _ in range(100):
                store["rows"] += 1
                time.sleep(0.002)  # simulate write work (~200ms total)

    def query():
        t0 = time.perf_counter()
        with lock:  # must wait for ingestion to release
            _ = store["rows"]
        latencies.append(time.perf_counter() - t0)

    w = threading.Thread(target=ingest)
    w.start()
    time.sleep(0.01)  # let ingestion grab the lock first
    qs = [threading.Thread(target=query) for _ in range(N_QUERIES)]
    for q in qs:
        q.start()
    for q in qs:
        q.join()
    w.join()
    return latencies


def run_cqrs():
    # Read view is an immutable snapshot swapped in atomically; queries never
    # block on the writer.
    read_view = {"rows": 0}
    latencies = []

    def ingest():
        staged = read_view["rows"]
        for _ in range(100):
            staged += 1
            time.sleep(0.002)
            # Publish snapshot periodically (atomic dict-key assignment).
            read_view["rows"] = staged

    def query():
        t0 = time.perf_counter()
        _ = read_view["rows"]  # lock-free read of current snapshot
        latencies.append(time.perf_counter() - t0)

    w = threading.Thread(target=ingest)
    w.start()
    time.sleep(0.01)
    qs = [threading.Thread(target=query) for _ in range(N_QUERIES)]
    for q in qs:
        q.start()
    for q in qs:
        q.join()
    w.join()
    return latencies


def main():
    print("=== CQRS (Command Query Responsibility Segregation) ===")
    print(f"Running {N_QUERIES} queries during a heavy ingestion batch.\n")

    naive = run_naive()
    cqrs = run_cqrs()

    def ms(xs):
        return 1000 * sum(xs) / len(xs)

    print(f"  {'design':<28} | avg query latency during ingest")
    print(f"  {'-'*28}-+--------------------------------")
    print(f"  {'naive single-lock store':<28} | {ms(naive):8.1f} ms")
    print(f"  {'CQRS split read-view':<28} | {ms(cqrs):8.1f} ms")

    print(
        "\nTakeaway: the naive store serializes queries behind the write lock, so "
        "query latency tracks ingestion time. CQRS lets reads serve from a snapshot, "
        "staying fast no matter how heavy the write load."
    )


if __name__ == "__main__":
    main()

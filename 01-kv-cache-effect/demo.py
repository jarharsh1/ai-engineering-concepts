"""KV Cache Effect — why long contexts blow up GPU memory.

During autoregressive decoding, a transformer caches the Key and Value tensors
for every past token so it doesn't recompute them. That cache grows linearly
with sequence length AND with the number of concurrent users. It often becomes
the binding constraint on how many users a single GPU can serve — long before
compute does.

    kv_bytes_per_token = 2 (K and V) * layers * hidden * dtype_bytes

No API key needed — this is pure arithmetic.
"""

# Model / hardware assumptions (Llama-3-8B-ish on a 40GB GPU).
LAYERS = 32
HIDDEN = 4096
DTYPE_BYTES = 2          # fp16
SEQ_LEN = 2048
GPU_BYTES = 40 * 1024**3  # 40 GB
WEIGHTS_BYTES = 16 * 1024**3  # 16 GB resident for weights


def kv_bytes_per_token() -> int:
    return 2 * LAYERS * HIDDEN * DTYPE_BYTES


def main() -> None:
    per_tok = kv_bytes_per_token()
    per_user = per_tok * SEQ_LEN
    kv_budget = GPU_BYTES - WEIGHTS_BYTES
    max_users = kv_budget // per_user

    print("=== KV Cache Effect ===")
    print(f"Assumptions: layers={LAYERS}, hidden={HIDDEN}, fp16, seq_len={SEQ_LEN}")
    print(f"GPU={GPU_BYTES/1024**3:.0f}GB, weights={WEIGHTS_BYTES/1024**3:.0f}GB\n")

    print(f"KV bytes/token     : {per_tok:,} bytes ({per_tok/1024:.0f} KB)")
    print(f"KV per user @ {SEQ_LEN} : {per_user/1024**2:.1f} MB")
    print(f"KV memory budget   : {kv_budget/1024**3:.1f} GB")
    print(f"Max concurrent users: {max_users}\n")

    print("Concurrency vs memory (OK -> OOM):")
    print(f"  {'users':>6} | {'kv_used_GB':>10} | status")
    print(f"  {'-'*6}-+-{'-'*10}-+-------")
    for users in [8, 16, 32, 64, 80, 96, 128]:
        used = users * per_user
        status = "OK" if used <= kv_budget else "OOM  <-- crash"
        print(f"  {users:>6} | {used/1024**3:>10.1f} | {status}")

    print(
        "\nTakeaway: doubling context or doubling users both double KV memory. "
        "Serving capacity is gated by KV cache, not FLOPs."
    )


if __name__ == "__main__":
    main()

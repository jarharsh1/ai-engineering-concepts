"""Prefix Caching — stop re-encoding the same long preamble.

Most production prompts share a huge, identical prefix: the system prompt, tool
definitions, few-shot examples, retrieved boilerplate. Prefix caching stores the
KV cache for that shared prefix once and reuses it across requests, so each new
request only pays to encode its *unique* suffix.

This is a token-accounting estimator (no model, no key):
  prefix tokens  = 2000   (shared system/tools/examples)
  unique tokens  = 50     (the actual user turn)
  requests/hour  = 1000
"""

PREFIX = 2000
UNIQUE = 50
REQ_PER_HOUR = 1000


def main():
    print("=== Prefix Caching ===")
    print(f"prefix={PREFIX} tokens, unique={UNIQUE} tokens, {REQ_PER_HOUR} req/hour\n")

    without = (PREFIX + UNIQUE) * REQ_PER_HOUR
    # With caching, the prefix is encoded once (first request / cache fill),
    # then only unique tokens are encoded per request.
    with_cache = PREFIX + UNIQUE * REQ_PER_HOUR
    saved = without - with_cache
    pct = saved / without * 100

    print(f"  {'mode':<22} | {'prefill tokens/hour':>20}")
    print(f"  {'-'*22}-+-{'-'*20}")
    print(f"  {'without prefix cache':<22} | {without:>20,}")
    print(f"  {'with prefix cache':<22} | {with_cache:>20,}")
    print(f"\n  tokens saved/hour : {saved:,}")
    print(f"  reduction         : {pct:.1f}%")

    print(
        "\nTakeaway: when the shared prefix dwarfs the unique part (2000 vs 50 here), "
        "prefix caching eliminates ~97% of prefill work. The longer and more shared "
        "your preamble, the bigger the win."
    )


if __name__ == "__main__":
    main()

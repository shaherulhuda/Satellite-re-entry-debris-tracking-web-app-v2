"""Derived architecture metrics behind the Session 6 model recommendation.

Usage: python session6_metrics.py session6_model_candidates.csv

Formulas follow session6_formula_reference.csv. Assumes bf16/fp16 KV cache
(2 bytes per element) and batch size 1. Standard library only.
"""
import csv
import sys

BYTES_PER_ELEM = 2
GIB = 1024**3
CONTEXTS = (8192, 32768, 131072)


def derive(row):
    L = int(row["num_layers"])
    d = int(row["hidden_size"])
    ff = int(row["intermediate_size"])
    n_heads = int(row["num_attention_heads"])
    n_kv = int(row["num_key_value_heads"])
    params_b = float(row["params_total_b"])

    head_dim = d / n_heads
    kv_per_token = 2 * L * n_kv * head_dim * BYTES_PER_ELEM
    attn = 2 * d**2 + 2 * d * (n_kv * head_dim)  # per layer
    ffn = 3 * d * ff  # per layer, SwiGLU-style

    return {
        "name": row["model_name"],
        "max_ctx": int(row["context_length_tokens"]),
        "head_dim": head_dim,
        "gqa_group": n_heads / n_kv,
        "gflops_per_token": 2 * params_b,
        "kv_bytes_per_token": kv_per_token,
        "kv_gib": {T: kv_per_token * T / GIB for T in CONTEXTS},
        "attn_params_m": attn / 1e6,
        "ffn_params_m": ffn / 1e6,
        "ffn_share": ffn / (ffn + attn),
    }


def main(path):
    with open(path, newline="") as f:
        models = [derive(r) for r in csv.DictReader(f) if r["model_id"]]

    for m in models:
        print(f"\n{m['name']}  (max context {m['max_ctx'] // 1024}K)")
        print(f"  head_dim={m['head_dim']:.0f}  GQA group={m['gqa_group']:.0f}"
              f"  ~{m['gflops_per_token']:.1f} GFLOPs/token")
        print(f"  KV cache: {m['kv_bytes_per_token'] / 1024:.0f} KiB/token")
        for T, g in m["kv_gib"].items():
            note = "" if T <= m["max_ctx"] else "  (beyond model's max context)"
            print(f"    {T // 1024:>4}K: {g:6.2f} GiB{note}")
        print(f"  attn={m['attn_params_m']:.1f}M  ffn={m['ffn_params_m']:.1f}M"
              f" per layer  (FFN {m['ffn_share']:.1%})")

    # Ranking for S6_MAIN: models that cover long context, ordered by cache cost.
    long_ctx = [m for m in models if m["max_ctx"] >= 131072]
    best = min(long_ctx, key=lambda m: m["kv_bytes_per_token"])
    worst = max(long_ctx, key=lambda m: m["kv_bytes_per_token"])
    saving = 1 - best["kv_bytes_per_token"] / worst["kv_bytes_per_token"]
    print(f"\n128K-capable model with smallest KV cache: {best['name']} "
          f"({best['kv_gib'][131072]:.1f} GiB vs {worst['kv_gib'][131072]:.1f} GiB "
          f"for {worst['name']}, {saving:.0%} less)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "session6_model_candidates.csv")

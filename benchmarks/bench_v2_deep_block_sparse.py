"""
benchmarks/bench_v2_deep_block_sparse.py
Milestone 2.4 Benchmark: Block-Sparse Elastic Capsules (B_block=16), Depth-Scaled
Residual Highway (alpha_res / sqrt(L)), Highway-Anchored Deep Hebbian DFA, and
12-Layer Deep Scaling on CPU (Zero-Autograd / Frozen W_0 Only).

Compares 4 architectural & depth configurations on the 25,000+ token English
TinyStories corpus (90% Train / 10% Validation):
  1. 4-Layer Element-wise Sparse (v2.1/v2.3 baseline scalar Top-K)
  2. 4-Layer Block-Sparse Capsules (B_block=16, Active-Slice Execution)
  3. 12-Layer Deep Naive (Unscaled Residual / Random Unanchored DFA)
  4. 12-Layer Deep SS-ENN v2.4 (Block-Sparse + Depth-Scaled Highway + Anchored DFA)

Outputs 3 comprehensive telemetry tables:
  - Table 1: Architecture & Depth Scaling Comparison (Train/Val Acc, PPL, tok/s, MB, L12/L1)
  - Table 2: 12-Layer Depth Profile L1..L12 (Active Capsules, Pre/Post RMS, k_l, Entropy)
  - Table 3: 12-Layer Block-Sparse Free Rollout +30 Tokens Audit (Frozen W_0 Only)

Strictly zero autograd / zero backward().
"""

import sys
import math
import time
import pathlib
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import EnglishSubwordTokenizer
from models import ScaledSpringNetwork
from benchmarks.english_dataset import build_english_tinystories_corpus, CANONICAL_SEED_PROMPTS
from benchmarks.bench_scale_english_cpu import (
    verify_all_repo_files_zero_autograd,
    calculate_entropy,
    build_minibatch_tensors,
    evaluate_batches,
)
from benchmarks.bench_v2_sleep_continual import audit_rollout_quality_and_consistency


def _train_and_benchmark_variant(
    name: str,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    train_seqs: list[list[int]],
    tokenizer: EnglishSubwordTokenizer,
    train_tokens_count: int,
    val_tokens_count: int,
    epochs: int = 12,
    seq_len: int = 36,
    seed: int = 42,
    model_kwargs: dict | None = None,
) -> dict:
    """Trains a single SS-ENN variant, freezes to W_0, and benchmarks throughput & stability."""
    kwargs = {
        "preset": "1.5M",
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": seq_len + 8,
        "eta_w": 0.14,
        "eta_0": 0.045,
        "gamma": 0.05,
        "beta_k": 0.022,
        "eta_b": 0.015,
        "enable_neurogenesis": True,
        "saturation_threshold": 0.80,
        "stiffness_threshold": 0.08,
        "error_threshold": 0.005,
        "cooldown_epochs": 2,
        "seed": seed,
    }
    if model_kwargs:
        kwargs.update(model_kwargs)

    model = ScaledSpringNetwork(**kwargs)
    model.calibrate_salience_from_tokens(
        train_seqs,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
    )

    total_train_sec = 0.0
    total_train_tokens = 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        ep_mse = 0.0
        ep_correct = 0
        ep_valid = 0
        for inp, tgt in train_batches:
            stats = model.train_minibatch(inp, tgt, pad_id=tokenizer.pad_id)
            n_val = stats["valid_tokens"]
            ep_mse += stats["mse_loss"] * n_val
            ep_correct += stats["correct"]
            ep_valid += n_val
        train_mse = ep_mse / max(ep_valid, 1)
        tok_err = 1.0 - (ep_correct / max(float(ep_valid), 1.0))
        model.step_neurogenesis(epoch=epoch, mean_epoch_error=train_mse, token_error_rate=tok_err)
        t1 = time.perf_counter()
        total_train_sec += (t1 - t0)
        total_train_tokens += ep_valid

    train_tok_per_sec = total_train_tokens / max(total_train_sec, 1e-6)

    # Snapshot stiffness & training participation entropy BEFORE production_freeze discards k
    layer_profiles_pre = []
    for l_idx, layer in enumerate(model.spring_layers):
        d_act = layer.d_active
        d_eff = model._effective_context_dim()
        k_bar = float(layer.k[:d_act, :d_eff].mean().item())
        if model.enable_block_sparse:
            n_blocks = max(1, d_act // model.block_size)
            k_blocks = (
                model.hierarchical_top_k_blocks[l_idx]
                if model.enable_hierarchical_compositional
                else max(1, min(n_blocks, int(round(model.top_k / float(model.block_size)))))
            )
            last_wins = model._last_epoch_block_wins[l_idx][:n_blocks]
            win_src = last_wins if float(last_wins.sum().item()) > 0.0 else model.block_win_counts[l_idx][:n_blocks]
            train_ent = calculate_entropy(win_src)
        else:
            n_blocks = d_act
            k_blocks = layer.top_k or d_act
            train_ent = calculate_entropy(layer.win_counts[:d_act])
        layer_profiles_pre.append({
            "layer": l_idx + 1,
            "d_active": d_act,
            "n_blocks": n_blocks,
            "active_capsules": k_blocks,
            "mean_k": k_bar,
            "train_entropy": train_ent,
        })

    mean_k_all = sum(p["mean_k"] for p in layer_profiles_pre) / max(len(layer_profiles_pre), 1)
    mean_train_ent = sum(p["train_entropy"] for p in layer_profiles_pre) / max(len(layer_profiles_pre), 1)

    # Freeze to W_0 only (destroys w, k, and B^(l))
    freeze_stats = model.production_freeze(prune_dormant=True)
    breakdown = model.count_parameters_breakdown()

    if model.enable_block_sparse:
        model.reset_frozen_block_stats()

    t_eval0 = time.perf_counter()
    frozen_train = evaluate_batches(model, train_batches, pad_id=tokenizer.pad_id)
    frozen_val = evaluate_batches(model, val_batches, pad_id=tokenizer.pad_id)
    t_eval1 = time.perf_counter()
    eval_tok_per_sec = (train_tokens_count + val_tokens_count) / max(t_eval1 - t_eval0, 1e-6)

    # Run clean single validation RMS / capsule profile pass on Frozen W_0
    pre_rms_acc = [0.0 for _ in range(model.num_layers)]
    post_rms_acc = [0.0 for _ in range(model.num_layers)]
    for inp, _ in val_batches:
        _ = model(inp)
        for l_idx in range(model.num_layers):
            pre_rms_acc[l_idx] += model._last_layer_raw_rms[l_idx]
            post_rms_acc[l_idx] += model._last_layer_normed_rms[l_idx]
    n_val_b = max(len(val_batches), 1)
    pre_rms_list = [v / n_val_b for v in pre_rms_acc]
    post_rms_list = [v / n_val_b for v in post_rms_acc]

    ratio_post_end_to_l1 = post_rms_list[-1] / max(post_rms_list[0], 1e-6)
    ratio_pre_end_to_l1 = pre_rms_list[-1] / max(pre_rms_list[0], 1e-6)

    layer_profiles = []
    for l_idx, pre_info in enumerate(layer_profiles_pre):
        if model.enable_block_sparse and model._frozen_block_wins is not None:
            n_b = pre_info["n_blocks"]
            frozen_ent = calculate_entropy(model._frozen_block_wins[l_idx][:n_b])
        else:
            frozen_ent = pre_info["train_entropy"]
        layer_profiles.append({
            **pre_info,
            "pre_rms": pre_rms_list[l_idx],
            "post_rms": post_rms_list[l_idx],
            "ratio_post_l1": post_rms_list[l_idx] / max(post_rms_list[0], 1e-6),
            "frozen_entropy": frozen_ent,
        })

    mean_frozen_ent = sum(p["frozen_entropy"] for p in layer_profiles) / max(len(layer_profiles), 1)

    return {
        "name": name,
        "num_layers": model.num_layers,
        "model": model,
        "frozen_train": frozen_train,
        "frozen_val": frozen_val,
        "train_tok_per_sec": train_tok_per_sec,
        "eval_tok_per_sec": eval_tok_per_sec,
        "frozen_mb": breakdown["frozen_footprint_mb"],
        "frozen_params": breakdown["frozen_w0_params"],
        "active_dims": freeze_stats["active_dims"],
        "mean_k": mean_k_all,
        "mean_train_ent": mean_train_ent,
        "mean_frozen_ent": mean_frozen_ent,
        "ratio_post_end_l1": ratio_post_end_to_l1,
        "ratio_pre_end_l1": ratio_pre_end_to_l1,
        "layer_profiles": layer_profiles,
    }


@torch.no_grad()
def run_v2_deep_block_sparse_benchmark(
    epochs: int = 12,
    batch_size: int = 32,
    seq_len: int = 36,
    verbose: bool = True,
) -> dict:
    """
    Executes Milestone 2.4 benchmark comparing Element-wise vs Block-Sparse Capsules
    and 4-Layer vs 12-Layer Deep Scaling on Frozen W_0 Only.
    """
    torch.set_grad_enabled(False)
    ast_ok, n_files = verify_all_repo_files_zero_autograd()

    corpus_info = build_english_tinystories_corpus(target_min_tokens=25000, train_ratio=0.90, seed=42)
    train_stories = corpus_info["train_stories"]
    val_stories = corpus_info["val_stories"]

    tokenizer = EnglishSubwordTokenizer(max_vocab_size=2048)
    tokenizer.fit(train_stories + val_stories, pad_to_max_vocab=True)

    train_batches, train_seqs, train_tokens_count = build_minibatch_tensors(
        train_stories, tokenizer, seq_len=seq_len, batch_size=batch_size
    )
    val_batches, _, val_tokens_count = build_minibatch_tensors(
        val_stories, tokenizer, seq_len=seq_len, batch_size=batch_size
    )

    if verbose:
        print("=" * 108)
        print("SS-ENN v2.0 (Milestone 2.4): Block-Sparse Elastic Capsules (B=16) & 12-Layer Deep Scaling Benchmark")
        print("=" * 108)
        print(f"AST Zero-Autograd Verification : {'PASS' if ast_ok else 'FAIL'} ({n_files} Python files scanned, 0 backward/grad)")
        print(
            f"English Corpus Split           : Train={len(train_stories)} ({train_tokens_count:,} tok) | "
            f"Val={len(val_stories)} ({val_tokens_count:,} tok) | V={tokenizer.vocab_size:,}"
        )

    variants_spec = [
        (
            "4L Element-wise Sparse",
            {
                "num_layers": 4,
                "enable_block_sparse": False,
                "enable_hierarchical_compositional": True,
                "use_role_bridge": False,
            },
        ),
        (
            "4L Block-Sparse (B=16)",
            {
                "num_layers": 4,
                "enable_block_sparse": True,
                "block_size": 16,
                "use_depth_scaled_highway": True,
                "use_highway_anchored_dfa": True,
            },
        ),
        (
            "12L Deep Naive (Unscaled/Rand)",
            {
                "num_layers": 12,
                "enable_block_sparse": True,
                "block_size": 16,
                "enable_hierarchical_compositional": False,
                "use_depth_scaled_highway": False,
                "use_highway_anchored_dfa": False,
                "eta_b": 0.0,
            },
        ),
        (
            "12L Deep SS-ENN v2.4 (B=16)",
            {
                "num_layers": 12,
                "enable_block_sparse": True,
                "block_size": 16,
                "use_depth_scaled_highway": True,
                "use_highway_anchored_dfa": True,
            },
        ),
    ]

    variant_results = []
    for label, v_kwargs in variants_spec:
        res = _train_and_benchmark_variant(
            name=label,
            train_batches=train_batches,
            val_batches=val_batches,
            train_seqs=train_seqs,
            tokenizer=tokenizer,
            train_tokens_count=train_tokens_count,
            val_tokens_count=val_tokens_count,
            epochs=epochs,
            seq_len=seq_len,
            seed=42,
            model_kwargs=v_kwargs,
        )
        variant_results.append(res)

    elem_4l = variant_results[0]
    block_4l = variant_results[1]
    naive_12l = variant_results[2]
    deep_12l = variant_results[3]

    # Audit +30-token free rollout on Frozen W_0 of 12-Layer Deep SS-ENN v2.4
    rollout_audit = audit_rollout_quality_and_consistency(
        deep_12l["model"], tokenizer, CANONICAL_SEED_PROMPTS, max_new_tokens=30
    )

    if verbose:
        print("\n--- Table 1: Architecture & Depth Scaling Comparison (Frozen W_0 Only) ---")
        print(
            f"{'Architecture':<32} | {'Train%':>6} | {'Val%':>6} | {'ValPPL':>6} | "
            f"{'Train(tok/s)':>12} | {'Eval(tok/s)':>11} | {'W0(MB)':>6} | {'Mean k':>6} | {'L_N/L1':>6}"
        )
        print("-" * 108)
        for r in variant_results:
            print(
                f"{r['name']:<32} | "
                f"{r['frozen_train']['acc']:5.2f}% | "
                f"{r['frozen_val']['acc']:5.2f}% | "
                f"{r['frozen_val']['ppl']:6.2f} | "
                f"{r['train_tok_per_sec']:12,.0f} | "
                f"{r['eval_tok_per_sec']:11,.0f} | "
                f"{r['frozen_mb']:6.2f} | "
                f"{r['mean_k']:6.4f} | "
                f"{r['ratio_post_end_l1']:6.4f}"
            )

        train_speedup = block_4l["train_tok_per_sec"] / max(elem_4l["train_tok_per_sec"], 1e-6)
        eval_speedup = block_4l["eval_tok_per_sec"] / max(elem_4l["eval_tok_per_sec"], 1e-6)
        print(
            f"Block-Sparse 4L Speedup vs Element-wise 4L : "
            f"Train {train_speedup:.2f}x | Frozen Eval {eval_speedup:.2f}x"
        )

        print("\n--- Table 2: 12-Layer Deep SS-ENN v2.4 Layer-Wise Profile L1..L12 (Frozen W_0) ---")
        print(
            f"Overall 12L Entropy (Train / Frozen) : {deep_12l['mean_train_ent']:.4f} / {deep_12l['mean_frozen_ent']:.4f} | "
            f"Variance Ratio L12/L1 (Post / Pre): {deep_12l['ratio_post_end_l1']:.4f} / {deep_12l['ratio_pre_end_l1']:.4f}"
        )
        print("-" * 108)
        print(
            f"{'Layer':<6} | {'Active Dims':>11} | {'Capsules(K/N)':>13} | {'Pre-RMS':>8} | "
            f"{'Post-RMS':>8} | {'L_l/L1':>7} | {'Mean k_l':>8} | {'Caps Ent':>8}"
        )
        print("-" * 108)
        for p in deep_12l["layer_profiles"]:
            cap_str = f"{p['active_capsules']:>2} / {p['n_blocks']:<2} (x16)"
            print(
                f"L{p['layer']:<5} | "
                f"{p['d_active']:>11} | "
                f"{cap_str:>13} | "
                f"{p['pre_rms']:8.4f} | "
                f"{p['post_rms']:8.4f} | "
                f"{p['ratio_post_l1']:7.4f} | "
                f"{p['mean_k']:8.4f} | "
                f"{p['frozen_entropy']:8.4f}"
            )

        print("\n--- Table 3: 12-Layer Block-Sparse +30-Token Free Rollout Audit (Frozen W_0 Only) ---")
        print(
            f"Entity Consistency: {rollout_audit['entity_consistency_pct']:.2f}% "
            f"({rollout_audit['consistent_count']}/{len(CANONICAL_SEED_PROMPTS)}) | "
            f"Adjacent Dup: {rollout_audit['adjacent_dup_rate_pct']:.2f}% | "
            f"Phrase Loops: {rollout_audit['phrase_loop_rate_pct']:.2f}%"
        )
        print("-" * 108)
        for idx, rec in enumerate(rollout_audit["records"], start=1):
            status = "PASS (NO DRIFT)" if rec["consistent"] else "DRIFT"
            print(
                f"[Sample #{idx:02d}] Prompt: \"{rec['prompt']}\" | "
                f"{rec['sent1_entity']} -> ({rec['bridge']}) -> {rec['sent2_entity']} [{status}] | "
                f"AdjDup={rec['adj_dups']} | Loop={rec['has_loop']}"
            )
            print(f"             Rollout: {rec['generated']}")
        print("=" * 108)

    # Strip live model references for lighter structured test return if preferred, keeping key metrics
    summary_variants = {
        r["name"]: {k: v for k, v in r.items() if k != "model"}
        for r in variant_results
    }
    return {
        "ast_ok": ast_ok,
        "scanned_files": n_files,
        "variants": summary_variants,
        "elem_4l": summary_variants[elem_4l["name"]],
        "block_4l": summary_variants[block_4l["name"]],
        "naive_12l": summary_variants[naive_12l["name"]],
        "deep_12l": summary_variants[deep_12l["name"]],
        "rollout_audit": rollout_audit,
        "deep_12l_model": deep_12l["model"],
    }


if __name__ == "__main__":
    run_v2_deep_block_sparse_benchmark(verbose=True)

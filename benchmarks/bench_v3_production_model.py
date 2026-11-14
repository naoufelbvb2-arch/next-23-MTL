"""
benchmarks/bench_v3_production_model.py
Phase 3.2 Benchmark: Explicit Manual Configuration (`SSENNConfig`), Zero-Allocation
Memory Estimator (`1.5M` -> `7B` Scale Audit), Clean Unified Production Model
(`SSENNProductionModel`), Physical Dormant Pruning, O(1) Generation Parity, and
100% Bitwise Mid-Training Checkpoint Save/Resume Verification.

Outputs 3 Telemetry Tables:
  - Table 1: Manual Scaling & Pre-Allocation Footprint Estimator Audit (No Tensor Alloc)
  - Table 2: End-to-End Training, Frozen W0 Compaction, O(1) Speed & BF16 Kahan Sanity
  - Table 3: Mid-Training Checkpoint Save/Resume Bitwise Parity Audit (FP32 & BF16)

Strictly zero autograd / zero backward().
"""

from __future__ import annotations

import pathlib
import sys
import time
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import EnglishSubwordTokenizer
from models import SSENNConfig, SSENNProductionModel
from benchmarks.english_dataset import CANONICAL_SEED_PROMPTS, build_english_tinystories_corpus
from benchmarks.bench_scale_english_cpu import (
    build_minibatch_tensors,
    evaluate_batches,
    verify_all_repo_files_zero_autograd,
)
from benchmarks.bench_v2_sleep_continual import audit_rollout_quality_and_consistency


def _max_state_abs_diff(model_a: SSENNProductionModel, model_b: SSENNProductionModel) -> float:
    max_diff = 0.0
    bufs_a = dict(model_a.named_buffers())
    bufs_b = dict(model_b.named_buffers())
    for k, va in bufs_a.items():
        vb = bufs_b.get(k)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            return float("inf")
        if va.dtype == torch.bool:
            diff = 1.0 if not torch.equal(va, vb) else 0.0
        else:
            diff = float((va.float() - vb.float()).abs().max().item())
        if diff > max_diff:
            max_diff = diff
    return max_diff


def _train_model_epochs(
    model: SSENNProductionModel,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
    start_epoch: int,
    end_epoch: int,
    shuffle_seed_base: int = 1234,
) -> dict:
    total_toks = 0
    t0 = time.perf_counter()
    last_train_acc = 0.0
    last_train_ce = 0.0
    for ep in range(start_epoch, end_epoch + 1):
        g = torch.Generator(device="cpu")
        g.manual_seed(shuffle_seed_base + ep)
        order = torch.randperm(len(train_batches), generator=g).tolist()
        ep_correct = 0
        ep_valid = 0
        ep_ce_sum = 0.0
        for b_idx in order:
            x_b, y_b = train_batches[b_idx]
            st = model.train_minibatch(x_b, y_b, pad_id=pad_id)
            ep_correct += int(st["correct"])
            ep_valid += int(st["valid_tokens"])
            ep_ce_sum += float(st["ce_loss"]) * max(int(st["valid_tokens"]), 1)
            total_toks += int(st["valid_tokens"])
        last_train_ce = ep_ce_sum / max(ep_valid, 1)
        last_train_acc = (ep_correct / max(ep_valid, 1)) * 100.0
        model.end_epoch(epoch=ep, mean_ce_loss=last_train_ce)

    elapsed = max(time.perf_counter() - t0, 1e-6)
    val_eval = evaluate_batches(model, val_batches, pad_id=pad_id)
    return {
        "train_acc_pct": last_train_acc,
        "train_ce": last_train_ce,
        "val_acc_pct": float(val_eval["acc"]),
        "val_ppl": float(val_eval["ppl"]),
        "elapsed_s": elapsed,
        "tok_s": float(total_toks) / elapsed,
    }


@torch.no_grad()
def run_v3_production_benchmark(
    epochs: int = 16,
    batch_size: int = 32,
    seq_len: int = 36,
    verbose: bool = True,
) -> dict:
    torch.set_grad_enabled(False)

    ast_ok, n_files = verify_all_repo_files_zero_autograd()
    if verbose:
        print("=" * 112)
        print("SS-ENN v3.0 PHASE 3.2 BENCHMARK: MANUAL CONFIG, PRODUCTION MODEL, ZERO-ALLOC ESTIMATOR & CHECKPOINT PARITY")
        print("=" * 112)
        print(f"[Gate Check] AST Zero-Autograd Verification : {'PASS' if ast_ok else 'FAIL'} ({n_files} Python files verified)")

    # =========================================================================
    # TABLE 1: Manual Scaling & Pre-Allocation Memory Footprint Audit
    # =========================================================================
    scale_configs: list[tuple[str, SSENNConfig]] = [
        (
            "Small CPU (6L d=64->128 FP32)",
            SSENNConfig(
                vocab_size=2048, num_layers=6, d_init=64, d_max=128, delta_d=16,
                block_size=16, top_k_blocks=2, max_seq_len=64, attn_top_k=8,
                dtype="float32", device="cpu",
            ),
        ),
        (
            "Edge-25M (8L d=256->512 BF16)",
            SSENNConfig(
                vocab_size=8192, num_layers=8, d_init=256, d_max=512, delta_d=64,
                block_size=16, top_k_blocks=4, max_seq_len=512, attn_top_k=8,
                dtype="bfloat16", device="cpu",
            ),
        ),
        (
            "Base-125M (12L d=512->1024 BF16)",
            SSENNConfig(
                vocab_size=16384, num_layers=12, d_init=512, d_max=1024, delta_d=128,
                block_size=16, top_k_blocks=8, max_seq_len=1024, attn_top_k=12,
                dtype="bfloat16", device="cuda",
            ),
        ),
        (
            "Medium-1.3B (24L d=1536->3072 BF16)",
            SSENNConfig(
                vocab_size=32000, num_layers=24, d_init=1536, d_max=3072, delta_d=256,
                block_size=16, top_k_blocks=16, max_seq_len=2048, attn_top_k=16,
                dtype="bfloat16", device="cuda",
            ),
        ),
        (
            "Large-7B (32L d=3072->6656 BF16)",
            SSENNConfig(
                vocab_size=32000, num_layers=32, d_init=3072, d_max=6656, delta_d=512,
                block_size=16, top_k_blocks=32, max_seq_len=2048, attn_top_k=16,
                dtype="bfloat16", device="xla",
            ),
        ),
        (
            "Large-7B (anchor_k -33.3% Kahan)",
            SSENNConfig(
                vocab_size=32000, num_layers=32, d_init=3072, d_max=6656, delta_d=512,
                block_size=16, top_k_blocks=32, max_seq_len=2048, attn_top_k=16,
                dtype="bfloat16", device="xla", kahan_mode="anchor_k",
            ),
        ),
    ]

    table1_rows = []
    for label, s_cfg in scale_configs:
        est = s_cfg.estimate_memory_footprint()
        table1_rows.append({"label": label, "config": s_cfg, "footprint": est})

    if verbose:
        print("\n[Table 1] Manual Scaling & Pre-Allocation Footprint Estimator Audit (Zero Tensor Allocation)")
        print("-" * 112)
        print(
            f"{'Manual Architecture Scale':<35} | {'d_init->d_max':<13} | {'Max Train':<11} | "
            f"{'DFA Savings':<11} | {'Kahan FP32':<10} | {'Train+Kahan':<11} | {'Frozen W0'}"
        )
        print("-" * 112)
        for row in table1_rows:
            fp = row["footprint"]
            cfg_r = row["config"]
            p_str = f"{fp['max_train_params'] / 1e9:.2f}B" if fp["max_train_params"] >= 1e9 else f"{fp['max_train_params'] / 1e6:.2f}M"
            tk_str = f"{fp['total_train_with_kahan_gb']:.2f} GB" if fp["total_train_with_kahan_mb"] >= 1024 else f"{fp['total_train_with_kahan_mb']:.1f} MB"
            w0_str = f"{fp['frozen_w0_gb']:.2f} GB" if fp["frozen_w0_mb"] >= 1024 else f"{fp['frozen_w0_mb']:.1f} MB"
            kah_str = f"{fp['kahan_max_overhead_mb']:.1f} MB" if fp["kahan_max_overhead_mb"] < 1024 else f"{fp['kahan_max_overhead_mb'] / 1024:.2f} GB"
            print(
                f"{row['label']:<35} | {f'{cfg_r.d_init}->{cfg_r.d_max}':<13} | {p_str:<11} | "
                f"{fp['dfa_memory_savings_pct']:>9.2f}% | {kah_str:<10} | {tk_str:<11} | {w0_str}"
            )
        print("-" * 112)

    # =========================================================================
    # Corpus Preparation
    # =========================================================================
    corpus = build_english_tinystories_corpus(target_min_tokens=20000, train_ratio=0.90, seed=42)
    all_stories = corpus["train_stories"] + corpus["val_stories"]
    tokenizer = EnglishSubwordTokenizer(max_vocab_size=2048)
    tokenizer.fit(all_stories, pad_to_max_vocab=True)
    pad_id = tokenizer.pad_id

    train_batches, train_seqs, _ = build_minibatch_tensors(
        corpus["train_stories"], tokenizer, seq_len=seq_len, batch_size=batch_size
    )
    val_batches, _, _ = build_minibatch_tensors(
        corpus["val_stories"], tokenizer, seq_len=seq_len, batch_size=batch_size
    )

    # =========================================================================
    # TABLE 2: SSENNProductionModel Training, Freezing & O(1) vs O(T^2) Audit
    # =========================================================================
    prod_cfg = SSENNConfig(
        vocab_size=tokenizer.vocab_size,
        num_layers=6,
        d_init=64,
        d_max=128,
        delta_d=16,
        block_size=16,
        top_k_blocks=2,
        max_seq_len=64,
        attn_top_k=8,
        salient_anchor_slots=2,
        device="cpu",
        dtype="float32",
        seed=42,
    )
    prod_model = SSENNProductionModel(prod_cfg)
    prod_model.calibrate_salience_from_tokens(train_seqs, pad_id=pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)

    train_stats = _train_model_epochs(
        prod_model, train_batches, val_batches, pad_id=pad_id, start_epoch=1, end_epoch=epochs
    )
    pre_freeze_train_eval = evaluate_batches(prod_model, train_batches, pad_id=pad_id)
    pre_freeze_val_eval = evaluate_batches(prod_model, val_batches, pad_id=pad_id)

    freeze_info = prod_model.production_freeze(prune_dormant=True)
    frozen_train_eval = evaluate_batches(prod_model, train_batches, pad_id=pad_id)
    frozen_val_eval = evaluate_batches(prod_model, val_batches, pad_id=pad_id)

    # Export Frozen W0 only & reconstruct minimal deployment model cleanly
    frozen_pkg = prod_model.export_frozen_w0_state()
    reloaded_frozen = SSENNProductionModel.from_frozen_w0_state(frozen_pkg)
    sample_x = val_batches[0][0][:4]
    reloaded_logit_diff = float((prod_model.forward(sample_x) - reloaded_frozen.forward(sample_x)).abs().max().item())

    # Compare O(1) Stateful Cache vs Naive O(T^2) 30-token Generation
    prompt_ids_list = [tokenizer.encode(p, add_bos=True, add_eos=False) for p in CANONICAL_SEED_PROMPTS]
    fast_outs, slow_outs = [], []
    fast_new_toks, slow_new_toks = 0, 0

    t_fast = time.perf_counter()
    for p_ids in prompt_ids_list:
        g_fast = prod_model.generate(p_ids, max_new_tokens=30, eos_id=tokenizer.eos_id, use_fast_cache=True)
        fast_outs.append(g_fast)
        fast_new_toks += max(0, len(g_fast) - len(p_ids))
    fast_time = max(time.perf_counter() - t_fast, 1e-6)

    t_slow = time.perf_counter()
    for p_ids in prompt_ids_list:
        g_slow = prod_model.generate(p_ids, max_new_tokens=30, eos_id=tokenizer.eos_id, use_fast_cache=False)
        slow_outs.append(g_slow)
        slow_new_toks += max(0, len(g_slow) - len(p_ids))
    slow_time = max(time.perf_counter() - t_slow, 1e-6)

    exact_gen_matches = sum(1 for a, b in zip(fast_outs, slow_outs) if a == b)
    gen_parity_pct = (exact_gen_matches / max(len(prompt_ids_list), 1)) * 100.0
    fast_gen_tok_s = float(fast_new_toks) / fast_time
    slow_gen_tok_s = float(slow_new_toks) / slow_time
    gen_speedup = fast_gen_tok_s / max(slow_gen_tok_s, 1e-6)

    rollout_audit = audit_rollout_quality_and_consistency(prod_model, tokenizer, CANONICAL_SEED_PROMPTS, max_new_tokens=30)

    # BF16 Kahan Sanity Run (CPU BF16 + FP32 Kahan Compensator)
    bf16_cfg = SSENNConfig(
        vocab_size=tokenizer.vocab_size, num_layers=4, d_init=64, d_max=64, delta_d=0,
        block_size=16, top_k_blocks=2, max_seq_len=64, attn_top_k=8,
        dtype="bfloat16", device="cpu", seed=77,
    )
    bf16_model = SSENNProductionModel(bf16_cfg)
    bf16_model.calibrate_salience_from_tokens(train_seqs[:64], pad_id=pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
    ep1_ce = bf16_model.train_minibatch(train_batches[0][0], train_batches[0][1], pad_id=pad_id)["ce_loss"]
    bf16_stats = _train_model_epochs(bf16_model, train_batches[:8], val_batches[:2], pad_id=pad_id, start_epoch=1, end_epoch=5)
    bf16_bd = bf16_model.count_parameters_breakdown()
    bf16_sample_logits = bf16_model.forward(prompt_ids_list[0])
    bf16_finite = bool(torch.isfinite(bf16_sample_logits).all().item())
    bf16_k_mean = float(bf16_model.spring_layers[0].k.float().mean().item())
    bf16_model.production_freeze(prune_dormant=True)
    bf16_frozen_gen = bf16_model.generate(prompt_ids_list[0], max_new_tokens=16, eos_id=tokenizer.eos_id)

    # BF16 anchor_k Sanity Run (-33.3% Kahan Compensator RAM: tracks w0 & k residuals only)
    bf16_ak_cfg = SSENNConfig(
        vocab_size=tokenizer.vocab_size, num_layers=4, d_init=64, d_max=64, delta_d=0,
        block_size=16, top_k_blocks=2, max_seq_len=64, attn_top_k=8,
        dtype="bfloat16", device="cpu", kahan_mode="anchor_k", seed=77,
    )
    bf16_ak_model = SSENNProductionModel(bf16_ak_cfg)
    bf16_ak_model.calibrate_salience_from_tokens(train_seqs[:64], pad_id=pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
    ak_ep1_ce = bf16_ak_model.train_minibatch(train_batches[0][0], train_batches[0][1], pad_id=pad_id)["ce_loss"]
    bf16_ak_stats = _train_model_epochs(bf16_ak_model, train_batches[:8], val_batches[:2], pad_id=pad_id, start_epoch=1, end_epoch=5)
    bf16_ak_bd = bf16_ak_model.count_parameters_breakdown()
    bf16_ak_finite = bool(torch.isfinite(bf16_ak_model.forward(prompt_ids_list[0])).all().item())
    bf16_ak_res_w_count = len(bf16_ak_model.compensator._res_w) if bf16_ak_model.compensator is not None else -1
    kahan_anchor_k_saved_pct = (
        (1.0 - float(bf16_ak_bd["kahan_residual_mb"]) / max(float(bf16_bd["kahan_residual_mb"]), 1e-9)) * 100.0
    )

    table2 = {
        "train_tok_s": train_stats["tok_s"],
        "pre_freeze_train_acc": float(pre_freeze_train_eval["acc"]),
        "pre_freeze_val_acc": float(pre_freeze_val_eval["acc"]),
        "frozen_train_acc": float(frozen_train_eval["acc"]),
        "frozen_val_acc": float(frozen_val_eval["acc"]),
        "frozen_val_ppl": float(frozen_val_eval["ppl"]),
        "before_live_mb": float(freeze_info["before_live_mb"]),
        "after_live_mb": float(freeze_info["after_live_mb"]),
        "memory_saved_pct": float(freeze_info["memory_saved_pct"]),
        "compacted_dims": freeze_info["compacted_dims"],
        "reloaded_frozen_logit_diff": reloaded_logit_diff,
        "fast_gen_tok_s": fast_gen_tok_s,
        "slow_gen_tok_s": slow_gen_tok_s,
        "gen_speedup": gen_speedup,
        "gen_parity_pct": gen_parity_pct,
        "entity_consistency_pct": float(rollout_audit["entity_consistency_pct"]),
        "adjacent_dup_rate_pct": float(rollout_audit["adjacent_dup_rate_pct"]),
        "phrase_loop_rate_pct": float(rollout_audit["phrase_loop_rate_pct"]),
        "bf16_ep1_ce": ep1_ce,
        "bf16_final_ce": bf16_stats["train_ce"],
        "bf16_kahan_mb": float(bf16_bd["kahan_residual_mb"]),
        "bf16_ak_final_ce": float(bf16_ak_stats["train_ce"]),
        "bf16_ak_kahan_mb": float(bf16_ak_bd["kahan_residual_mb"]),
        "bf16_ak_saved_pct": float(kahan_anchor_k_saved_pct),
        "bf16_ak_res_w_count": int(bf16_ak_res_w_count),
        "bf16_ak_finite": bf16_ak_finite,
        "bf16_mean_k": bf16_k_mean,
        "bf16_finite": bf16_finite,
        "bf16_gen_len": len(bf16_frozen_gen),
    }

    if verbose:
        print("\n[Table 2] SSENNProductionModel End-to-End Training, Freezing, O(1) Parity & BF16 Kahan Sanity")
        print("-" * 112)
        print(f"  * FP32 Train Throughput  : {table2['train_tok_s']:,.1f} tok/s | Pre-Freeze Val Acc: {table2['pre_freeze_val_acc']:.2f}%")
        print(
            f"  * Frozen W0 (Compacted)  : Train Acc {table2['frozen_train_acc']:.2f}% | "
            f"Val Acc {table2['frozen_val_acc']:.2f}% | Val PPL {table2['frozen_val_ppl']:.2f} | "
            f"Live MB {table2['before_live_mb']:.2f} -> {table2['after_live_mb']:.2f} (-{table2['memory_saved_pct']:.1f}%)"
        )
        print(
            f"  * Frozen W0 Reload Diff  : {table2['reloaded_frozen_logit_diff']:.2e} | "
            f"Compacted Dims: {table2['compacted_dims']}"
        )
        print(
            f"  * O(1) vs O(T^2) Rollout : O(1) {table2['fast_gen_tok_s']:,.1f} tok/s vs O(T^2) "
            f"{table2['slow_gen_tok_s']:,.1f} tok/s ({table2['gen_speedup']:.2f}x) | "
            f"Seq Match: {table2['gen_parity_pct']:.2f}%"
        )
        print(
            f"  * Rollout Linguistic Audit: Entity Consistency {table2['entity_consistency_pct']:.1f}% | "
            f"Adjacent Dups {table2['adjacent_dup_rate_pct']:.2f}% | Loops {table2['phrase_loop_rate_pct']:.1f}%"
        )
        print(
            f"  * BF16 Kahan Sanity (CPU): CE {table2['bf16_ep1_ce']:.3f} -> {table2['bf16_final_ce']:.3f} | "
            f"Kahan FP32 Buf {table2['bf16_kahan_mb']:.2f} MB | Mean k {table2['bf16_mean_k']:.4f} | "
            f"Finite={table2['bf16_finite']}"
        )
        print(
            f"  * BF16 anchor_k Mode     : CE -> {table2['bf16_ak_final_ce']:.3f} | "
            f"Kahan FP32 Buf {table2['bf16_ak_kahan_mb']:.2f} MB (-{table2['bf16_ak_saved_pct']:.1f}%) | "
            f"_res_w={table2['bf16_ak_res_w_count']} | Finite={table2['bf16_ak_finite']}"
        )
        print("-" * 112)

    # =========================================================================
    # TABLE 3: Mid-Training Checkpoint Save/Resume Bitwise Parity Audit
    # =========================================================================
    table3_rows = []
    for dt_name, total_eps, split_ep in (("float32", 6, 3), ("bfloat16", 4, 2)):
        cfg_c = SSENNConfig(
            vocab_size=tokenizer.vocab_size, num_layers=6, d_init=64, d_max=128, delta_d=16,
            block_size=16, top_k_blocks=2, max_seq_len=64, attn_top_k=8,
            dtype=dt_name, device="cpu", seed=99,
        )
        # Run A: continuous epochs 1..total_eps
        cont_model = SSENNProductionModel(cfg_c)
        cont_model.calibrate_salience_from_tokens(train_seqs[:64], pad_id=pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
        _train_model_epochs(cont_model, train_batches[:8], val_batches[:2], pad_id=pad_id, start_epoch=1, end_epoch=total_eps, shuffle_seed_base=500)

        # Run B: 1..split_ep -> checkpoint -> fresh model -> split_ep+1..total_eps
        part_model = SSENNProductionModel(cfg_c)
        part_model.calibrate_salience_from_tokens(train_seqs[:64], pad_id=pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
        _train_model_epochs(part_model, train_batches[:8], val_batches[:2], pad_id=pad_id, start_epoch=1, end_epoch=split_ep, shuffle_seed_base=500)
        ckpt = part_model.get_full_checkpoint_state()

        resumed_model = SSENNProductionModel.from_checkpoint_state(ckpt)
        _train_model_epochs(resumed_model, train_batches[:8], val_batches[:2], pad_id=pad_id, start_epoch=split_ep + 1, end_epoch=total_eps, shuffle_seed_base=500)

        pre_freeze_w_diff = _max_state_abs_diff(cont_model, resumed_model)
        cont_model.production_freeze(prune_dormant=True)
        resumed_model.production_freeze(prune_dormant=True)
        frozen_w_diff = _max_state_abs_diff(cont_model, resumed_model)
        logit_diff = float((cont_model.forward(sample_x) - resumed_model.forward(sample_x)).abs().max().item())

        roll_matches = 0
        for p_ids in prompt_ids_list:
            g_a = cont_model.generate(p_ids, max_new_tokens=30, eos_id=tokenizer.eos_id, use_fast_cache=True)
            g_b = resumed_model.generate(p_ids, max_new_tokens=30, eos_id=tokenizer.eos_id, use_fast_cache=True)
            if g_a == g_b:
                roll_matches += 1
        roll_pct = (roll_matches / max(len(prompt_ids_list), 1)) * 100.0

        table3_rows.append({
            "dtype": dt_name,
            "split": f"{split_ep} + {total_eps - split_ep} vs {total_eps}",
            "pre_freeze_weight_diff": pre_freeze_w_diff,
            "frozen_weight_diff": frozen_w_diff,
            "logit_diff": logit_diff,
            "rollout_parity_pct": roll_pct,
        })

    if verbose:
        print("\n[Table 3] Mid-Training Checkpoint Save/Resume Bitwise Parity Audit (Fresh Instance Reload)")
        print("-" * 112)
        print(f"{'DType':<10} | {'Schedule':<14} | {'Pre-Freeze W Diff':<18} | {'Frozen W0 Diff':<16} | {'Logit Diff':<14} | {'Rollout Parity'}")
        print("-" * 112)
        for r in table3_rows:
            print(
                f"{r['dtype']:<10} | {r['split']:<14} | {r['pre_freeze_weight_diff']:<18.2e} | "
                f"{r['frozen_weight_diff']:<16.2e} | {r['logit_diff']:<14.2e} | {r['rollout_parity_pct']:.2f}%"
            )
        print("-" * 112)

    return {
        "ast_zero_autograd": ast_ok,
        "ast_verified_files": n_files,
        "table1": table1_rows,
        "table2": table2,
        "table3": table3_rows,
    }


if __name__ == "__main__":
    run_v3_production_benchmark()

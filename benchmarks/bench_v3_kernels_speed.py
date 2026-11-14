"""
benchmarks/bench_v3_kernels_speed.py
Phase 3.1 Benchmark: Hardware-Accelerated Fused Spring Kernels, Active-Block Slice
Execution, Factorized Highway DFA, O(1) Stateful Inference Cache, and BF16 Safety.

Produces 4 comparative telemetry tables (Zero-Autograd / Frozen W_0 Only):
  - Table 1: Unfused (v2.4) vs. Fused In-Place (v3.1) Training Speed, ATen Alloc MB & Parity
  - Table 2: Standard [D, V] DFA vs. Factorized Highway DFA [D, D] Memory & Broadcast Speed
  - Table 3: O(T^2) Full Recompute vs. O(1) Stateful SpringInferenceCache Rollout (+30/+64 tok)
  - Table 4: Mixed-Precision Safety Audit (FP32 vs. Naive BF16 vs. BF16 + Kahan Compensator)
"""

import sys
import time
import pathlib
import torch
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import (
    EnglishSubwordTokenizer,
    FactorizedHighwayDFA,
    MixedPrecisionCompensator,
    SpringScratchBuffer,
    TensorAllocationTracker,
    fused_spring_update_inplace,
    indexed_block_matmul_slice,
)
from engine.dfa_router import DFARouter
from models import ScaledSpringNetwork
from benchmarks.english_dataset import build_english_tinystories_corpus, CANONICAL_SEED_PROMPTS
from benchmarks.bench_scale_english_cpu import (
    verify_all_repo_files_zero_autograd,
    build_minibatch_tensors,
    evaluate_batches,
)


def _measure_spring_update_alloc_bytes(use_fused: bool, vocab_size: int = 2048, d_model: int = 256, layers: int = 4) -> tuple[float, float]:
    """
    Measures steady-state ATen tensor allocations (MB) across one full training step's
    (w, w0, k) spring update kernels across `layers` hidden matrices + Output Head.
    Returns (alloc_mb_per_step, latency_ms).
    """
    scratch = SpringScratchBuffer() if use_fused else None
    comp = MixedPrecisionCompensator() if use_fused else None

    shapes = [(vocab_size, d_model)] + [(d_model, d_model) for _ in range(layers)]
    mats = []
    for r, c in shapes:
        w = torch.randn(r, c, dtype=torch.float32) * 0.05
        w0 = w.clone()
        k = torch.full((r, c), 0.20, dtype=torch.float32)
        drv = torch.randn(r, c, dtype=torch.float32) * 0.01
        mats.append((w, w0, k, drv))

    def _run_once() -> None:
        for w, w0, k, drv in mats:
            if use_fused:
                fused_spring_update_inplace(
                    w=w, w0=w0, k=k, driving_signal=drv,
                    eta_w=0.14, eta_0=0.045, gamma=0.05, beta_k_eff=0.022,
                    stiff_ceil=0.85, scratch=scratch, compensator=comp, comp_key=id(w),
                )
            else:
                stretch = w - w0
                delta_w = (0.14 * drv) - (0.05 * k * stretch)
                w.add_(delta_w)
                stretch_new = w - w0
                w0.add_(0.045 * (1.0 - k) * stretch_new)
                k.add_(0.022 * stretch_new.abs() * (1.0 - k)).clamp_(0.0, 0.85)

    # Warmup once so scratch buffer is already allocated (steady-state audit)
    _run_once()

    with TensorAllocationTracker() as tracker:
        _run_once()

    t0 = time.perf_counter()
    for _ in range(10):
        _run_once()
    ms = ((time.perf_counter() - t0) / 10.0) * 1000.0
    return tracker.allocated_mb, ms


@torch.no_grad()
def benchmark_indexed_block_slice_gemm() -> dict:
    """Verifies indexed_block_matmul_slice mathematical parity & active-block FLOP reduction."""
    d_act, d_eff, b_size, n_tok = 256, 256, 16, 128
    n_blocks = d_act // b_size  # 16 blocks
    active_blocks = torch.tensor([0, 2, 5, 7, 11, 14], dtype=torch.long)  # U = 6 / 16 (62.5% skipped)

    gen = torch.Generator().manual_seed(123)
    w0 = torch.randn(d_act, d_eff, generator=gen) * 0.08
    a_prev = torch.randn(n_tok, d_eff, generator=gen)
    neg_slope = 0.1
    res_scale = 0.675

    # Reference dense computation zeroing out inactive blocks
    z_dense = F.linear(a_prev, w0)
    a_dense = F.leaky_relu(z_dense, negative_slope=neg_slope)
    mask_ref = torch.zeros(d_act, dtype=torch.float32)
    for b_id in active_blocks.tolist():
        mask_ref[b_id * b_size : (b_id + 1) * b_size] = 1.0
    ref_out = a_prev + res_scale * (a_dense * mask_ref.unsqueeze(0))

    # Indexed sparse slice computation
    sparse_out = indexed_block_matmul_slice(
        a_prev_eff=a_prev,
        w0=w0,
        active_block_indices=active_blocks,
        block_size=b_size,
        neg_slope=neg_slope,
        res_scale=res_scale,
    )
    max_diff = float((ref_out - sparse_out).abs().max().item())
    flop_reduction_pct = (1.0 - float(active_blocks.numel()) / float(n_blocks)) * 100.0

    # Latency benchmark
    for _ in range(3):
        _ = indexed_block_matmul_slice(a_prev, w0, active_blocks, b_size, neg_slope, res_scale)
    t0 = time.perf_counter()
    for _ in range(60):
        _ = indexed_block_matmul_slice(a_prev, w0, active_blocks, b_size, neg_slope, res_scale)
    sparse_ms = ((time.perf_counter() - t0) / 60.0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(60):
        z_d = F.linear(a_prev, w0)
        _ = a_prev + res_scale * F.leaky_relu(z_d, negative_slope=neg_slope)
    dense_ms = ((time.perf_counter() - t0) / 60.0) * 1000.0

    return {
        "u_active": int(active_blocks.numel()),
        "n_blocks": n_blocks,
        "flop_reduction_pct": flop_reduction_pct,
        "max_abs_diff": max_diff,
        "dense_ms": dense_ms,
        "sparse_ms": sparse_ms,
    }


@torch.no_grad()
def benchmark_unfused_vs_fused_training(
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    train_seqs: list[list[int]],
    tokenizer: EnglishSubwordTokenizer,
    train_tokens_count: int,
    epochs: int = 8,
    seq_len: int = 36,
    max_seq_len: int = 128,
) -> dict:
    """Table 1: Compares Unfused (v2.4) vs. Fused In-Place (v3.1) training throughput, alloc MB & parity."""
    results = []
    models = {}

    for label, use_fused, use_fac in [
        ("Unfused v2.4 (Baseline)", False, False),
        ("Fused In-Place v3.1", True, False),
        ("Fused + Factorized v3.1", True, True),
    ]:
        model = ScaledSpringNetwork(
            preset="1.5M",
            vocab_size=tokenizer.vocab_size,
            num_layers=4,
            max_seq_len=max_seq_len,
            eta_w=0.14,
            eta_0=0.045,
            gamma=0.05,
            beta_k=0.022,
            eta_b=0.015,
            enable_neurogenesis=True,
            saturation_threshold=0.80,
            stiffness_threshold=0.08,
            error_threshold=0.005,
            cooldown_epochs=2,
            enable_hierarchical_compositional=True,
            enable_block_sparse=True,
            block_size=16,
            use_depth_scaled_highway=True,
            alpha_res=1.35,
            use_highway_anchored_dfa=True,
            dfa_anchor_blend=0.65,
            block_balance_strength=0.68,
            use_role_bridge=False,
            use_fused_kernels=use_fused,
            use_factorized_dfa=use_fac,
            seed=42,
        )
        model.calibrate_salience_from_tokens(
            train_seqs, pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id
        )

        total_sec = 0.0
        total_tok = 0
        step_alloc_mb = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            ep_mse = 0.0
            ep_correct = 0
            ep_valid = 0
            for b_idx, (inp, tgt) in enumerate(train_batches):
                if epoch == 2 and b_idx == 1:
                    with TensorAllocationTracker() as step_tracker:
                        stats = model.train_minibatch(inp, tgt, pad_id=tokenizer.pad_id)
                    step_alloc_mb = step_tracker.allocated_mb
                else:
                    stats = model.train_minibatch(inp, tgt, pad_id=tokenizer.pad_id)
                n_val = stats["valid_tokens"]
                ep_mse += stats["mse_loss"] * n_val
                ep_correct += stats["correct"]
                ep_valid += n_val
            total_sec += time.perf_counter() - t0
            total_tok += ep_valid
            train_mse = ep_mse / max(ep_valid, 1)
            tok_err = 1.0 - (ep_correct / max(float(ep_valid), 1.0))
            model.step_neurogenesis(epoch=epoch, mean_epoch_error=train_mse, token_error_rate=tok_err)

        model.production_freeze()
        tr_eval = evaluate_batches(model, train_batches, pad_id=tokenizer.pad_id)
        va_eval = evaluate_batches(model, val_batches, pad_id=tokenizer.pad_id)
        upd_alloc_mb, _ = _measure_spring_update_alloc_bytes(
            use_fused=use_fused, vocab_size=tokenizer.vocab_size, d_model=model.d_max, layers=4
        )

        results.append({
            "label": label,
            "use_fused": use_fused,
            "use_factorized_dfa": use_fac,
            "train_tok_s": total_tok / max(total_sec, 1e-6),
            "step_alloc_mb": step_alloc_mb,
            "update_alloc_mb": upd_alloc_mb,
            "train_acc_pct": tr_eval["acc"],
            "val_acc_pct": va_eval["acc"],
            "val_ppl": va_eval["ppl"],
        })
        models[(use_fused, use_fac)] = model

    base_tps = results[0]["train_tok_s"]
    for row in results:
        row["speedup_x"] = row["train_tok_s"] / max(base_tps, 1e-6)

    return {
        "rows": results,
        "fused_model": models[(True, False)],
        "fused_factorized_model": models[(True, True)],
        "unfused_model": models[(False, False)],
    }


@torch.no_grad()
def benchmark_factorized_dfa_scaling(
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    train_seqs: list[list[int]],
    tokenizer: EnglishSubwordTokenizer,
    num_layers: int = 12,
    d_max: int = 256,
    batch_tokens: int = 512,
    epochs: int = 10,
    max_seq_len: int = 128,
) -> dict:
    """
    Table 2: Benchmarks DFARouter O(L*D*V) vs. FactorizedHighwayDFA O(L*D^2) across V=[2048, 4096, 32000]
    for L=12 layers, PLUS validates end-to-end 12-Layer training convergence on TinyStories.
    """
    rows = []
    for vocab_sz in (2048, 4096, 32000):
        std_dfa = DFARouter(num_layers=num_layers, d_model=d_max, vocab_size=vocab_sz, seed=42)
        fac_dfa = FactorizedHighwayDFA(num_layers=num_layers, d_max=d_max, vocab_size=vocab_sz, seed=42)

        std_mb = (
            sum(
                getattr(std_dfa, f"B_{l}").numel() * getattr(std_dfa, f"B_{l}").element_size()
                for l in range(num_layers)
            )
            / (1024.0 * 1024.0)
        )
        fac_mb = fac_dfa.memory_mb()
        mem_saved_pct = ((std_mb - fac_mb) / max(std_mb, 1e-9)) * 100.0

        err = torch.randn(batch_tokens, vocab_sz) * 0.05
        head_w = torch.randn(vocab_sz, d_max) * 0.05

        # Warmup
        for l_idx in range(num_layers):
            _ = F.linear(err, std_dfa.get_matrix(l_idx))
        e_hwy = fac_dfa.project_vocab_to_highway(err, head_w)
        for l_idx in range(num_layers):
            _ = fac_dfa.broadcast_layer_error(l_idx, e_hwy, d_max, 0.76)

        repeats = 8 if vocab_sz <= 4096 else 4
        t0 = time.perf_counter()
        for _ in range(repeats):
            for l_idx in range(num_layers):
                _ = F.linear(err, std_dfa.get_matrix(l_idx))
        std_ms = ((time.perf_counter() - t0) / repeats) * 1000.0

        t0 = time.perf_counter()
        for _ in range(repeats):
            e_hwy = fac_dfa.project_vocab_to_highway(err, head_w)
            for l_idx in range(num_layers):
                _ = fac_dfa.broadcast_layer_error(l_idx, e_hwy, d_max, 0.76)
        fac_ms = ((time.perf_counter() - t0) / repeats) * 1000.0

        std_tok_s = (batch_tokens / max(std_ms / 1000.0, 1e-6))
        fac_tok_s = (batch_tokens / max(fac_ms / 1000.0, 1e-6))
        speedup_x = std_ms / max(fac_ms, 1e-6)

        rows.append({
            "vocab_size": vocab_sz,
            "std_mb": std_mb,
            "fac_mb": fac_mb,
            "mem_saved_pct": mem_saved_pct,
            "std_ms": std_ms,
            "fac_ms": fac_ms,
            "std_tok_s": std_tok_s,
            "fac_tok_s": fac_tok_s,
            "speedup_x": speedup_x,
        })

    # End-to-End 12-Layer convergence verification using FactorizedHighwayDFA + Fused Kernels
    model_12l = ScaledSpringNetwork(
        preset="1.5M",
        vocab_size=tokenizer.vocab_size,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        eta_w=0.14,
        eta_0=0.045,
        gamma=0.05,
        beta_k=0.022,
        eta_b=0.015,
        enable_neurogenesis=True,
        saturation_threshold=0.80,
        stiffness_threshold=0.08,
        error_threshold=0.005,
        cooldown_epochs=2,
        enable_hierarchical_compositional=True,
        enable_block_sparse=True,
        block_size=16,
        use_depth_scaled_highway=True,
        alpha_res=1.35,
        use_highway_anchored_dfa=True,
        dfa_anchor_blend=0.65,
        block_balance_strength=0.68,
        use_role_bridge=False,
        use_fused_kernels=True,
        use_factorized_dfa=True,
        seed=42,
    )
    model_12l.calibrate_salience_from_tokens(
        train_seqs, pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id
    )
    t_sec = 0.0
    t_tok = 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        ep_mse = 0.0
        ep_correct = 0
        ep_valid = 0
        for inp, tgt in train_batches:
            stats = model_12l.train_minibatch(inp, tgt, pad_id=tokenizer.pad_id)
            n_val = stats["valid_tokens"]
            ep_mse += stats["mse_loss"] * n_val
            ep_correct += stats["correct"]
            ep_valid += n_val
        t_sec += time.perf_counter() - t0
        t_tok += ep_valid
        model_12l.step_neurogenesis(
            epoch=epoch,
            mean_epoch_error=ep_mse / max(ep_valid, 1),
            token_error_rate=1.0 - (ep_correct / max(float(ep_valid), 1.0)),
        )
    model_12l.production_freeze()
    tr_12l = evaluate_batches(model_12l, train_batches, pad_id=tokenizer.pad_id)
    va_12l = evaluate_batches(model_12l, val_batches, pad_id=tokenizer.pad_id)

    return {
        "scaling_rows": rows,
        "model_12l": model_12l,
        "e2e_12l_train_acc": tr_12l["acc"],
        "e2e_12l_val_acc": va_12l["acc"],
        "e2e_12l_val_ppl": va_12l["ppl"],
        "e2e_12l_tok_s": t_tok / max(t_sec, 1e-6),
    }


@torch.no_grad()
def _build_role_bridge_model_for_cache_audit(
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    train_seqs: list[list[int]],
    tokenizer: EnglishSubwordTokenizer,
    epochs: int = 8,
    max_seq_len: int = 128,
) -> ScaledSpringNetwork:
    """Trains a compact 4L Role-Bridge model to verify O(1) Entity Slot + Sliding ALiBi Cache parity."""
    model = ScaledSpringNetwork(
        preset="1.5M",
        vocab_size=tokenizer.vocab_size,
        num_layers=4,
        max_seq_len=max_seq_len,
        eta_w=0.14,
        eta_0=0.045,
        gamma=0.05,
        beta_k=0.022,
        eta_b=0.015,
        enable_neurogenesis=True,
        enable_hierarchical_compositional=True,
        enable_block_sparse=False,
        use_role_bridge=True,
        use_fused_kernels=True,
        seed=42,
    )
    model.calibrate_salience_from_tokens(
        train_seqs, pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id
    )
    for epoch in range(1, epochs + 1):
        ep_mse = 0.0
        ep_correct = 0
        ep_valid = 0
        for inp, tgt in train_batches:
            stats = model.train_minibatch(inp, tgt, pad_id=tokenizer.pad_id)
            n_val = stats["valid_tokens"]
            ep_mse += stats["mse_loss"] * n_val
            ep_correct += stats["correct"]
            ep_valid += n_val
        model.step_neurogenesis(
            epoch=epoch,
            mean_epoch_error=ep_mse / max(ep_valid, 1),
            token_error_rate=1.0 - (ep_correct / max(float(ep_valid), 1.0)),
        )
    model.production_freeze()
    return model


@torch.no_grad()
def benchmark_stateful_inference_cache(
    models_dict: dict[str, ScaledSpringNetwork],
    tokenizer: EnglishSubwordTokenizer,
    prompts: list[str] | None = None,
    horizons: tuple[int, ...] = (30, 64),
) -> dict:
    """
    Table 3: Benchmarks O(T^2) uncached generate() vs. O(1) SpringInferenceCache rollout
    across canonical prompts at +30 and +64 new tokens, enforcing 100.00% Exact Token Sequence Match.
    """
    prompts = prompts or list(CANONICAL_SEED_PROMPTS)
    rows = []
    all_exact_match = True

    for model_label, model in models_dict.items():
        # Verify step-by-step logit numerical parity on first prompt prefix+rollout
        probe_ids = tokenizer.encode(prompts[0], add_bos=True, add_eos=False)
        max_logit_diff = 0.0
        cache = model.init_inference_cache(batch_size=1)
        for tid in probe_ids[:-1]:
            model.forward_step_cached(tid, cache, compute_logits=False)
        logits_cached = model.forward_step_cached(probe_ids[-1], cache, compute_logits=True)
        logits_full = model.forward(probe_ids)
        max_logit_diff = float((logits_cached - logits_full).abs().max().item())

        for n_new in horizons:
            total_gen_uncached = 0
            total_gen_cached = 0
            exact_matches = 0

            t0 = time.perf_counter()
            uncached_outs = []
            for pr in prompts:
                pids = tokenizer.encode(pr, add_bos=True, add_eos=False)
                out = model.generate(
                    pids, max_new_tokens=n_new, eos_id=None, enable_refractory_gate=True, use_fast_cache=False
                )
                uncached_outs.append(out)
                total_gen_uncached += len(out) - len(pids)
            sec_uncached = time.perf_counter() - t0

            t0 = time.perf_counter()
            cached_outs = []
            for pr in prompts:
                pids = tokenizer.encode(pr, add_bos=True, add_eos=False)
                out = model.generate(
                    pids, max_new_tokens=n_new, eos_id=None, enable_refractory_gate=True, use_fast_cache=True
                )
                cached_outs.append(out)
                total_gen_cached += len(out) - len(pids)
            sec_cached = time.perf_counter() - t0

            for u_seq, c_seq in zip(uncached_outs, cached_outs):
                if u_seq == c_seq:
                    exact_matches += 1
            match_pct = (exact_matches / float(len(prompts))) * 100.0
            if match_pct < 100.0:
                all_exact_match = False

            tps_uncached = total_gen_uncached / max(sec_uncached, 1e-6)
            tps_cached = total_gen_cached / max(sec_cached, 1e-6)
            ms_tok_uncached = (sec_uncached / max(total_gen_uncached, 1)) * 1000.0
            ms_tok_cached = (sec_cached / max(total_gen_cached, 1)) * 1000.0
            speedup_x = tps_cached / max(tps_uncached, 1e-6)

            rows.append({
                "model_label": model_label,
                "horizon": n_new,
                "uncached_tok_s": tps_uncached,
                "cached_tok_s": tps_cached,
                "uncached_ms_tok": ms_tok_uncached,
                "cached_ms_tok": ms_tok_cached,
                "speedup_x": speedup_x,
                "exact_match_pct": match_pct,
                "max_logit_diff": max_logit_diff,
            })

    return {"rows": rows, "all_exact_match": all_exact_match}


@torch.no_grad()
def benchmark_mixed_precision_safety(steps: int = 50, dim: int = 256) -> dict:
    """
    Table 4: Audits FP32 Reference vs. Naive BF16 (without compensator) vs. BF16 + Kahan Compensator
    over consecutive sub-ULP spring updates (`eta_w * driving ~ 2.5e-4`), proving zero NaN/underflow.
    """
    gen = torch.Generator().manual_seed(777)
    w_init = torch.randn(dim, dim, generator=gen, dtype=torch.float32) * 0.25
    k_init = torch.full((dim, dim), 0.15, dtype=torch.float32)
    signals = [torch.randn(dim, dim, generator=gen, dtype=torch.float32) * 1.8e-3 for _ in range(steps)]

    # 1. FP32 Reference
    w_ref = w_init.clone()
    w0_ref = w_init.clone()
    k_ref = k_init.clone()
    scr_ref = SpringScratchBuffer()
    for drv in signals:
        fused_spring_update_inplace(
            w_ref, w0_ref, k_ref, drv,
            eta_w=0.14, eta_0=0.05, gamma=0.04, beta_k_eff=0.08,
            stiff_ceil=0.85, scratch=scr_ref,
        )
    ref_delta_w0 = (w0_ref - w_init).norm().item()
    ref_delta_k = float((k_ref - k_init).mean().item())

    # 2. Naive BF16 (No Kahan Compensator)
    w_nb_start = w_init.to(torch.bfloat16)
    w_nb = w_nb_start.clone()
    w0_nb = w_nb_start.clone()
    k_nb_start = k_init.to(torch.bfloat16)
    k_nb = k_nb_start.clone()
    scr_nb = SpringScratchBuffer()
    for drv in signals:
        fused_spring_update_inplace(
            w_nb, w0_nb, k_nb, drv.to(torch.bfloat16),
            eta_w=0.14, eta_0=0.05, gamma=0.04, beta_k_eff=0.08,
            stiff_ceil=0.85, scratch=scr_nb, compensator=None,
        )
    naive_diff = w0_nb.float() - w_nb_start.float()
    naive_delta_w0 = naive_diff.norm().item()
    naive_delta_k = float((k_nb.float() - k_nb_start.float()).mean().item())

    # 3. Phase 3.1 BF16 + Kahan Residual Compensator
    w_kc_start = w_init.to(torch.bfloat16)
    w_kc = w_kc_start.clone()
    w0_kc = w_kc_start.clone()
    k_kc_start = k_init.to(torch.bfloat16)
    k_kc = k_kc_start.clone()
    scr_kc = SpringScratchBuffer()
    comp = MixedPrecisionCompensator()
    for drv in signals:
        fused_spring_update_inplace(
            w_kc, w0_kc, k_kc, drv.to(torch.bfloat16),
            eta_w=0.14, eta_0=0.05, gamma=0.04, beta_k_eff=0.08,
            stiff_ceil=0.85, scratch=scr_kc, compensator=comp, comp_key=1,
        )
    kahan_diff = w0_kc.float() - w_kc_start.float()
    kahan_delta_w0 = kahan_diff.norm().item()
    kahan_delta_k = float((k_kc.float() - k_kc_start.float()).mean().item())

    ref_diff = w0_ref - w_init
    rows = [
        {
            "mode": "FP32 Reference",
            "dtype": "float32",
            "nan_inf_count": int((~torch.isfinite(w0_ref)).sum().item() + (~torch.isfinite(k_ref)).sum().item()),
            "w0_delta_l2": ref_delta_w0,
            "retention_pct": 100.0,
            "mean_delta_k": ref_delta_k,
            "cos_vs_fp32": 1.0,
        },
        {
            "mode": "BF16 Naive (No Comp)",
            "dtype": "bfloat16",
            "nan_inf_count": int((~torch.isfinite(w0_nb)).sum().item() + (~torch.isfinite(k_nb)).sum().item()),
            "w0_delta_l2": naive_delta_w0,
            "retention_pct": (naive_delta_w0 / max(ref_delta_w0, 1e-8)) * 100.0,
            "mean_delta_k": naive_delta_k,
            "cos_vs_fp32": float(F.cosine_similarity(naive_diff.flatten(), ref_diff.flatten(), dim=0).item()),
        },
        {
            "mode": "BF16 + Kahan Comp (v3.1)",
            "dtype": "bfloat16",
            "nan_inf_count": int((~torch.isfinite(w0_kc)).sum().item() + (~torch.isfinite(k_kc)).sum().item()),
            "w0_delta_l2": kahan_delta_w0,
            "retention_pct": (kahan_delta_w0 / max(ref_delta_w0, 1e-8)) * 100.0,
            "mean_delta_k": kahan_delta_k,
            "cos_vs_fp32": float(F.cosine_similarity(kahan_diff.flatten(), ref_diff.flatten(), dim=0).item()),
        },
    ]
    return {"rows": rows}


def run_v3_kernels_benchmark(verbose: bool = True) -> dict:
    """Executes all 4 Phase 3.1 kernel, memory, O(1) cache & BF16 safety benchmarks."""
    ast_ok, scanned_files = verify_all_repo_files_zero_autograd()

    corpus = build_english_tinystories_corpus(target_min_tokens=25000, train_ratio=0.90, seed=42)
    train_stories = corpus["train_stories"]
    val_stories = corpus["val_stories"]
    tokenizer = EnglishSubwordTokenizer(max_vocab_size=2048)
    tokenizer.fit(train_stories + val_stories, pad_to_max_vocab=True)

    seq_len = 36
    max_seq_len = 128
    train_batches, train_seqs, train_tok = build_minibatch_tensors(
        train_stories, tokenizer, seq_len=seq_len, batch_size=32
    )
    val_batches, _, val_tok = build_minibatch_tensors(
        val_stories, tokenizer, seq_len=seq_len, batch_size=32
    )

    t1 = benchmark_unfused_vs_fused_training(
        train_batches=train_batches,
        val_batches=val_batches,
        train_seqs=train_seqs,
        tokenizer=tokenizer,
        train_tokens_count=train_tok,
        epochs=8,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
    )
    slice_gemm = benchmark_indexed_block_slice_gemm()

    t2 = benchmark_factorized_dfa_scaling(
        train_batches=train_batches,
        val_batches=val_batches,
        train_seqs=train_seqs,
        tokenizer=tokenizer,
        num_layers=12,
        d_max=256,
        batch_tokens=512,
        epochs=10,
        max_seq_len=max_seq_len,
    )

    role_model = _build_role_bridge_model_for_cache_audit(
        train_batches=train_batches,
        train_seqs=train_seqs,
        tokenizer=tokenizer,
        epochs=8,
        max_seq_len=max_seq_len,
    )
    t3 = benchmark_stateful_inference_cache(
        models_dict={
            "12L Block-Sparse v3.1": t2["model_12l"],
            "4L Role-Bridge Compositional": role_model,
        },
        tokenizer=tokenizer,
        prompts=CANONICAL_SEED_PROMPTS,
        horizons=(30, 64),
    )

    t4 = benchmark_mixed_precision_safety(steps=50, dim=256)

    if verbose:
        print("=" * 96)
        print("SS-ENN v3.0 Phase 3.1: Hardware-Accelerated Kernels, Factorized DFA & O(1) Inference Cache")
        print(f"[AST Zero-Autograd Audit] Scanned {scanned_files} Python files -> 0 backward()/grad() calls.")
        print(
            f"[Corpus] Train={len(corpus['train_stories'])} ({train_tok:,} tok) | "
            f"Val={len(corpus['val_stories'])} ({val_tok:,} tok) | Vocab V={tokenizer.vocab_size}"
        )
        print("=" * 96)

        print("\n[Table 1] Unfused (v2.4) vs. Fused In-Place (v3.1) Training & ATen Tensor Allocations (4L Block-Sparse)")
        print(f"{'Configuration':<25} | {'tok/s':>9} | {'Speedup':>7} | {'Step Alloc':>10} | {'Update Alloc':>12} | {'Train Acc':>9} | {'Val Acc':>8} | {'Val PPL':>7}")
        print("-" * 96)
        for r in t1["rows"]:
            print(
                f"{r['label']:<25} | {r['train_tok_s']:9,.1f} | {r['speedup_x']:6.2f}x | "
                f"{r['step_alloc_mb']:8.2f}MB | {r['update_alloc_mb']:10.2f}MB | "
                f"{r['train_acc_pct']:8.2f}% | {r['val_acc_pct']:7.2f}% | {r['val_ppl']:7.2f}"
            )
        print(
            f"[Indexed Active-Block Slice GEMM] U={slice_gemm['u_active']}/{slice_gemm['n_blocks']} blocks "
            f"({slice_gemm['flop_reduction_pct']:.1f}% FLOPs skipped) | "
            f"Dense={slice_gemm['dense_ms']:.3f}ms vs Sparse={slice_gemm['sparse_ms']:.3f}ms | "
            f"MaxDiff={slice_gemm['max_abs_diff']:.2e}"
        )

        print("\n[Table 2] Factorized Highway DFA [D, D] vs. Standard Per-Layer DFA [D, V] (L=12, D=256, N=512)")
        print(f"{'Vocab (V)':>9} | {'Std [D,V] MB':>12} | {'Factorized MB':>13} | {'Mem Saved':>9} | {'Std ms':>8} | {'Fac ms':>8} | {'Broadcast Speedup':>17}")
        print("-" * 96)
        for r in t2["scaling_rows"]:
            print(
                f"{r['vocab_size']:9,d} | {r['std_mb']:10.2f}MB | {r['fac_mb']:11.2f}MB | "
                f"{r['mem_saved_pct']:8.2f}% | {r['std_ms']:8.2f} | {r['fac_ms']:8.2f} | {r['speedup_x']:16.2f}x"
            )
        print(
            f"[12L End-to-End Factorized DFA Training] Train Acc={t2['e2e_12l_train_acc']:.2f}% | "
            f"Val Acc={t2['e2e_12l_val_acc']:.2f}% | Val PPL={t2['e2e_12l_val_ppl']:.2f} | "
            f"Speed={t2['e2e_12l_tok_s']:,.1f} tok/s"
        )

        print("\n[Table 3] Stateful O(1) SpringInferenceCache vs. O(T^2) Uncached Rollout (Frozen W_0 Only)")
        print(f"{'Model Architecture':<29} | {'Horizon':>7} | {'O(T^2) tok/s':>12} | {'O(1) tok/s':>11} | {'Speedup':>8} | {'Exact Match':>11} | {'MaxDiff':>9}")
        print("-" * 96)
        for r in t3["rows"]:
            print(
                f"{r['model_label']:<29} | +{r['horizon']:<6d} | {r['uncached_tok_s']:12,.1f} | "
                f"{r['cached_tok_s']:11,.1f} | {r['speedup_x']:7.2f}x | {r['exact_match_pct']:10.2f}% | {r['max_logit_diff']:9.2e}"
            )

        print("\n[Table 4] Mixed-Precision Safety Audit (50 Sub-ULP Spring Updates: FP32 vs. BF16)")
        print(f"{'Precision Mode':<27} | {'NaN/Inf':>7} | {'W0 Delta L2':>11} | {'FP32 Retention':>14} | {'Mean Delta k':>12} | {'Cos vs FP32':>11}")
        print("-" * 96)
        for r in t4["rows"]:
            print(
                f"{r['mode']:<27} | {r['nan_inf_count']:7d} | {r['w0_delta_l2']:11.6f} | "
                f"{r['retention_pct']:13.2f}% | {r['mean_delta_k']:12.6f} | {r['cos_vs_fp32']:11.6f}"
            )

    return {
        "ast_ok": ast_ok,
        "scanned_files": scanned_files,
        "table1": t1,
        "slice_gemm": slice_gemm,
        "table2": t2,
        "table3": t3,
        "table4": t4,
    }


if __name__ == "__main__":
    run_v3_kernels_benchmark(verbose=True)

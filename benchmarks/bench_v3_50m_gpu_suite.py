"""
benchmarks/bench_v3_50m_gpu_suite.py
SS-ENN v3.1.0 Scientific GPU Suite for Scale-50M (50.05M Train -> 21.63M Frozen W0).

Runs 6 end-to-end experiments on real HuggingFace binary shards (Phase 1 TinyStories
& Phase 2 Simple Wikipedia) prepared with MultilingualByteBPETokenizer (V=12,288):
  Exp 1: Pre-Allocation Footprint Audit, Live Buffer Parity & MMap Streaming Speed
  Exp 2: Phase 1 Foundational Training, Neurogenesis, Mean Stiffness, VRAM Peak & Atomic Resume
  Exp 3: Dual-Stage Sleep Consolidation & Phase 2 Continual Retraining (Retention + Rejuvenation)
  Exp 4: Production Freeze (prune_dormant=True) & Non-Destructive CPU Export (0.00e+00 Diff)
  Exp 5: GPU/CPU O(1) Stateful Cache vs O(T^2) Rollout Speedup (+64 & +128 Tokens, 100% Parity)
  Exp 6: Hardware-Calibrated Scaling & Colab Compute-Unit Projection (125M -> 7B across T4/L4/A100)

Strictly Zero-Autograd.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import sys
import tempfile
import time
from dataclasses import dataclass

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bench_scale_english_cpu import verify_all_repo_files_zero_autograd
from benchmarks.bench_v3_production_model import _max_state_abs_diff
from engine.tokenizer import MultilingualByteBPETokenizer, load_tokenizer
from models.config_v3 import SSENNConfig
from models.model_v3_production import SSENNProductionModel
from trainer.checkpoint_vault import AtomicCheckpointVault
from trainer.colab_orchestrator import ElasticCloudTrainer, evaluate_validation_batches
from trainer.mmap_dataset import BinaryShardBuilder, MMapTokenDataset, ResumableBatchStreamer


def _sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_device_and_dtype(req_device: str, req_dtype: str) -> tuple[str, str, str, str]:
    if req_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = req_device

    gpu_name = "CPU"
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if req_dtype == "auto":
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            dtype = req_dtype
    else:
        if device.startswith("cuda"):
            device = "cpu"
        dtype = "float32" if req_dtype == "auto" else req_dtype

    kahan_mode = "full" if dtype in ("bfloat16", "float16") else "off"
    return device, dtype, kahan_mode, gpu_name


def _active_stiffness_stats(model: SSENNProductionModel) -> dict[str, float]:
    """Computes mean active stiffness k_bar and locked/unlocked breakdown across layers."""
    if model.frozen:
        return {"k_bar": 0.0, "k_locked_mean": 0.0, "k_unlocked_mean": 0.0}
    d_eff, _ = model._effective_dims()
    all_k: list[torch.Tensor] = []
    locked_k: list[torch.Tensor] = []
    unlocked_k: list[torch.Tensor] = []
    for l_idx, layer in enumerate(model.spring_layers):
        d_act = layer.d_active
        k_slice = layer.k[:d_act, :d_eff].float()
        lock_mask = model.armor_locked_masks[l_idx][:d_act]
        all_k.append(k_slice.reshape(-1))
        if lock_mask.any():
            locked_k.append(k_slice[lock_mask].reshape(-1))
        if (~lock_mask).any():
            unlocked_k.append(k_slice[~lock_mask].reshape(-1))

    def _mean(parts: list[torch.Tensor]) -> float:
        if not parts:
            return 0.0
        return float(torch.cat(parts).mean().item())

    return {
        "k_bar": _mean(all_k),
        "k_locked_mean": _mean(locked_k),
        "k_unlocked_mean": _mean(unlocked_k),
    }


def _active_token_flops_proxy(cfg: SSENNConfig) -> float:
    """Estimates per-token active FLOPs (2 * MACs) for hardware-calibrated scaling projections."""
    d_eff = float(cfg.d_init)
    d_active_blk = float(cfg.top_k_blocks * cfg.block_size)
    seq_t = float(cfg.max_seq_len)
    # 1. Sparse causal attention projections (Q, K, V) + context projections
    attn_macs = 3.0 * (d_eff * d_eff) + 2.0 * min(float(cfg.attn_top_k), seq_t) * d_eff
    # 2. L sparse spring layers (dense d_eff->d_eff on active blocks approximation + K_blk*B matmul)
    layer_macs = float(cfg.num_layers) * (d_eff * d_eff + d_active_blk * d_eff)
    # 3. Output LM head + Factorized Highway DFA update overhead
    head_and_dfa_macs = 2.0 * float(cfg.vocab_size) * d_eff
    return 2.0 * (attn_macs + layer_macs + head_and_dfa_macs)


def build_smoke_shards(work_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Creates tiny offline shards so `--smoke-test` exercises all 6 experiments locally."""
    p1_dir = work_dir / "shards_phase1_stories"
    p2_dir = work_dir / "shards_phase2_knowledge"
    p1_stories = [
        "Once upon a time a little fox ran across the sunny meadow to meet his kind friend.",
        "Luna baked warm sweet bread and shared every loaf with her quiet village neighbors.",
        "Every morning the bright blue bird sang clear happy songs above the silver river.",
    ] * 12
    p2_wiki = [
        "Physics studies matter energy motion and force across space and time scales.",
        "Photosynthesis allows green plants to convert sunlight water and carbon dioxide into glucose.",
        "Astronomy investigates galaxies stars planets and orbital mechanics throughout the universe.",
    ] * 12

    tok = MultilingualByteBPETokenizer(max_vocab_size=512).fit(p1_stories, pad_to_max_vocab=True)
    tok.save(p1_dir / "tokenizer.json")
    p2_dir.mkdir(parents=True, exist_ok=True)
    tok.save(p2_dir / "tokenizer.json")

    BinaryShardBuilder.build_from_stories(p1_stories[:30], tok, seq_len=24, output_dir=p1_dir, shard_name="train_shard")
    BinaryShardBuilder.build_from_stories(p1_stories[30:], tok, seq_len=24, output_dir=p1_dir, shard_name="val_shard")
    BinaryShardBuilder.build_from_stories(p2_wiki[:30], tok, seq_len=24, output_dir=p2_dir, shard_name="train_shard")
    BinaryShardBuilder.build_from_stories(p2_wiki[30:], tok, seq_len=24, output_dir=p2_dir, shard_name="val_shard")
    return p1_dir, p2_dir


@dataclass
class GPUProfile:
    name: str
    vram_gb: float
    rel_speed_vs_t4: float
    colab_cu_per_hr: float


COLAB_GPU_PROFILES = [
    GPUProfile("Colab T4 (16GB)", 15.0, 1.00, 1.96),
    GPUProfile("Colab L4 (24GB)", 22.5, 2.35, 4.82),
    GPUProfile("Colab A100 (40GB)", 38.0, 6.20, 13.08),
]


def _estimate_scaling_table(base_cfg: SSENNConfig, measured_tok_s: float, gpu_name: str) -> list[dict]:
    """Computes Experiment 6 scaling & Colab Compute Unit table from measured throughput."""
    # Infer base multiplier from running hardware name so projections normalize to T4=1.0
    gn = gpu_name.upper()
    if "A100" in gn:
        base_hw_mult = 6.20
    elif "L4" in gn:
        base_hw_mult = 2.35
    elif "V100" in gn:
        base_hw_mult = 2.10
    elif "T4" in gn:
        base_hw_mult = 1.00
    else:
        base_hw_mult = 1.00

    base_flops = _active_token_flops_proxy(base_cfg)
    t4_base_tok_s = max(measured_tok_s / base_hw_mult, 1.0)

    scales = [
        ("125M Base", SSENNConfig(vocab_size=16384, num_layers=12, d_init=512, d_max=1024, delta_d=128, block_size=16, top_k_blocks=8, max_seq_len=1024, attn_top_k=12, dtype="bfloat16", kahan_mode="full", device="cuda")),
        ("350M Medium", SSENNConfig(vocab_size=32000, num_layers=16, d_init=768, d_max=1536, delta_d=128, block_size=16, top_k_blocks=12, max_seq_len=1024, attn_top_k=12, dtype="bfloat16", kahan_mode="full", device="cuda")),
        ("1.3B Large", SSENNConfig(vocab_size=32000, num_layers=24, d_init=1536, d_max=3072, delta_d=256, block_size=16, top_k_blocks=16, max_seq_len=2048, attn_top_k=16, dtype="bfloat16", kahan_mode="full", device="cuda")),
        ("7B Anchor-K", SSENNConfig(vocab_size=32000, num_layers=32, d_init=3072, d_max=6656, delta_d=512, block_size=16, top_k_blocks=32, max_seq_len=2048, attn_top_k=16, dtype="bfloat16", kahan_mode="anchor_k", device="cuda")),
    ]

    rows: list[dict] = []
    for label, s_cfg in scales:
        fp = s_cfg.estimate_memory_footprint()
        s_flops = _active_token_flops_proxy(s_cfg)
        flops_ratio = max(s_flops / max(base_flops, 1.0), 1e-6)
        vram_req_gb = float(fp["total_train_with_kahan_gb"])
        frozen_mb = float(fp["frozen_w0_mb"])

        hw_preds: dict[str, dict] = {}
        for prof in COLAB_GPU_PROFILES:
            fits = vram_req_gb * 1.15 <= prof.vram_gb  # Leave ~15% activation headroom
            pred_tok_s = (t4_base_tok_s * prof.rel_speed_vs_t4) / flops_ratio
            hrs_per_100m = (100_000_000.0 / max(pred_tok_s, 1.0)) / 3600.0
            cu_per_100m = hrs_per_100m * prof.colab_cu_per_hr
            hw_preds[prof.name] = {
                "fits": fits,
                "tok_s": pred_tok_s,
                "hours_per_100m_tok": hrs_per_100m,
                "compute_units_per_100m_tok": cu_per_100m,
            }
        rows.append({
            "scale": label,
            "max_train_params": int(fp["max_train_params"]),
            "frozen_w0_params": int(fp["frozen_w0_params"]),
            "train_plus_kahan_gb": vram_req_gb,
            "frozen_w0_mb": frozen_mb,
            "kahan_mode": fp["kahan_mode"],
            "hardware": hw_preds,
        })
    return rows


def run_suite(args: argparse.Namespace) -> dict:
    torch.set_grad_enabled(False)
    passed_ast, file_cnt = verify_all_repo_files_zero_autograd()
    if not passed_ast:
        raise RuntimeError("AST Zero-Autograd verification failed!")

    device, dtype, kahan_mode, gpu_name = _resolve_device_and_dtype(args.device, args.dtype)
    print("=" * 108)
    print("SS-ENN v3.1.0 SCIENTIFIC GPU BENCHMARK SUITE (Zero-Autograd 6-Experiment Audit)")
    print("=" * 108)
    print(f"  AST Zero-Autograd : PASS ({file_cnt} Python files verified)")
    print(f"  Accelerator       : {gpu_name} (device={device}, dtype={dtype}, kahan_mode={kahan_mode})")

    temp_ctx = tempfile.TemporaryDirectory() if args.smoke_test else None
    try:
        if args.smoke_test:
            tmp_root = pathlib.Path(temp_ctx.name)
            p1_dir, p2_dir = build_smoke_shards(tmp_root)
            out_dir = tmp_root / "results"
            cfg = SSENNConfig(
                vocab_size=512, num_layers=4, d_init=64, d_max=128, delta_d=32,
                block_size=16, top_k_blocks=2, max_seq_len=64, attn_top_k=4,
                salient_anchor_slots=1, dtype=dtype, kahan_mode=kahan_mode, device=device, seed=42,
            )
            p1_epochs = 2
            p2_epochs = 4
            batch_size = 16
            ckpt_steps = 4
            expected_train_params = None
        else:
            p1_dir = pathlib.Path(args.phase1_dir)
            p2_dir = pathlib.Path(args.phase2_dir)
            out_dir = pathlib.Path(args.output_dir)
            cfg = SSENNConfig(
                vocab_size=12288,
                num_layers=8,
                d_init=384,
                d_max=640,
                delta_d=64,
                block_size=16,
                top_k_blocks=6,
                max_seq_len=256,
                attn_top_k=12,
                salient_anchor_slots=2,
                dtype=dtype,
                kahan_mode=kahan_mode,
                device=device,
                seed=42,
            )
            p1_epochs = int(args.p1_epochs)
            p2_epochs = int(args.p2_epochs)
            batch_size = int(args.batch_size)
            ckpt_steps = int(args.ckpt_every_steps)
            expected_train_params = 50_053_120

        out_dir.mkdir(parents=True, exist_ok=True)
        vault_root = out_dir / "vault_50m"
        vault_p1 = vault_root / "phase1"
        vault_p2 = vault_root / "phase2"

        tok = load_tokenizer(p1_dir / "tokenizer.json")
        results: dict = {
            "ast_zero_autograd_files": int(file_cnt),
            "accelerator": gpu_name,
            "device": device,
            "dtype": dtype,
            "kahan_mode": kahan_mode,
            "config": cfg.to_dict(),
        }

        with (
            MMapTokenDataset(p1_dir, shard_name="train_shard", verify_checksum=True) as p1_train_ds,
            MMapTokenDataset(p1_dir, shard_name="val_shard", verify_checksum=True) as p1_val_ds,
            MMapTokenDataset(p2_dir, shard_name="train_shard", verify_checksum=True) as p2_train_ds,
            MMapTokenDataset(p2_dir, shard_name="val_shard", verify_checksum=True) as p2_val_ds,
        ):
            # ------------------------------------------------------------------
            # Experiment 1: Pre-Allocation Footprint & MMap Streaming Audit
            # ------------------------------------------------------------------
            print("\n[Experiment 1] Pre-Allocation Footprint Match & MMap Disk Streaming Audit")
            est = cfg.estimate_memory_footprint()
            if expected_train_params is not None:
                assert est["max_train_params"] == expected_train_params, est["max_train_params"]
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            trainer = ElasticCloudTrainer(
                config=cfg,
                vault_dir=vault_p1,
                checkpoint_every_steps=ckpt_steps,
                keep_best_frozen=True,
                sleep_every_epochs=0,
                verbose=True,
            )
            brk_init = trainer.model.count_parameters_breakdown()
            assert est["exact_live_buffer_params_at_max"] == brk_init["live_buffer_params"], (
                est["exact_live_buffer_params_at_max"],
                brk_init["live_buffer_params"],
            )

            for ds_name, ds in [
                ("p1_train", p1_train_ds), ("p1_val", p1_val_ds),
                ("p2_train", p2_train_ds), ("p2_val", p2_val_ds),
            ]:
                assert ds.verify_integrity() is True, f"Integrity check failed on {ds_name}"

            # Benchmark pure MMap streaming speed across Phase 1 train shard
            probe_streamer = ResumableBatchStreamer(p1_train_ds, batch_size=batch_size, shuffle=False, prefetch_batches=2)
            t_stream0 = time.perf_counter()
            streamed_tok = 0
            for _, xb, _ in probe_streamer.iter_epoch(epoch=1, start_cursor=0):
                streamed_tok += int(xb.numel())
            stream_s = max(time.perf_counter() - t_stream0, 1e-6)
            stream_tok_s = streamed_tok / stream_s

            exp1 = {
                "max_train_params": int(est["max_train_params"]),
                "frozen_w0_params": int(est["frozen_w0_params"]),
                "exact_live_buffer_params_at_max": int(est["exact_live_buffer_params_at_max"]),
                "actual_live_buffer_params": int(brk_init["live_buffer_params"]),
                "est_total_train_with_kahan_mb": float(est["total_train_with_kahan_mb"]),
                "actual_total_live_mb": float(brk_init["total_live_mb"]),
                "est_frozen_w0_mb": float(est["frozen_w0_mb"]),
                "p1_train_tokens": int(p1_train_ds.total_stored_tokens),
                "p1_val_tokens": int(p1_val_ds.total_stored_tokens),
                "p2_train_tokens": int(p2_train_ds.total_stored_tokens),
                "p2_val_tokens": int(p2_val_ds.total_stored_tokens),
                "all_shards_integrity_verified": True,
                "mmap_stream_tok_s": float(stream_tok_s),
                "mmap_meta_ram_kb": float((p1_dir / "train_shard.meta.json").stat().st_size / 1024.0),
            }
            results["experiment_1_audit"] = exp1
            print(
                f"  * Params Match    : Train {exp1['max_train_params']:,} | Frozen W0 {exp1['frozen_w0_params']:,} | "
                f"Exact Buffers {exp1['actual_live_buffer_params']:,} == Est {exp1['exact_live_buffer_params_at_max']:,}"
            )
            print(
                f"  * Live RAM/VRAM   : {exp1['actual_total_live_mb']:.2f} MB (Est {exp1['est_total_train_with_kahan_mb']:.2f} MB) | "
                f"MMap Stream: {stream_tok_s:,.0f} tok/s | Integrity: PASS"
            )

            # ------------------------------------------------------------------
            # Experiment 2: Phase 1 Foundational Training + Neurogenesis + Vault
            # ------------------------------------------------------------------
            print("\n[Experiment 2] Phase 1 Training (TinyStories Domain A), Neurogenesis & Atomic Vault")
            has_p1_ckpt = trainer.vault.latest_path.exists() or trainer.vault.backup_path.exists()
            if not has_p1_ckpt:
                trainer.calibrate_salience_from_dataset(
                    p1_train_ds, pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id
                )
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _sync_device(device)
            t_p1 = time.perf_counter()
            p1_res = trainer.fit(
                train_dataset=p1_train_ds,
                val_batches=p1_val_ds,
                epochs=p1_epochs,
                start_epoch=1,
                batch_size=batch_size,
                pad_id=tok.pad_id,
                auto_resume=True,
                stop_after_steps=args.max_steps_per_phase,
                save_on_stop_after_steps=bool(args.max_steps_per_phase),
            )
            _sync_device(device)
            p1_wall_s = max(time.perf_counter() - t_p1, 1e-6)
            p1_vram_peak_mb = (
                float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                if device.startswith("cuda") and torch.cuda.is_available()
                else 0.0
            )
            p1_stiff = _active_stiffness_stats(trainer.model)

            # Atomic Checkpoint Reload Parity Audit (0.00e+00)
            probe_trainer = ElasticCloudTrainer(config=cfg, vault_dir=vault_p1, keep_best_frozen=False, verbose=False)
            resume_chk = probe_trainer.restore_from_vault()
            resume_diff = _max_state_abs_diff(trainer.model, probe_trainer.model) if resume_chk["resumed"] else float("nan")
            del probe_trainer
            gc.collect()

            p1_ep_hist = list(p1_res["epochs_history"])
            if not p1_ep_hist and trainer.vault.telemetry_path.exists():
                for line in trainer.vault.telemetry_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        ev = json.loads(line)
                        if ev.get("event") == "epoch_end":
                            p1_ep_hist.append(ev)

            p1_val_metrics = evaluate_validation_batches(trainer.model, p1_val_ds, pad_id=tok.pad_id, batch_size=batch_size)
            exp2 = {
                "status": p1_res["status"],
                "wall_s": float(p1_wall_s),
                "tok_s": float(p1_res["tok_s"]),
                "vram_peak_mb": p1_vram_peak_mb,
                "final_step": int(p1_res["final_global_step"]),
                "last_train_acc_pct": float(p1_ep_hist[-1]["train_acc_pct"]) if (p1_ep_hist and p1_res["last_train_acc_pct"] == 0.0) else float(p1_res["last_train_acc_pct"]),
                "last_train_ce": float(p1_ep_hist[-1]["train_ce"]) if (p1_ep_hist and p1_res["last_train_ce"] == 0.0) else float(p1_res["last_train_ce"]),
                "val_acc_pct": float(p1_val_metrics["acc_pct"]),
                "val_ppl": float(p1_val_metrics["ppl"]),
                "allocated_dims_after_p1": list(trainer.model.get_allocated_dimensions()),
                "growth_events": list(trainer.model.growth_log),
                "mean_stiffness_k": p1_stiff["k_bar"],
                "atomic_vault_resume_diff": float(resume_diff),
                "epochs_history": p1_ep_hist,
            }
            results["experiment_2_phase1"] = exp2
            print(
                f"  * Phase 1 Done    : {exp2['tok_s']:,.1f} tok/s | VRAM Peak: {p1_vram_peak_mb:.1f} MB | "
                f"Val Acc: {exp2['val_acc_pct']:.2f}% (PPL {exp2['val_ppl']:.2f})"
            )
            print(
                f"  * Structure/Vault : Dims={exp2['allocated_dims_after_p1']} | k_bar={exp2['mean_stiffness_k']:.4f} | "
                f"Atomic Resume Diff={resume_diff:.2e}"
            )

            # ------------------------------------------------------------------
            # Experiment 3: Sleep Consolidation & Phase 2 Continual Retraining
            # ------------------------------------------------------------------
            print("\n[Experiment 3] Dual-Stage Sleep Consolidation & Phase 2 Continual Knowledge Retraining")
            p2_before_metrics = evaluate_validation_batches(trainer.model, p2_val_ds, pad_id=tok.pad_id, batch_size=batch_size)
            sleep_info = trainer.model.sleep_consolidate(phase_id=1)
            rejuv_after_sleep = [len(slots) for slots in trainer.model.rejuvenated_slots]
            p1_post_sleep_metrics = evaluate_validation_batches(trainer.model, p1_val_ds, pad_id=tok.pad_id, batch_size=batch_size)

            # Reset peak VRAM & track Phase 2 in dedicated sub-vault so Phase 1 checkpoint remains intact
            trainer.vault = AtomicCheckpointVault(vault_p2)
            trainer.best_val_acc = -1.0
            trainer.best_val_ppl = float("inf")
            has_p2_ckpt = trainer.vault.latest_path.exists() or trainer.vault.backup_path.exists()
            p2_start_epoch = max(int(trainer.model.current_epoch), p1_epochs) + 1
            p2_target_epoch = p2_start_epoch + p2_epochs - 1

            _sync_device(device)
            t_p2 = time.perf_counter()
            p2_res = trainer.fit(
                train_dataset=p2_train_ds,
                val_batches=p2_val_ds,
                epochs=p2_target_epoch,
                start_epoch=p2_start_epoch,
                batch_size=batch_size,
                pad_id=tok.pad_id,
                auto_resume=has_p2_ckpt,
                stop_after_steps=(
                    int(trainer.global_step) + int(args.max_steps_per_phase)
                    if args.max_steps_per_phase is not None
                    else None
                ),
                save_on_stop_after_steps=bool(args.max_steps_per_phase),
            )
            _sync_device(device)
            p2_wall_s = max(time.perf_counter() - t_p2, 1e-6)
            rejuv_after_p2 = [len(slots) for slots in trainer.model.rejuvenated_slots]
            rejuv_revived = sum(rejuv_after_sleep) - sum(rejuv_after_p2)
            p2_stiff = _active_stiffness_stats(trainer.model)

            p1_post_p2_metrics = evaluate_validation_batches(trainer.model, p1_val_ds, pad_id=tok.pad_id, batch_size=batch_size)
            p2_post_p2_metrics = evaluate_validation_batches(trainer.model, p2_val_ds, pad_id=tok.pad_id, batch_size=batch_size)

            exp3 = {
                "sleep_merged_total": int(sleep_info["merged_total"]),
                "sleep_pruned_total": int(sleep_info["pruned_total"]),
                "sleep_locked_total": int(sleep_info["locked_total"]),
                "sleep_recycled_total": int(sleep_info["recycled_total"]),
                "rejuvenated_slots_after_sleep": int(sum(rejuv_after_sleep)),
                "rejuvenated_slots_remaining_after_p2": int(sum(rejuv_after_p2)),
                "rejuvenated_slots_revived": int(rejuv_revived),
                "p2_tok_s": float(p2_res["tok_s"]),
                "p2_wall_s": float(p2_wall_s),
                "p1_val_before_sleep_acc": float(p1_val_metrics["acc_pct"]),
                "p1_val_post_sleep_acc": float(p1_post_sleep_metrics["acc_pct"]),
                "p1_val_post_p2_acc": float(p1_post_p2_metrics["acc_pct"]),
                "p1_val_post_p2_ppl": float(p1_post_p2_metrics["ppl"]),
                "p2_val_zero_shot_acc": float(p2_before_metrics["acc_pct"]),
                "p2_val_post_p2_acc": float(p2_post_p2_metrics["acc_pct"]),
                "p2_val_post_p2_ppl": float(p2_post_p2_metrics["ppl"]),
                "k_locked_mean": p2_stiff["k_locked_mean"],
                "k_unlocked_mean": p2_stiff["k_unlocked_mean"],
                "allocated_dims_after_p2": list(trainer.model.get_allocated_dimensions()),
            }
            results["experiment_3_sleep_continual"] = exp3
            print(
                f"  * Sleep Consolidated : merged={exp3['sleep_merged_total']} | pruned={exp3['sleep_pruned_total']} | "
                f"locked={exp3['sleep_locked_total']} | pool={exp3['rejuvenated_slots_after_sleep']} -> revived={exp3['rejuvenated_slots_revived']}"
            )
            print(
                f"  * Continual Scores   : Phase 1 Retention={exp3['p1_val_post_p2_acc']:.2f}% (PPL {exp3['p1_val_post_p2_ppl']:.2f}) | "
                f"Phase 2 Knowledge={exp3['p2_val_zero_shot_acc']:.2f}% -> {exp3['p2_val_post_p2_acc']:.2f}% (PPL {exp3['p2_val_post_p2_ppl']:.2f})"
            )

            # ------------------------------------------------------------------
            # Experiment 4: Production Freeze & Non-Destructive CPU Export
            # ------------------------------------------------------------------
            print("\n[Experiment 4] Production Freeze (prune_dormant=True) & Non-Destructive CPU Export Parity")
            pre_freeze_brk = trainer.model.count_parameters_breakdown()
            nd_state, nd_stats = trainer.model.export_frozen_w0_state_nondestructive(map_location="cpu")
            assert not trainer.model.frozen, "Non-destructive export mutated live frozen flag!"

            freeze_stats = trainer.model.production_freeze(prune_dormant=True)
            live_frozen_state = trainer.model.export_frozen_w0_state()

            # Compare non-destructive CPU export against destructive freeze tensor-by-tensor
            nd_max_diff = 0.0
            for tkey, nd_val in nd_state["w0_tensors"].items():
                ref_val = live_frozen_state["w0_tensors"][tkey]
                if isinstance(nd_val, torch.Tensor):
                    diff = float((nd_val.float() - ref_val.detach().cpu().float()).abs().max().item())
                    nd_max_diff = max(nd_max_diff, diff)
                elif isinstance(nd_val, list):
                    for idx_l, nd_lt in enumerate(nd_val):
                        diff = float((nd_lt.float() - ref_val[idx_l].detach().cpu().float()).abs().max().item())
                        nd_max_diff = max(nd_max_diff, diff)

            # Save final compacted W0 checkpoint into vault root
            final_w0_path = out_dir / "final_50m_frozen_w0.pt"
            torch.save(nd_state, final_w0_path)
            p1_frozen_val = evaluate_validation_batches(trainer.model, p1_val_ds, pad_id=tok.pad_id, batch_size=batch_size)
            p2_frozen_val = evaluate_validation_batches(trainer.model, p2_val_ds, pad_id=tok.pad_id, batch_size=batch_size)

            exp4 = {
                "before_live_mb": float(pre_freeze_brk["total_live_mb"]),
                "before_kahan_mb": float(pre_freeze_brk["kahan_residual_mb"]),
                "after_frozen_mb": float(freeze_stats["after_live_mb"]),
                "memory_saved_pct": float(freeze_stats["memory_saved_pct"]),
                "before_params": int(freeze_stats["before_params"]),
                "after_params": int(freeze_stats["after_params"]),
                "compacted_dims": list(freeze_stats["compacted_dims"]),
                "nondestructive_export_diff": float(nd_max_diff),
                "p1_frozen_val_acc": float(p1_frozen_val["acc_pct"]),
                "p1_frozen_val_ppl": float(p1_frozen_val["ppl"]),
                "p2_frozen_val_acc": float(p2_frozen_val["acc_pct"]),
                "p2_frozen_val_ppl": float(p2_frozen_val["ppl"]),
                "final_w0_disk_mb": float(final_w0_path.stat().st_size / (1024 * 1024)),
            }
            results["experiment_4_freeze"] = exp4
            print(
                f"  * Memory Pruned     : {exp4['before_live_mb']:.2f} MB (Kahan {exp4['before_kahan_mb']:.2f} MB) -> "
                f"{exp4['after_frozen_mb']:.2f} MB (-{exp4['memory_saved_pct']:.1f}%) | Export Diff={nd_max_diff:.2e}"
            )
            print(
                f"  * Frozen W0 Scores  : P1 Val={exp4['p1_frozen_val_acc']:.2f}% (PPL {exp4['p1_frozen_val_ppl']:.2f}) | "
                f"P2 Val={exp4['p2_frozen_val_acc']:.2f}% (PPL {exp4['p2_frozen_val_ppl']:.2f})"
            )

        # ------------------------------------------------------------------
        # Experiment 5: O(1) Stateful Cache vs O(T^2) Rollout (+64 & +128)
        # ------------------------------------------------------------------
        print("\n[Experiment 5] O(1) Stateful Cache vs O(T^2) Rollout Speedup (+64 & +128 Tokens)")
        prompt_text = "Once upon a time there was a"
        prompt_ids = tok.encode(prompt_text, add_bos=True, add_eos=False)
        horizons = [32, 48] if args.smoke_test else [64, 128]
        # Clamp horizon to max_seq_len - len(prompt_ids) cleanly
        horizons = [min(h, max(8, cfg.max_seq_len - len(prompt_ids))) for h in horizons]

        # Warmup GPU kernels before timing
        _ = trainer.model.generate(prompt_ids, max_new_tokens=8, eos_id=-1, use_fast_cache=True)
        _ = trainer.model.generate(prompt_ids, max_new_tokens=8, eos_id=-1, use_fast_cache=False)
        _sync_device(device)

        exp5_horizons: list[dict] = []
        for h_len in horizons:
            _sync_device(device)
            t_fast = time.perf_counter()
            out_fast = trainer.model.generate(prompt_ids, max_new_tokens=h_len, eos_id=-1, use_fast_cache=True)
            _sync_device(device)
            fast_s = max(time.perf_counter() - t_fast, 1e-6)

            _sync_device(device)
            t_slow = time.perf_counter()
            out_slow = trainer.model.generate(prompt_ids, max_new_tokens=h_len, eos_id=-1, use_fast_cache=False)
            _sync_device(device)
            slow_s = max(time.perf_counter() - t_slow, 1e-6)

            n_gen = max(1, len(out_fast) - len(prompt_ids))
            match_pct = 100.0 if out_fast == out_slow else (
                sum(int(a == b) for a, b in zip(out_fast, out_slow)) / max(len(out_fast), len(out_slow), 1) * 100.0
            )
            fast_tok_s = n_gen / fast_s
            slow_tok_s = n_gen / slow_s
            speedup_x = fast_tok_s / max(slow_tok_s, 1e-9)
            rec = {
                "horizon_tokens": int(n_gen),
                "o1_tok_s": float(fast_tok_s),
                "o1_ms_per_tok": float((fast_s * 1000.0) / n_gen),
                "ot2_tok_s": float(slow_tok_s),
                "ot2_ms_per_tok": float((slow_s * 1000.0) / n_gen),
                "speedup_x": float(speedup_x),
                "sequence_match_pct": float(match_pct),
            }
            exp5_horizons.append(rec)
            print(
                f"  * Horizon +{n_gen:3d} tok  : O(1) {fast_tok_s:8.1f} tok/s ({rec['o1_ms_per_tok']:.2f} ms/tok) vs "
                f"O(T^2) {slow_tok_s:8.1f} tok/s ({speedup_x:.2f}x speedup) | Parity: {match_pct:.2f}%"
            )

        sample_gen_ids = trainer.model.generate(
            prompt_ids, max_new_tokens=32, eos_id=tok.eos_id, use_fast_cache=True
        )
        sample_decoded = tok.decode(sample_gen_ids, skip_special_tokens=True)
        results["experiment_5_inference"] = {
            "prompt": prompt_text,
            "horizons": exp5_horizons,
            "sample_natural_rollout": sample_decoded,
        }

        # ------------------------------------------------------------------
        # Experiment 6: Hardware-Calibrated Scaling & Cost Projection Table
        # ------------------------------------------------------------------
        print("\n[Experiment 6] Hardware-Calibrated Scaling & Colab Compute-Unit Projection (125M -> 7B)")
        measured_train_tok_s = max(exp2["tok_s"] if exp2["tok_s"] > 0.0 else exp3["p2_tok_s"], 1.0)
        scaling_rows = _estimate_scaling_table(cfg, measured_tok_s=measured_train_tok_s, gpu_name=gpu_name)
        results["experiment_6_scaling_projection"] = scaling_rows

        print("-" * 108)
        print(
            f"{'Scale / Mode':<14} | {'Train VRAM':<10} | {'Frozen W0':<9} | "
            f"{'T4 (16GB) 100M-tok':<24} | {'L4 (24GB) 100M-tok':<24} | {'A100 (40GB) 100M-tok':<24}"
        )
        print("-" * 108)
        for r in scaling_rows:
            t4 = r["hardware"]["Colab T4 (16GB)"]
            l4 = r["hardware"]["Colab L4 (24GB)"]
            a100 = r["hardware"]["Colab A100 (40GB)"]
            fmt_hw = lambda d: (
                f"{d['tok_s']:6.0f}/s ({d['hours_per_100m_tok']:.1f}h/{d['compute_units_per_100m_tok']:.0f}CU)"
                if d["fits"]
                else "OOM (VRAM exceeded)"
            )
            print(
                f"{r['scale']:<14} | {r['train_plus_kahan_gb']:7.2f} GB | {r['frozen_w0_mb']:6.1f} MB | "
                f"{fmt_hw(t4):<24} | {fmt_hw(l4):<24} | {fmt_hw(a100):<24}"
            )
        print("-" * 108)

        # Save JSON & Markdown summary artifacts cleanly to output_dir
        json_path = out_dir / "bench_50m_results.json"
        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        md_lines = [
            "# SS-ENN v3.1.0 Scale-50M Scientific GPU Benchmark Report",
            f"- **Accelerator:** `{gpu_name}` (`device={device}`, `dtype={dtype}`, `kahan_mode={kahan_mode}`)",
            f"- **AST Zero-Autograd:** `PASS ({file_cnt} files)`",
            f"- **Model Params:** `{exp1['max_train_params']:,}` max train -> `{exp4['after_params']:,}` frozen W0",
            "",
            "| Experiment | Key Metric | Value |",
            "| :--- | :--- | :--- |",
            f"| **1. Audit & MMap** | Live Buffers / Stream Speed | `{exp1['actual_live_buffer_params']:,}` exact match / `{exp1['mmap_stream_tok_s']:,.0f} tok/s` |",
            f"| **2. Phase 1 Stories** | GPU Train Speed / VRAM Peak / Val Acc | `{exp2['tok_s']:,.1f} tok/s` / `{exp2['vram_peak_mb']:.1f} MB` / `{exp2['val_acc_pct']:.2f}% (PPL {exp2['val_ppl']:.2f})` |",
            f"| **3. Sleep + Phase 2** | Merged/Pruned/Locked / P1 Retention / P2 Val | `{exp3['sleep_merged_total']}/{exp3['sleep_pruned_total']}/{exp3['sleep_locked_total']}` / `{exp3['p1_val_post_p2_acc']:.2f}%` / `{exp3['p2_val_post_p2_acc']:.2f}%` |",
            f"| **4. Frozen W0** | Memory Saved / Non-Destructive Diff | `{exp4['before_live_mb']:.2f} -> {exp4['after_frozen_mb']:.2f} MB (-{exp4['memory_saved_pct']:.1f}%)` / `{exp4['nondestructive_export_diff']:.2e}` |",
            f"| **5. O(1) vs O(T^2)** | Max Horizon Speedup & Match | `{exp5_horizons[-1]['speedup_x']:.2f}x ({exp5_horizons[-1]['o1_tok_s']:.1f} tok/s)` / `{exp5_horizons[-1]['sequence_match_pct']:.2f}%` |",
        ]
        md_path = out_dir / "bench_50m_report.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"\nSaved scientific artifacts:\n  - {json_path}\n  - {md_path}")
        return results
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SS-ENN v3.1.3 50M GPU Scientific Benchmark Suite (Zero-Autograd)")
    p.add_argument("--phase1-dir", type=str, default="/content/local_50m/shards_phase1_stories", help="Phase 1 stories shard dir.")
    p.add_argument("--phase2-dir", type=str, default="/content/local_50m/shards_phase2_knowledge", help="Phase 2 knowledge shard dir.")
    p.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/SS_ENN_Workspace/exp_50m_scientific/results_v3_1_3", help="Output vault & report directory.")
    p.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu.")
    p.add_argument("--dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"), help="Weight precision.")
    p.add_argument("--p1-epochs", type=int, default=2, help="Phase 1 foundational epochs.")
    p.add_argument("--p2-epochs", type=int, default=4, help="Phase 2 continual retraining epochs.")
    p.add_argument("--batch-size", type=int, default=64, help="Training/validation minibatch size.")
    p.add_argument("--ckpt-every-steps", type=int, default=100, help="Periodic checkpoint interval in steps.")
    p.add_argument("--max-steps-per-phase", type=int, default=None, help="Optional step cap per phase for quick runs.")
    p.add_argument("--smoke-test", action="store_true", help="Run fast self-contained CPU smoke audit of all 6 experiments.")
    return p.parse_args(argv)


if __name__ == "__main__":
    run_suite(parse_args())

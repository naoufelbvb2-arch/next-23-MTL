"""
benchmarks/bench_v4_125m_gpu_suite.py
SS-ENN v4.1 Base-125M Unified Hybrid Foundation GPU Benchmark Suite (Zero-Autograd).

Trains the frozen v4.1.0 architecture at Base-125M capacity (`114.56M` Max Train ->
`44.30M` Deployed W0, `12` layers, `d_init=640 -> d_max=1024`, `delta_d=128`, `H=8`,
`seq_len=256`, `attn_top_k=16`, `salient_anchor_slots=4`) on a unified shuffled ~25M-token
hybrid shard (`TinyStories` + `Simple Wikipedia`) and evaluates BOTH domains per epoch:
  Exp 1: Pre-Allocation Footprint Parity & MMap Streaming Audit (`~25M` Tokens)
  Exp 2: Unified Foundation Training (6 Epochs, `sleep=0`), Dual-Domain PPL & BPB + Atomic Resume
  Exp 3: Production Freeze (`prune_dormant=True`), Non-Destructive Export (`0.00e+00`) & Dual Eval
  Exp 4: O(1) Cache vs O(T^2) Rollout (`+128` & `+256` Horizon) + Dual-Domain Live Generation

Strictly Zero-Autograd.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import random
import sys
import tempfile
import time

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bench_scale_english_cpu import verify_all_repo_files_zero_autograd
from benchmarks.bench_v3_50m_gpu_suite import (
    _active_stiffness_stats,
    _resolve_device_and_dtype,
    _sync_device,
)
from benchmarks.bench_v3_production_model import _max_state_abs_diff
from engine.tokenizer import MultilingualByteBPETokenizer, load_tokenizer
from models.config_v3 import SSENNConfig
from trainer.colab_orchestrator import ElasticCloudTrainer, evaluate_validation_batches
from trainer.mmap_dataset import BinaryShardBuilder, MMapTokenDataset, ResumableBatchStreamer


def _compute_bytes_per_valid_token(ds: MMapTokenDataset, tok) -> float:
    """Computes mean UTF-8 bytes per valid target token for exact Bits-Per-Byte (BPB)."""
    arr = ds._mmap
    if arr is None or arr.shape[0] == 0:
        return 4.0
    pad_id = int(tok.pad_id)
    total_bytes = 0
    total_valid = 0
    for idx in range(arr.shape[0]):
        tids = [int(t) for t in arr[idx, 1:] if int(t) != pad_id]
        if tids:
            total_valid += len(tids)
            total_bytes += len(tok.decode(tids, skip_special_tokens=True).encode("utf-8"))
    return float(total_bytes) / max(float(total_valid), 1.0)


def build_smoke_hybrid_shards(work_dir: pathlib.Path) -> pathlib.Path:
    """Builds compact offline hybrid shards for fast local `--smoke-test` verification."""
    shard_dir = work_dir / "shards_25m_hybrid_seq256"
    stories = [
        "Once upon a time a little fox ran across the sunny meadow to meet his kind friend.",
        "Luna baked warm sweet bread and shared every loaf with her quiet village neighbors.",
        "Every morning the bright blue bird sang clear happy songs above the silver river.",
    ] * 16
    wiki = [
        "Physics studies matter energy motion and force across space and time scales in nature.",
        "Photosynthesis allows green plants to convert sunlight water and carbon dioxide into glucose.",
        "Astronomy investigates galaxies stars planets and orbital mechanics throughout the universe.",
    ] * 16

    tok = MultilingualByteBPETokenizer(max_vocab_size=512).fit(stories + wiki, pad_to_max_vocab=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    tok.save(shard_dir / "tokenizer.json")

    hybrid_train = stories[:36] + wiki[:36]
    random.Random(42).shuffle(hybrid_train)
    BinaryShardBuilder.build_from_stories(hybrid_train, tok, seq_len=32, output_dir=shard_dir, shard_name="train_shard")
    BinaryShardBuilder.build_from_stories(stories[36:], tok, seq_len=32, output_dir=shard_dir, shard_name="val_stories_shard")
    BinaryShardBuilder.build_from_stories(wiki[36:], tok, seq_len=32, output_dir=shard_dir, shard_name="val_wiki_shard")
    return shard_dir


def _dual_metrics_summary(
    st_metrics: dict[str, float],
    wk_metrics: dict[str, float],
    st_bytes_per_tok: float,
    wk_bytes_per_tok: float,
) -> dict[str, float]:
    """Computes per-domain BPB and token-weighted combined perplexity/BPB."""
    inv_ln2 = 1.0 / math.log(2.0)
    st_ce = float(st_metrics["ce"])
    wk_ce = float(wk_metrics["ce"])
    st_n = max(float(st_metrics["valid_tokens"]), 1.0)
    wk_n = max(float(wk_metrics["valid_tokens"]), 1.0)
    comb_ce = (st_ce * st_n + wk_ce * wk_n) / (st_n + wk_n)
    comb_bpb = ((st_ce * st_n + wk_ce * wk_n) * inv_ln2) / (
        st_n * max(st_bytes_per_tok, 1e-6) + wk_n * max(wk_bytes_per_tok, 1e-6)
    )
    return {
        "stories_val_acc_pct": float(st_metrics["acc_pct"]),
        "stories_val_ce": st_ce,
        "stories_val_ppl": float(st_metrics["ppl"]),
        "stories_val_bpb": (st_ce * inv_ln2) / max(st_bytes_per_tok, 1e-6),
        "wiki_val_acc_pct": float(wk_metrics["acc_pct"]),
        "wiki_val_ce": wk_ce,
        "wiki_val_ppl": float(wk_metrics["ppl"]),
        "wiki_val_bpb": (wk_ce * inv_ln2) / max(wk_bytes_per_tok, 1e-6),
        "combined_weighted_ce": comb_ce,
        "combined_weighted_ppl": math.exp(min(comb_ce, 20.0)),
        "combined_weighted_bpb": comb_bpb,
    }


def _load_domain_bytes_per_tok(
    shard_dir: pathlib.Path,
    val_st_ds: MMapTokenDataset,
    val_wk_ds: MMapTokenDataset,
    tok,
) -> tuple[float, float]:
    stats_path = shard_dir / "shard_domain_stats.json"
    if stats_path.exists():
        data = json.loads(stats_path.read_text(encoding="utf-8"))
        st_bpt = float(data.get("val_stories_bytes_per_valid_tok", 0.0))
        wk_bpt = float(data.get("val_wiki_bytes_per_valid_tok", 0.0))
        if st_bpt > 0.0 and wk_bpt > 0.0:
            return st_bpt, wk_bpt
    return _compute_bytes_per_valid_token(val_st_ds, tok), _compute_bytes_per_valid_token(val_wk_ds, tok)


def run_suite(args: argparse.Namespace) -> dict:
    torch.set_grad_enabled(False)
    passed_ast, file_cnt = verify_all_repo_files_zero_autograd()
    if not passed_ast:
        raise RuntimeError("AST Zero-Autograd verification failed!")

    device, dtype, kahan_mode, gpu_name = _resolve_device_and_dtype(args.device, args.dtype)
    print("=" * 112)
    print("SS-ENN v4.1 BASE-125M UNIFIED HYBRID FOUNDATION GPU SUITE (Zero-Autograd, A100/L4/T4)")
    print("=" * 112)
    print(f"  AST Zero-Autograd : PASS ({file_cnt} Python files verified)")
    print(f"  Accelerator       : {gpu_name} (device={device}, dtype={dtype}, kahan_mode={kahan_mode})")

    temp_ctx = tempfile.TemporaryDirectory() if args.smoke_test else None
    try:
        if args.smoke_test:
            tmp_root = pathlib.Path(temp_ctx.name)
            shard_dir = build_smoke_hybrid_shards(tmp_root)
            out_dir = tmp_root / "results_125m"
            cfg = SSENNConfig(
                vocab_size=512,
                num_layers=4,
                d_init=64,
                d_max=128,
                delta_d=32,
                block_size=16,
                top_k_blocks=2,
                max_seq_len=64,
                attn_top_k=8,
                salient_anchor_slots=2,
                num_attn_heads=8,
                dtype=dtype,
                kahan_mode=kahan_mode,
                device=device,
                seed=42,
            )
            epochs = int(args.epochs) if args.epochs != 6 else 4
            batch_size = 16
            ckpt_steps = 4
            expected_train_params = None
        else:
            shard_dir = pathlib.Path(args.shard_dir)
            out_dir = pathlib.Path(args.output_dir)
            cfg = SSENNConfig(
                vocab_size=12288,
                num_layers=12,
                d_init=640,
                d_max=1024,
                delta_d=128,
                block_size=16,
                top_k_blocks=8,
                max_seq_len=256,
                attn_top_k=16,
                salient_anchor_slots=4,
                num_attn_heads=8,
                dtype=dtype,
                kahan_mode=kahan_mode,
                device=device,
                seed=42,
            )
            epochs = int(args.epochs)
            batch_size = int(args.batch_size)
            ckpt_steps = int(args.ckpt_every_steps)
            expected_train_params = 114_556_928

        out_dir.mkdir(parents=True, exist_ok=True)
        vault_dir = out_dir / "vault_125m"
        dual_hist_path = out_dir / "dual_domain_history.jsonl"
        tok = load_tokenizer(shard_dir / "tokenizer.json")
        results: dict = {
            "ast_zero_autograd_files": int(file_cnt),
            "accelerator": gpu_name,
            "device": device,
            "dtype": dtype,
            "kahan_mode": kahan_mode,
            "config": cfg.to_dict(),
        }

        with (
            MMapTokenDataset(shard_dir, shard_name="train_shard", verify_checksum=True) as train_ds,
            MMapTokenDataset(shard_dir, shard_name="val_stories_shard", verify_checksum=True) as val_st_ds,
            MMapTokenDataset(shard_dir, shard_name="val_wiki_shard", verify_checksum=True) as val_wk_ds,
        ):
            # ------------------------------------------------------------------
            # Experiment 1: Pre-Allocation Footprint Match & MMap Streaming Audit
            # ------------------------------------------------------------------
            print("\n[Experiment 1] Pre-Allocation Footprint Parity & MMap Streaming Audit (~25M Tokens)")
            est = cfg.estimate_memory_footprint()
            if expected_train_params is not None:
                assert est["max_train_params"] == expected_train_params, est["max_train_params"]

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            trainer = ElasticCloudTrainer(
                config=cfg,
                vault_dir=vault_dir,
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

            for sname, ds in [("train_shard", train_ds), ("val_stories_shard", val_st_ds), ("val_wiki_shard", val_wk_ds)]:
                assert ds.verify_integrity() is True, f"Integrity check failed on {sname}"

            probe_streamer = ResumableBatchStreamer(train_ds, batch_size=batch_size, shuffle=False, prefetch_batches=2)
            t_stream0 = time.perf_counter()
            streamed_tok = 0
            for _, xb, _ in probe_streamer.iter_epoch(epoch=1, start_cursor=0):
                streamed_tok += int(xb.numel())
            stream_tok_s = streamed_tok / max(time.perf_counter() - t_stream0, 1e-6)
            st_bpt, wk_bpt = _load_domain_bytes_per_tok(shard_dir, val_st_ds, val_wk_ds, tok)

            exp1 = {
                "max_train_params": int(est["max_train_params"]),
                "deployed_w0_params": int(est["deployed_w0_params"]),
                "active_params_per_token": int(est["active_params_per_token"]),
                "training_state_elements": int(est["training_state_elements"]),
                "dfa_memory_savings_pct": float(est["dfa_memory_savings_pct"]),
                "exact_live_buffer_params_at_max": int(est["exact_live_buffer_params_at_max"]),
                "actual_live_buffer_params": int(brk_init["live_buffer_params"]),
                "est_total_train_with_kahan_mb": float(est["total_train_with_kahan_mb"]),
                "actual_total_live_mb": float(brk_init["total_live_mb"]),
                "est_frozen_w0_mb": float(est["frozen_w0_mb"]),
                "train_stored_tokens": int(train_ds.total_stored_tokens),
                "train_valid_tokens": int(train_ds.total_valid_tokens),
                "val_stories_stored_tokens": int(val_st_ds.total_stored_tokens),
                "val_wiki_stored_tokens": int(val_wk_ds.total_stored_tokens),
                "stories_bytes_per_tok": float(st_bpt),
                "wiki_bytes_per_tok": float(wk_bpt),
                "mmap_stream_tok_s": float(stream_tok_s),
                "mmap_meta_ram_kb": float((shard_dir / "train_shard.meta.json").stat().st_size / 1024.0),
            }
            results["experiment_1_audit"] = exp1
            print(
                f"  * Params Match    : Max Train {exp1['max_train_params']:,} | Deployed W0 {exp1['deployed_w0_params']:,} "
                f"(Active/Tok {exp1['active_params_per_token']:,}) | Exact Buffers {exp1['actual_live_buffer_params']:,} == {exp1['exact_live_buffer_params_at_max']:,}"
            )
            print(
                f"  * Live RAM/VRAM   : {exp1['actual_total_live_mb']:.2f} MB (Est Train+Kahan {exp1['est_total_train_with_kahan_mb']:.2f} MB) | "
                f"MMap Stream: {stream_tok_s:,.0f} tok/s | Integrity: PASS"
            )

            # ------------------------------------------------------------------
            # Experiment 2: Unified 6-Epoch Foundation Training + Dual Val Eval
            # ------------------------------------------------------------------
            print(f"\n[Experiment 2] Unified Foundation Training ({epochs} Epochs, sleep=0, Dual Stories + Wiki Eval)")
            has_prior_ckpt = trainer.vault.latest_path.exists() or trainer.vault.backup_path.exists()
            if not has_prior_ckpt:
                trainer.calibrate_salience_from_dataset(
                    train_ds, pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id
                )

            dual_epochs: list[dict] = []
            if has_prior_ckpt and dual_hist_path.exists():
                for line in dual_hist_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        dual_epochs.append(json.loads(line))

            epoch_clock = [time.perf_counter()]
            orig_append = trainer.vault.append_telemetry

            def _hooked_append_telemetry(record: dict) -> None:
                orig_append(record)
                if record.get("event") == "epoch_end":
                    ep_num = int(record["epoch"])
                    wk_m = evaluate_validation_batches(trainer.model, val_wk_ds, pad_id=tok.pad_id, batch_size=batch_size)
                    st_ppl = max(float(record["val_ppl"]), 1e-6)
                    st_m = {
                        "acc_pct": float(record["val_acc_pct"]),
                        "ce": math.log(st_ppl),
                        "ppl": st_ppl,
                        "valid_tokens": float(val_st_ds.total_valid_tokens),
                    }
                    dual = _dual_metrics_summary(st_m, wk_m, st_bpt, wk_bpt)
                    ep_vram = (
                        float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                        if device.startswith("cuda") and torch.cuda.is_available()
                        else 0.0
                    )
                    ep_dt = max(time.perf_counter() - epoch_clock[0], 1e-6)
                    epoch_clock[0] = time.perf_counter()
                    entry = {
                        "epoch": ep_num,
                        "global_step": int(record["global_step"]),
                        "train_acc_pct": float(record["train_acc_pct"]),
                        "train_ce": float(record["train_ce"]),
                        "active_dims": list(record["active_dims"]),
                        "grew": bool(record["grew"]),
                        "epoch_wall_s": float(ep_dt),
                        "vram_peak_mb": ep_vram,
                        **dual,
                    }
                    dual_epochs[:] = [d for d in dual_epochs if int(d["epoch"]) != ep_num] + [entry]
                    with open(dual_hist_path, "w", encoding="utf-8") as df:
                        for row in dual_epochs:
                            df.write(json.dumps(row) + "\n")
                    print(
                        f"    [Dual Eval Ep {ep_num:02d}] Stories: {entry['stories_val_acc_pct']:.2f}% "
                        f"(PPL {entry['stories_val_ppl']:.2f}, BPB {entry['stories_val_bpb']:.3f}) | "
                        f"Wiki: {entry['wiki_val_acc_pct']:.2f}% (PPL {entry['wiki_val_ppl']:.2f}, BPB {entry['wiki_val_bpb']:.3f}) | "
                        f"Combined PPL {entry['combined_weighted_ppl']:.2f} | Dims {entry['active_dims']} | VRAM {ep_vram:.1f} MB",
                        flush=True,
                    )

            trainer.vault.append_telemetry = _hooked_append_telemetry
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _sync_device(device)
            t_fit0 = time.perf_counter()
            fit_res = trainer.fit(
                train_dataset=train_ds,
                val_batches=val_st_ds,
                epochs=epochs,
                start_epoch=1,
                batch_size=batch_size,
                pad_id=tok.pad_id,
                auto_resume=True,
                stop_after_steps=args.max_steps,
                save_on_stop_after_steps=bool(args.max_steps),
            )
            _sync_device(device)
            fit_wall_s = max(time.perf_counter() - t_fit0, 1e-6)
            vram_peak_mb = (
                float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                if device.startswith("cuda") and torch.cuda.is_available()
                else 0.0
            )
            stiff_stats = _active_stiffness_stats(trainer.model)

            probe_trainer = ElasticCloudTrainer(config=cfg, vault_dir=vault_dir, keep_best_frozen=False, verbose=False)
            resume_chk = probe_trainer.restore_from_vault()
            resume_diff = _max_state_abs_diff(trainer.model, probe_trainer.model) if resume_chk["resumed"] else float("nan")
            del probe_trainer
            gc.collect()

            final_st = evaluate_validation_batches(trainer.model, val_st_ds, pad_id=tok.pad_id, batch_size=batch_size)
            final_wk = evaluate_validation_batches(trainer.model, val_wk_ds, pad_id=tok.pad_id, batch_size=batch_size)
            final_dual = _dual_metrics_summary(final_st, final_wk, st_bpt, wk_bpt)
            exp2 = {
                "status": fit_res["status"],
                "wall_s": float(fit_wall_s),
                "tok_s": float(fit_res["tok_s"]),
                "vram_peak_mb": float(vram_peak_mb),
                "final_step": int(fit_res["final_global_step"]),
                "last_train_acc_pct": float(fit_res["last_train_acc_pct"]),
                "last_train_ce": float(fit_res["last_train_ce"]),
                "allocated_dims": list(trainer.model.get_allocated_dimensions()),
                "growth_events": list(trainer.model.growth_log),
                "mean_stiffness_k": float(stiff_stats["k_bar"]),
                "atomic_vault_resume_diff": float(resume_diff),
                "dual_epochs_history": dual_epochs,
                **final_dual,
            }
            results["experiment_2_foundation_training"] = exp2
            print(
                f"  * Training Done   : {exp2['tok_s']:,.1f} tok/s | VRAM Peak: {vram_peak_mb:.1f} MB | "
                f"Stories PPL {exp2['stories_val_ppl']:.2f} (vs 37.37) | Wiki PPL {exp2['wiki_val_ppl']:.2f} (vs 363.5) | "
                f"Atomic Diff={resume_diff:.2e}"
            )

            # ------------------------------------------------------------------
            # Experiment 3: Production Freeze & Non-Destructive W0 Export
            # ------------------------------------------------------------------
            print("\n[Experiment 3] Production Freeze (prune_dormant=True), Export Parity & Frozen Dual-Domain Eval")
            pre_brk = trainer.model.count_parameters_breakdown()
            nd_state, _ = trainer.model.export_frozen_w0_state_nondestructive(map_location="cpu")
            assert not trainer.model.frozen

            freeze_stats = trainer.model.production_freeze(prune_dormant=True)
            live_frozen = trainer.model.export_frozen_w0_state()
            nd_diff = 0.0
            for tkey, nd_val in nd_state["w0_tensors"].items():
                ref_val = live_frozen["w0_tensors"][tkey]
                if isinstance(nd_val, torch.Tensor):
                    nd_diff = max(nd_diff, float((nd_val.float() - ref_val.detach().cpu().float()).abs().max().item()))
                elif isinstance(nd_val, list):
                    for l_i, nd_lt in enumerate(nd_val):
                        nd_diff = max(nd_diff, float((nd_lt.float() - ref_val[l_i].detach().cpu().float()).abs().max().item()))

            final_w0_path = out_dir / "final_125m_v4_1_frozen_w0.pt"
            torch.save(nd_state, final_w0_path)
            frz_st = evaluate_validation_batches(trainer.model, val_st_ds, pad_id=tok.pad_id, batch_size=batch_size)
            frz_wk = evaluate_validation_batches(trainer.model, val_wk_ds, pad_id=tok.pad_id, batch_size=batch_size)
            frz_dual = _dual_metrics_summary(frz_st, frz_wk, st_bpt, wk_bpt)

            exp3 = {
                "before_live_mb": float(pre_brk["total_live_mb"]),
                "before_kahan_mb": float(pre_brk["kahan_residual_mb"]),
                "after_frozen_mb": float(freeze_stats["after_live_mb"]),
                "memory_saved_pct": float(freeze_stats["memory_saved_pct"]),
                "before_params": int(freeze_stats["before_params"]),
                "after_params": int(freeze_stats["after_params"]),
                "compacted_dims": list(freeze_stats["compacted_dims"]),
                "nondestructive_export_diff": float(nd_diff),
                "final_w0_disk_mb": float(final_w0_path.stat().st_size / (1024 * 1024)),
                **frz_dual,
            }
            results["experiment_3_freeze"] = exp3
            print(
                f"  * Memory Pruned   : {exp3['before_live_mb']:.2f} MB (Kahan {exp3['before_kahan_mb']:.2f} MB) -> "
                f"{exp3['after_frozen_mb']:.2f} MB (-{exp3['memory_saved_pct']:.1f}%) | Disk: {exp3['final_w0_disk_mb']:.2f} MB | Diff={nd_diff:.2e}"
            )
            print(
                f"  * Frozen W0 Dual  : Stories {exp3['stories_val_acc_pct']:.2f}% (PPL {exp3['stories_val_ppl']:.2f}, BPB {exp3['stories_val_bpb']:.3f}) | "
                f"Wiki {exp3['wiki_val_acc_pct']:.2f}% (PPL {exp3['wiki_val_ppl']:.2f}, BPB {exp3['wiki_val_bpb']:.3f}) | "
                f"Combined PPL {exp3['combined_weighted_ppl']:.2f}"
            )

        # ------------------------------------------------------------------
        # Experiment 4: O(1) vs O(T^2) Rollout & Dual-Domain Live Generation
        # ------------------------------------------------------------------
        print("\n[Experiment 4] O(1) Cache vs O(T^2) Speedup (+128 & +256) & Dual-Domain Generation")
        _ = trainer.model.generate([tok.bos_id, 10, 11], max_new_tokens=8, eos_id=-1, use_fast_cache=True)
        _ = trainer.model.generate([tok.bos_id, 10, 11], max_new_tokens=8, eos_id=-1, use_fast_cache=False)

        horizon_specs = [
            (tok.encode("Once upon a time there was a", add_bos=True, add_eos=False), 32 if args.smoke_test else 128),
            ([tok.bos_id], max(8, min(48 if args.smoke_test else 256, cfg.max_seq_len - 1))),
        ]
        exp4_horizons: list[dict] = []
        for p_ids, req_h in horizon_specs:
            h_eff = min(req_h, max(8, cfg.max_seq_len - len(p_ids)))
            _sync_device(device)
            tf0 = time.perf_counter()
            out_fast = trainer.model.generate(p_ids, max_new_tokens=h_eff, eos_id=-1, use_fast_cache=True)
            _sync_device(device)
            fast_s = max(time.perf_counter() - tf0, 1e-6)

            _sync_device(device)
            ts0 = time.perf_counter()
            out_slow = trainer.model.generate(p_ids, max_new_tokens=h_eff, eos_id=-1, use_fast_cache=False)
            _sync_device(device)
            slow_s = max(time.perf_counter() - ts0, 1e-6)

            n_gen = max(1, len(out_fast) - len(p_ids))
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
            exp4_horizons.append(rec)
            print(
                f"  * Horizon +{n_gen:3d} tok : O(1) {fast_tok_s:8.1f} tok/s ({rec['o1_ms_per_tok']:.2f} ms/tok) vs "
                f"O(T^2) {slow_tok_s:8.1f} tok/s ({speedup_x:.2f}x speedup) | Parity: {match_pct:.2f}%"
            )

        dual_prompts = [
            ("story", "Once upon a time there was a"),
            ("story", "Lily wanted to help her"),
            ("wiki", "The Earth is the"),
            ("wiki", "Water is a"),
            ("wiki", "Isaac Newton was a"),
        ]
        dual_samples: list[dict] = []
        for domain, prompt_str in dual_prompts:
            pids = tok.encode(prompt_str, add_bos=True, add_eos=False)
            gen_ids = trainer.model.generate(
                pids,
                max_new_tokens=32 if args.smoke_test else 64,
                eos_id=tok.eos_id,
                use_fast_cache=True,
                frequency_penalty=0.35,
                presence_penalty=0.15,
                penalty_window=64,
            )
            dual_samples.append({
                "domain": domain,
                "prompt": prompt_str,
                "generated": tok.decode(gen_ids, skip_special_tokens=True),
            })
        results["experiment_4_inference"] = {
            "horizons": exp4_horizons,
            "dual_domain_samples": dual_samples,
        }

        json_path = out_dir / "bench_125m_results.json"
        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        md_lines = [
            "# SS-ENN v4.1.0 Base-125M Unified Hybrid Foundation GPU Benchmark Report",
            f"- **Accelerator:** `{gpu_name}` (`device={device}`, `dtype={dtype}`, `kahan_mode={kahan_mode}`, `heads={cfg.num_attn_heads}`)",
            f"- **AST Zero-Autograd:** `PASS ({file_cnt} files)`",
            f"- **Params:** `Train {exp1['max_train_params']:,}` | `Deployed W0 {exp1['deployed_w0_params']:,}` | `Active/Tok {exp1['active_params_per_token']:,}` -> `{exp3['after_params']:,}` pruned W0",
            "",
            "| Experiment | Key Metric | Value |",
            "| :--- | :--- | :--- |",
            f"| **1. Audit & MMap** | Live Buffers / Stream Speed | `{exp1['actual_live_buffer_params']:,}` exact match / `{exp1['mmap_stream_tok_s']:,.0f} tok/s` |",
            f"| **2. Train ({epochs}ep)** | GPU Speed / VRAM Peak / Resume Diff | `{exp2['tok_s']:,.1f} tok/s` / `{exp2['vram_peak_mb']:.1f} MB` / `{exp2['atomic_vault_resume_diff']:.2e}` |",
            f"| **3. Frozen Stories** | Val Acc / PPL / BPB (vs `37.37`) | `{exp3['stories_val_acc_pct']:.2f}%` / **`{exp3['stories_val_ppl']:.2f}`** / `{exp3['stories_val_bpb']:.3f}` |",
            f"| **3b. Frozen Wiki** | Val Acc / PPL / BPB (vs `363.5`) | `{exp3['wiki_val_acc_pct']:.2f}%` / **`{exp3['wiki_val_ppl']:.2f}`** / `{exp3['wiki_val_bpb']:.3f}` |",
            f"| **3c. W0 Compaction** | Memory Saved / Disk / Export Diff | `{exp3['before_live_mb']:.1f} -> {exp3['after_frozen_mb']:.1f} MB (-{exp3['memory_saved_pct']:.1f}%)` / `{exp3['final_w0_disk_mb']:.2f} MB` / `{exp3['nondestructive_export_diff']:.2e}` |",
            f"| **4. O(1) vs O(T^2)** | Max Horizon Speedup & Match | `{exp4_horizons[-1]['speedup_x']:.2f}x ({exp4_horizons[-1]['o1_tok_s']:.1f} tok/s)` / `{exp4_horizons[-1]['sequence_match_pct']:.2f}%` |",
        ]
        md_path = out_dir / "bench_125m_report.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"\nSaved Base-125M scientific artifacts:\n  - {json_path}\n  - {md_path}")
        return results
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SS-ENN v4.1 Base-125M Unified Hybrid Foundation GPU Benchmark Suite")
    p.add_argument("--shard-dir", type=str, default="/content/local_125m/shards_25m_hybrid_seq256", help="Directory with train_shard, val_stories_shard, val_wiki_shard.")
    p.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/SS_ENN_Workspace/exp_125m_v4_1_a100/results_125m", help="Output vault & report directory.")
    p.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu.")
    p.add_argument("--dtype", type=str, default="auto", choices=("auto", "bfloat16", "float16", "float32"), help="Weight precision (auto prefers bfloat16 on A100/L4).")
    p.add_argument("--epochs", type=int, default=6, help="Unified foundation epochs.")
    p.add_argument("--batch-size", type=int, default=64, help="Training/validation minibatch size.")
    p.add_argument("--ckpt-every-steps", type=int, default=200, help="Periodic atomic vault checkpoint interval.")
    p.add_argument("--max-steps", type=int, default=None, help="Optional max step cap.")
    p.add_argument("--smoke-test", action="store_true", help="Run fast self-contained CPU smoke test.")
    return p.parse_args(argv)


if __name__ == "__main__":
    run_suite(parse_args())

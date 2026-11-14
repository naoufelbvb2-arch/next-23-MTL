"""
benchmarks/bench_v3_trainer_fault_tolerance.py
SS-ENN v3.0 Phase 3.3 Benchmark: Fault-Tolerant Cloud/Colab Trainer, Atomic
Checkpoint Vault, Preemption Watchdog & Memory-Mapped Binary Shard Streamer.

Audits 4 mandatory cloud-training telemetry gates (Zero-Autograd):
  - Table 1: `MMapTokenDataset` & `ResumableBatchStreamer` throughput (tok/s),
    O(1) RAM overhead vs disk shard size, and deterministic cursor parity.
  - Table 2: Uninterrupted Run vs Sudden Mid-Epoch Kill (`auto_resume=True`)
    bitwise weight parity (`0.00e+00` Pre-Freeze & Frozen W0) across `FP32` and
    `BF16 + anchor_k`.
  - Table 3: Corrupted `checkpoint_latest.pt` (both truncation & bit-flip)
    self-healing fallback to `checkpoint_backup.pt` (100% crash-free recovery).
  - Table 4: Real `SIGTERM` Preemption Watchdog emergency save, compacted
    `best_frozen_w0.pt` export, and structured `training_telemetry.jsonl` audit.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bench_scale_english_cpu import (  # noqa: E402
    build_minibatch_tensors,
    verify_all_repo_files_zero_autograd,
)
from benchmarks.english_dataset import build_english_tinystories_corpus  # noqa: E402
from engine.tokenizer import EnglishSubwordTokenizer  # noqa: E402
from models.config_v3 import SSENNConfig  # noqa: E402
from models.model_v3_production import SSENNProductionModel  # noqa: E402
from trainer import (  # noqa: E402
    AtomicCheckpointVault,
    BinaryShardBuilder,
    ElasticCloudTrainer,
    MMapTokenDataset,
    PreemptionSignalHandler,
    ResumableBatchStreamer,
    evaluate_validation_batches,
)


def _max_model_buffer_diff(m1: SSENNProductionModel, m2: SSENNProductionModel) -> float:
    max_d = 0.0
    b1 = dict(m1.named_buffers())
    b2 = dict(m2.named_buffers())
    for name in set(b1.keys()) | set(b2.keys()):
        t1, t2 = b1.get(name), b2.get(name)
        if (t1 is None) != (t2 is None):
            return float("inf")
        if t1 is not None and t2 is not None:
            diff = float((t1.float() - t2.float()).abs().max().item())
            if diff > max_d:
                max_d = diff
    if m1.compensator is not None and m2.compensator is not None:
        for sname in ("_res_w", "_res_w0", "_res_k"):
            d1 = getattr(m1.compensator, sname)
            d2 = getattr(m2.compensator, sname)
            for ck in set(d1.keys()) | set(d2.keys()):
                if (ck in d1) != (ck in d2):
                    return float("inf")
                diff = float((d1[ck].float() - d2[ck].float()).abs().max().item())
                if diff > max_d:
                    max_d = diff
    return max_d


def _max_frozen_model_diff(m1: SSENNProductionModel, m2: SSENNProductionModel) -> float:
    return _max_model_buffer_diff(m1, m2)


@torch.no_grad()
def run_v3_trainer_fault_tolerance_benchmark(
    epochs: int = 14,
    batch_size: int = 32,
    seq_len: int = 36,
    verbose: bool = True,
) -> dict:
    torch.set_grad_enabled(False)

    ast_ok, n_files = verify_all_repo_files_zero_autograd()
    if verbose:
        print("=" * 112)
        print("SS-ENN v3.0 PHASE 3.3 BENCHMARK: FAULT-TOLERANT CLOUD TRAINER, ATOMIC VAULT & MMAP STREAMER")
        print("=" * 112)
        print(
            f"[Gate Check] AST Zero-Autograd Verification : "
            f"{'PASS' if ast_ok else 'FAIL'} ({n_files} Python files verified)"
        )

    corpus = build_english_tinystories_corpus(target_min_tokens=20000, train_ratio=0.90, seed=42)
    all_stories = corpus["train_stories"] + corpus["val_stories"]
    tokenizer = EnglishSubwordTokenizer(max_vocab_size=2048)
    tokenizer.fit(all_stories, pad_to_max_vocab=True)
    pad_id = tokenizer.pad_id

    val_batches, _, _ = build_minibatch_tensors(
        corpus["val_stories"], tokenizer, seq_len=seq_len, batch_size=batch_size
    )

    with tempfile.TemporaryDirectory(prefix="ssenn_p33_") as tmp_dir_str:
        tmp_root = Path(tmp_dir_str)
        shard_dir = tmp_root / "shards"

        # =====================================================================
        # TABLE 1: MMapTokenDataset & ResumableBatchStreamer Speed, RAM & Cursor
        # =====================================================================
        tiny_meta = BinaryShardBuilder.build_from_stories(
            stories=corpus["train_stories"],
            tokenizer=tokenizer,
            seq_len=seq_len,
            output_dir=shard_dir,
            shard_name="tinystories_train",
        )

        # Build a larger multi-million-token binary shard to verify O(1) RAM scaling
        synth_rng = np.random.default_rng(2026)
        synth_rows = synth_rng.integers(
            low=3, high=tokenizer.vocab_size, size=(16384, 129), dtype=np.uint16
        )
        large_meta = BinaryShardBuilder.build_from_sequences(
            sequences=synth_rows,
            vocab_size=tokenizer.vocab_size,
            seq_len=128,
            output_dir=shard_dir,
            shard_name="scale_2m_tokens",
            pad_id=pad_id,
        )

        table1_rows: list[dict] = []
        with (
            MMapTokenDataset(shard_dir, shard_name="tinystories_train", verify_checksum=True) as ds_tiny,
            MMapTokenDataset(shard_dir, shard_name="scale_2m_tokens", verify_checksum=True) as ds_large,
        ):
            for label, ds, b_sz in (
                ("TinyStories Shard (uint16)", ds_tiny, batch_size),
                ("Scale 2.1M-Tok Shard (uint16)", ds_large, 128),
            ):
                streamer = ResumableBatchStreamer(
                    ds, batch_size=b_sz, shuffle=True, base_seed=777, prefetch_batches=2
                )
                tracemalloc.start()
                t0 = time.perf_counter()
                tok_read = 0
                full_batches: list[tuple[int, torch.Tensor, torch.Tensor]] = []
                for b_idx, x_b, y_b in streamer.iter_epoch(epoch=1):
                    tok_read += int(x_b.numel())
                    if len(ds) < 1000:
                        full_batches.append((b_idx, x_b.clone(), y_b.clone()))
                elapsed_s = max(time.perf_counter() - t0, 1e-6)
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                cursor_match_pct = 100.0
                if len(ds) < 1000:
                    # Verify mid-epoch state save/restore produces zero skipped & zero repeated batches
                    s_mid = ResumableBatchStreamer(
                        ds, batch_size=b_sz, shuffle=True, base_seed=777, prefetch_batches=2
                    )
                    collected: list[tuple[int, torch.Tensor, torch.Tensor]] = []
                    saved_state = None
                    split_k = len(full_batches) // 2
                    for b_idx, x_b, y_b in s_mid.iter_epoch(epoch=1):
                        collected.append((b_idx, x_b.clone(), y_b.clone()))
                        if len(collected) == split_k:
                            saved_state = s_mid.get_streamer_state()
                            break
                    s_resumed = ResumableBatchStreamer(
                        ds, batch_size=b_sz, shuffle=True, base_seed=9999, prefetch_batches=2
                    )
                    s_resumed.load_streamer_state(saved_state)
                    for b_idx, x_b, y_b in s_resumed.iter_epoch():
                        collected.append((b_idx, x_b.clone(), y_b.clone()))

                    exact = len(collected) == len(full_batches)
                    if exact:
                        for (i1, x1, y1), (i2, x2, y2) in zip(full_batches, collected):
                            if i1 != i2 or not torch.equal(x1, x2) or not torch.equal(y1, y2):
                                exact = False
                                break
                    cursor_match_pct = 100.0 if exact else 0.0
                else:
                    # Standalone cursor slice verification on large shard (split at batch 10)
                    s_part = ResumableBatchStreamer(
                        ds, batch_size=b_sz, shuffle=True, base_seed=777, prefetch_batches=2
                    )
                    first_12 = []
                    mid_st = None
                    for b_idx, x_b, y_b in s_part.iter_epoch(epoch=1):
                        first_12.append((b_idx, x_b.clone(), y_b.clone()))
                        if b_idx == 5:
                            mid_st = s_part.get_streamer_state()
                        if b_idx == 11:
                            break
                    s_rec = ResumableBatchStreamer(
                        ds, batch_size=b_sz, shuffle=True, base_seed=1, prefetch_batches=2
                    )
                    s_rec.load_streamer_state(mid_st)
                    tail_6 = []
                    for b_idx, x_b, y_b in s_rec.iter_epoch():
                        tail_6.append((b_idx, x_b.clone(), y_b.clone()))
                        if b_idx == 11:
                            break
                    exact_large = len(tail_6) == 6
                    if exact_large:
                        for (i1, x1, y1), (i2, x2, y2) in zip(first_12[6:12], tail_6):
                            if i1 != i2 or not torch.equal(x1, x2) or not torch.equal(y1, y2):
                                exact_large = False
                                break
                    cursor_match_pct = 100.0 if exact_large else 0.0

                table1_rows.append(
                    {
                        "label": label,
                        "num_sequences": len(ds),
                        "total_tokens": ds.total_stored_tokens,
                        "disk_mb": ds.disk_size_mb(),
                        "meta_ram_kb": float(ds.metadata_ram_bytes()) / 1024.0,
                        "peak_stream_mb": float(peak_bytes) / (1024.0 * 1024.0),
                        "stream_tok_s": float(tok_read) / elapsed_s,
                        "cursor_parity_pct": cursor_match_pct,
                        "sha256_ok": bool(ds.verify_integrity()),
                    }
                )

            if verbose:
                print("\n[Table 1] MMapTokenDataset (numpy.memmap) & ResumableBatchStreamer Telemetry")
                print("-" * 112)
                print(
                    f"{'Shard Workload':<29} | {'Tokens':>9} | {'Disk MB':>7} | "
                    f"{'Meta RAM':>8} | {'Stream Peak':>11} | {'Stream tok/s':>12} | {'Cursor Parity'}"
                )
                print("-" * 112)
                for r in table1_rows:
                    print(
                        f"{r['label']:<29} | {r['total_tokens']:>9,} | {r['disk_mb']:>7.2f} | "
                        f"{r['meta_ram_kb']:>6.2f} KB | {r['peak_stream_mb']:>8.2f} MB | "
                        f"{r['stream_tok_s']:>12,.1f} | {r['cursor_parity_pct']:>11.2f}%"
                    )
                print("-" * 112)

            # =================================================================
            # TABLE 2: Uninterrupted vs Mid-Epoch Sudden Kill Auto-Resume Parity
            # =================================================================
            batches_per_ep = (len(ds_tiny) + batch_size - 1) // batch_size
            table2_rows: list[dict] = []

            for dtype_name, kahan_mode, n_eps, kill_step in (
                ("float32", "off", epochs, 6 * batches_per_ep + batches_per_ep // 2),
                ("bfloat16", "anchor_k", 4, batches_per_ep + batches_per_ep // 2 + 1),
            ):
                cfg = SSENNConfig(
                    vocab_size=tokenizer.vocab_size,
                    num_layers=6 if dtype_name == "float32" else 4,
                    d_init=64,
                    d_max=128 if dtype_name == "float32" else 64,
                    delta_d=16 if dtype_name == "float32" else 0,
                    block_size=16,
                    top_k_blocks=2,
                    max_seq_len=max(seq_len, 64),
                    attn_top_k=8,
                    dtype=dtype_name,
                    kahan_mode=kahan_mode,
                    device="cpu",
                    seed=42,
                )

                v_unint = tmp_root / f"vault_unint_{dtype_name}"
                t_unint = ElasticCloudTrainer(
                    config=cfg, vault_dir=v_unint, checkpoint_every_steps=5, keep_best_frozen=True
                )
                t_unint.calibrate_salience_from_dataset(
                    ds_tiny, pad_id=pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id
                )
                res_unint = t_unint.fit(
                    train_dataset=ds_tiny,
                    val_batches=val_batches,
                    epochs=n_eps,
                    batch_size=batch_size,
                    pad_id=pad_id,
                    shuffle=True,
                    seed=1234,
                    prefetch_batches=2,
                    auto_resume=False,
                )

                # Simulated sudden mid-epoch crash AFTER periodic checkpoint (losses uncommitted step)
                v_kill = tmp_root / f"vault_kill_{dtype_name}"
                t_kill = ElasticCloudTrainer(
                    config=cfg, vault_dir=v_kill, checkpoint_every_steps=5, keep_best_frozen=True
                )
                t_kill.calibrate_salience_from_dataset(
                    ds_tiny, pad_id=pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id
                )
                res_killed = t_kill.fit(
                    train_dataset=ds_tiny,
                    val_batches=val_batches,
                    epochs=n_eps,
                    batch_size=batch_size,
                    pad_id=pad_id,
                    shuffle=True,
                    seed=1234,
                    prefetch_batches=2,
                    auto_resume=False,
                    stop_after_steps=kill_step,
                    save_on_stop_after_steps=False,
                )

                # Completely fresh trainer instance recovers automatically from vault
                t_resume = ElasticCloudTrainer(
                    config=cfg, vault_dir=v_kill, checkpoint_every_steps=5, keep_best_frozen=True
                )
                res_resumed = t_resume.fit(
                    train_dataset=ds_tiny,
                    val_batches=val_batches,
                    epochs=n_eps,
                    batch_size=batch_size,
                    pad_id=pad_id,
                    shuffle=True,
                    seed=1234,
                    prefetch_batches=2,
                    auto_resume=True,
                )

                pre_diff = _max_model_buffer_diff(t_unint.model, t_resume.model)
                t_unint.model.production_freeze(prune_dormant=True)
                t_resume.model.production_freeze(prune_dormant=True)
                frozen_diff = _max_frozen_model_diff(t_unint.model, t_resume.model)

                val_u = evaluate_validation_batches(t_unint.model, val_batches, pad_id=pad_id)
                val_r = evaluate_validation_batches(t_resume.model, val_batches, pad_id=pad_id)
                acc_match = (
                    100.0
                    if abs(val_u["acc_pct"] - val_r["acc_pct"]) < 1e-6
                    and abs(val_u["ppl"] - val_r["ppl"]) < 1e-6
                    else 0.0
                )

                table2_rows.append(
                    {
                        "dtype": f"{dtype_name} ({kahan_mode})",
                        "epochs": n_eps,
                        "kill_step": int(kill_step),
                        "recovered_step": int(res_resumed["resumed_global_step"]),
                        "resumed_epoch": int(res_resumed["resumed_epoch"]),
                        "resumed_cursor": int(res_resumed["resumed_batch_cursor"]),
                        "pre_freeze_diff": float(pre_diff),
                        "frozen_w0_diff": float(frozen_diff),
                        "val_acc_pct": float(val_r["acc_pct"]),
                        "val_ppl": float(val_r["ppl"]),
                        "val_parity_pct": float(acc_match),
                    }
                )

            if verbose:
                print("\n[Table 2] Uninterrupted vs Mid-Epoch Sudden Kill (`auto_resume=True`) Bitwise Parity")
                print("-" * 112)
                print(
                    f"{'Precision':<19} | {'Kill -> Recv':<16} | {'Epoch/Cursor':<12} | "
                    f"{'Pre-Freeze Diff':<15} | {'Frozen W0 Diff':<14} | {'Val Acc (PPL)'}"
                )
                print("-" * 112)
                for r in table2_rows:
                    kr = f"Step {r['kill_step']} -> {r['recovered_step']}"
                    ec = f"Ep{r['resumed_epoch']} / b={r['resumed_cursor']}"
                    ap = f"{r['val_acc_pct']:.2f}% ({r['val_ppl']:.2f})"
                    print(
                        f"{r['dtype']:<19} | {kr:<16} | {ec:<12} | "
                        f"{r['pre_freeze_diff']:<15.2e} | {r['frozen_w0_diff']:<14.2e} | {ap}"
                    )
                print("-" * 112)

            # =================================================================
            # TABLE 3: Corrupted Primary Checkpoint Self-Healing Backup Fallback
            # =================================================================
            cfg_t3 = SSENNConfig(
                vocab_size=tokenizer.vocab_size,
                num_layers=6,
                d_init=64,
                d_max=128,
                delta_d=16,
                block_size=16,
                top_k_blocks=2,
                max_seq_len=max(seq_len, 64),
                attn_top_k=8,
                dtype="float32",
                device="cpu",
                seed=55,
            )
            v_base_corr = tmp_root / "vault_corr_base"
            t_seed = ElasticCloudTrainer(
                config=cfg_t3, vault_dir=v_base_corr, checkpoint_every_steps=5, keep_best_frozen=True
            )
            t_seed.calibrate_salience_from_dataset(
                ds_tiny, pad_id=pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id
            )
            # Mid-epoch saves at 5..20..25 -> checkpoint_backup.pt=Step 20, checkpoint_latest.pt=Step 25
            t_seed.fit(
                train_dataset=ds_tiny,
                val_batches=val_batches,
                epochs=epochs,
                batch_size=batch_size,
                pad_id=pad_id,
                shuffle=True,
                seed=333,
                prefetch_batches=2,
                auto_resume=False,
                stop_after_steps=25,
                save_on_stop_after_steps=False,
            )

            table3_rows: list[dict] = []
            for fault_mode in ("truncate", "bitflip"):
                v_case = tmp_root / f"vault_corr_{fault_mode}"
                shutil.copytree(v_base_corr, v_case)
                vault_case = AtomicCheckpointVault(v_case)
                vault_case.corrupt_file_for_test(which="latest", mode=fault_mode)

                t_heal = ElasticCloudTrainer(
                    config=cfg_t3, vault_dir=v_case, checkpoint_every_steps=5, keep_best_frozen=True
                )
                res_heal = t_heal.fit(
                    train_dataset=ds_tiny,
                    val_batches=val_batches,
                    epochs=epochs,
                    batch_size=batch_size,
                    pad_id=pad_id,
                    shuffle=True,
                    seed=333,
                    prefetch_batches=2,
                    auto_resume=True,
                )
                finite_ok = all(
                    bool(torch.isfinite(buf).all().item())
                    for _, buf in t_heal.model.named_buffers()
                    if buf is not None
                )
                table3_rows.append(
                    {
                        "fault_mode": fault_mode,
                        "primary_corrupted": bool(res_heal["primary_corrupted"]),
                        "resume_source": str(res_heal["resume_source"]),
                        "recovered_step": int(res_heal["resumed_global_step"]),
                        "final_step": int(res_heal["final_global_step"]),
                        "final_val_acc": float(res_heal["final_val_acc_pct"]),
                        "final_val_ppl": float(res_heal["final_val_ppl"]),
                        "finite_ok": finite_ok,
                    }
                )

            if verbose:
                print("\n[Table 3] Corrupted `checkpoint_latest.pt` -> Self-Healing `checkpoint_backup.pt` Recovery")
                print("-" * 112)
                print(
                    f"{'Injected Fault Mode':<22} | {'Detected':<9} | {'Fallback Source':<17} | "
                    f"{'Recv -> End':<13} | {'Val Acc':>8} | {'Val PPL':>7} | {'Finite'}"
                )
                print("-" * 112)
                for r in table3_rows:
                    print(
                        f"{r['fault_mode']:<22} | {str(r['primary_corrupted']):<9} | "
                        f"{r['resume_source']:<17} | {f'{r['recovered_step']} -> {r['final_step']}':<13} | "
                        f"{r['final_val_acc']:>7.2f}% | {r['final_val_ppl']:>7.2f} | {r['finite_ok']}"
                    )
                print("-" * 112)

            # =================================================================
            # TABLE 4: SIGTERM Preemption Watchdog, best_frozen_w0.pt & Telemetry
            # =================================================================
            v_preempt = tmp_root / "vault_preempt"
            cfg_t4 = SSENNConfig(
                vocab_size=tokenizer.vocab_size,
                num_layers=6,
                d_init=64,
                d_max=128,
                delta_d=16,
                block_size=16,
                top_k_blocks=2,
                max_seq_len=max(seq_len, 64),
                attn_top_k=8,
                dtype="float32",
                device="cpu",
                seed=88,
            )
            t_pre = ElasticCloudTrainer(
                config=cfg_t4, vault_dir=v_preempt, checkpoint_every_steps=0, keep_best_frozen=True
            )
            t_pre.calibrate_salience_from_dataset(
                ds_tiny, pad_id=pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id
            )

            sig_used: dict[str, bool | str | None] = {"os_sigterm": False, "sig_name": None}

            def _trigger_preempt_at_22(
                step: int, ep: int, b_idx: int, trainer_obj: ElasticCloudTrainer, sh: PreemptionSignalHandler
            ) -> None:
                if step == 22:
                    sig_used["os_sigterm"] = sh.raise_os_sigterm()
                    sig_used["sig_name"] = sh.signal_name

            res_pre = t_pre.fit(
                train_dataset=ds_tiny,
                val_batches=val_batches,
                epochs=epochs,
                batch_size=batch_size,
                pad_id=pad_id,
                shuffle=True,
                seed=909,
                prefetch_batches=2,
                auto_resume=False,
                on_step_callback=_trigger_preempt_at_22,
            )

            # Fresh process resumes from the emergency preemption save cleanly
            t_post_pre = ElasticCloudTrainer(
                config=cfg_t4, vault_dir=v_preempt, checkpoint_every_steps=0, keep_best_frozen=True
            )
            res_post_pre = t_post_pre.fit(
                train_dataset=ds_tiny,
                val_batches=val_batches,
                epochs=epochs,
                batch_size=batch_size,
                pad_id=pad_id,
                shuffle=True,
                seed=909,
                prefetch_batches=2,
                auto_resume=True,
            )

            frozen_state = AtomicCheckpointVault(v_preempt).load_frozen_w0()
            reloaded_best = SSENNProductionModel.from_frozen_w0_state(frozen_state)
            best_eval = evaluate_validation_batches(reloaded_best, val_batches, pad_id=pad_id)
            telemetry_rows = AtomicCheckpointVault(v_preempt).read_telemetry()
            event_types = sorted({r.get("event", "") for r in telemetry_rows})

            table4 = {
                "preempt_status": res_pre["status"],
                "preempt_step": int(res_pre["final_global_step"]),
                "os_sigterm_raised": bool(sig_used["os_sigterm"]),
                "signal_name": str(sig_used["sig_name"]),
                "emergency_saves": int(res_pre["emergency_saves"]),
                "post_resume_source": str(res_post_pre["resume_source"]),
                "post_resume_step": int(res_post_pre["resumed_global_step"]),
                "final_step": int(res_post_pre["final_global_step"]),
                "best_frozen_w0_exists": bool(res_post_pre["best_frozen_w0_exists"]),
                "best_frozen_kb": float(AtomicCheckpointVault(v_preempt).best_frozen_path.stat().st_size) / 1024.0,
                "best_frozen_is_frozen": bool(reloaded_best.frozen),
                "best_frozen_has_no_train_bufs": (
                    reloaded_best.q_w is None
                    and reloaded_best.factorized_dfa is None
                    and reloaded_best.compensator is None
                ),
                "best_frozen_val_acc": float(best_eval["acc_pct"]),
                "best_frozen_val_ppl": float(best_eval["ppl"]),
                "telemetry_records": len(telemetry_rows),
                "telemetry_events": event_types,
            }

            if verbose:
                print("\n[Table 4] Preemption Watchdog (SIGTERM), `best_frozen_w0.pt` & `training_telemetry.jsonl` Audit")
                print("-" * 112)
                print(
                    f"  * Preemption Interrupt  : status='{table4['preempt_status']}' @ Step {table4['preempt_step']} | "
                    f"signal={table4['signal_name']} (os_native={table4['os_sigterm_raised']}) | "
                    f"emergency_saves={table4['emergency_saves']}"
                )
                print(
                    f"  * Seamless Resume       : source='{table4['post_resume_source']}' @ Step {table4['post_resume_step']} "
                    f"-> completed Step {table4['final_step']}"
                )
                print(
                    f"  * Best Frozen W0 Export : exists={table4['best_frozen_w0_exists']} | "
                    f"size={table4['best_frozen_kb']:.1f} KB | frozen={table4['best_frozen_is_frozen']} | "
                    f"Val Acc {table4['best_frozen_val_acc']:.2f}% (PPL {table4['best_frozen_val_ppl']:.2f})"
                )
                print(
                    f"  * Telemetry Stream      : records={table4['telemetry_records']} | "
                    f"events={table4['telemetry_events']}"
                )
                print("-" * 112)

    return {
        "ast_zero_autograd_ok": ast_ok,
        "ast_files_verified": n_files,
        "table1": table1_rows,
        "table2": table2_rows,
        "table3": table3_rows,
        "table4": table4,
    }


if __name__ == "__main__":
    run_v3_trainer_fault_tolerance_benchmark()

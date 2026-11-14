"""
trainer/colab_orchestrator.py
Fault-Tolerant Elastic Cloud/Colab Training Orchestrator for SS-ENN v3.0 Phase 3.3.

Connects:
  - `SSENNConfig` & `SSENNProductionModel` (`models/config_v3.py`, `models/model_v3_production.py`)
  - `MMapTokenDataset` & `ResumableBatchStreamer` (`trainer/mmap_dataset.py`)
  - `AtomicCheckpointVault` & `PreemptionSignalHandler` (`trainer/checkpoint_vault.py`)

Features:
  - Automatic vault discovery and bit-exact mid-epoch resumption (`auto_resume=True`)
  - Periodic step-based (`checkpoint_every_steps`) & wall-clock (`checkpoint_every_seconds`)
    atomic checkpoints including running epoch loss accumulators
  - Non-destructive clone-and-freeze best `Frozen W0` artifact exporter (`best_frozen_w0.pt`)
  - Graceful `SIGINT`/`SIGTERM` preemption emergency save handler
  - Optional inter-epoch dual-stage sleep consolidation (`sleep_between_epochs`)
  - Append-only JSONL telemetry stream (`training_telemetry.jsonl`)

Strictly Zero-Autograd.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from models.config_v3 import SSENNConfig
from models.model_v3_production import SSENNProductionModel

from .checkpoint_vault import AtomicCheckpointVault, PreemptionSignalHandler
from .mmap_dataset import MMapTokenDataset, ResumableBatchStreamer


@torch.no_grad()
def evaluate_validation_batches(
    model: SSENNProductionModel,
    val_batches: Sequence[tuple[torch.Tensor, torch.Tensor]] | MMapTokenDataset | None,
    pad_id: int = 0,
    batch_size: int = 32,
) -> dict[str, float]:
    """Side-effect-free validation accuracy, mean CE loss, and perplexity (supports O(1) RAM MMap streaming)."""
    if val_batches is None or len(val_batches) == 0:
        return {"acc_pct": 0.0, "ce": 0.0, "ppl": 1.0, "valid_tokens": 0}
    correct = 0
    valid = 0
    ce_sum = 0.0
    dev = model.model_device

    if isinstance(val_batches, MMapTokenDataset):
        n_val = len(val_batches)
        step_bs = max(1, int(batch_size))
        batch_iter = (
            val_batches.get_batch(list(range(start, min(start + step_bs, n_val))))
            for start in range(0, n_val, step_bs)
        )
    else:
        batch_iter = iter(val_batches)

    for x_b, y_b in batch_iter:
        x_d = x_b.to(device=dev)
        y_d = y_b.to(device=dev)
        logits = model.forward(x_d)
        v_mask = y_d != pad_id
        if not v_mask.any():
            continue
        flat_l = logits[v_mask].float()
        flat_t = y_d[v_mask]
        ce_sum += float(F.cross_entropy(flat_l, flat_t, reduction="sum").item())
        correct += int((flat_l.argmax(dim=-1) == flat_t).sum().item())
        valid += int(v_mask.sum().item())
    mean_ce = ce_sum / max(valid, 1)
    acc = (float(correct) / max(valid, 1)) * 100.0
    return {
        "acc_pct": acc,
        "ce": mean_ce,
        "ppl": math.exp(min(mean_ce, 20.0)),
        "valid_tokens": valid,
    }


class ElasticCloudTrainer:
    """
    Production fault-tolerant orchestrator for Zero-Autograd SS-ENN training
    on local CPUs, Colab T4/L4/A100 GPUs, and preemptible cloud instances.
    """

    def __init__(
        self,
        config: SSENNConfig,
        vault_dir: str | Path,
        checkpoint_every_steps: int = 0,
        checkpoint_every_seconds: float = 0.0,
        keep_best_frozen: bool = True,
        sleep_between_epochs: bool = False,
        sleep_every_epochs: int = 0,
        sleep_min_epoch: int = 0,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.vault = AtomicCheckpointVault(vault_dir)
        self.model = SSENNProductionModel(config)
        self.checkpoint_every_steps = int(checkpoint_every_steps)
        self.checkpoint_every_seconds = float(checkpoint_every_seconds)
        self.keep_best_frozen = bool(keep_best_frozen)
        self.sleep_every_epochs = int(sleep_every_epochs) if sleep_every_epochs > 0 else (1 if sleep_between_epochs else 0)
        self.sleep_between_epochs = bool(self.sleep_every_epochs > 0)
        self.sleep_min_epoch = max(0, int(sleep_min_epoch))
        self.verbose = bool(verbose)

        self.global_step: int = 0
        self.best_val_acc: float = -1.0
        self.best_val_ppl: float = float("inf")
        self.sleep_phase_counter: int = 1
        self.preemption_saves: int = 0
        self.corrupt_recoveries: int = 0
        self.best_frozen_exports: int = 0
        self._current_epoch_accum: dict = {
            "epoch": 1,
            "correct": 0,
            "valid": 0,
            "ce_sum": 0.0,
            "steps": 0,
            "tokens": 0,
        }

    @torch.no_grad()
    def calibrate_salience_from_dataset(
        self,
        dataset: MMapTokenDataset,
        pad_id: int = 0,
        bos_id: int | None = 1,
        eos_id: int | None = 2,
        max_sequences: int = 512,
    ) -> None:
        """Calibrates Zipf token salience from the memory-mapped dataset when starting fresh."""
        n = min(len(dataset), max_sequences)
        if n <= 0:
            return
        x_b, y_b = dataset.get_batch(list(range(n)))
        full_seqs = torch.cat([x_b, y_b[:, -1:]], dim=1)
        self.model.calibrate_salience_from_tokens(
            full_seqs.tolist(), pad_id=pad_id, bos_id=bos_id, eos_id=eos_id
        )

    @torch.no_grad()
    def build_checkpoint_payload(self, streamer: ResumableBatchStreamer | None) -> dict:
        """Captures full bitwise model, Kahan residuals, streamer cursor, and epoch sums."""
        return {
            "model_state": self.model.get_full_checkpoint_state(),
            "streamer_state": streamer.get_streamer_state() if streamer is not None else None,
            "trainer_state": {
                "global_step": int(self.global_step),
                "best_val_acc": float(self.best_val_acc),
                "best_val_ppl": float(self.best_val_ppl),
                "sleep_phase_counter": int(self.sleep_phase_counter),
                "preemption_saves": int(self.preemption_saves),
                "corrupt_recoveries": int(self.corrupt_recoveries),
                "best_frozen_exports": int(self.best_frozen_exports),
                "epoch_accum": {
                    "epoch": int(self._current_epoch_accum["epoch"]),
                    "correct": int(self._current_epoch_accum["correct"]),
                    "valid": int(self._current_epoch_accum["valid"]),
                    "ce_sum": float(self._current_epoch_accum["ce_sum"]),
                    "steps": int(self._current_epoch_accum["steps"]),
                    "tokens": int(self._current_epoch_accum["tokens"]),
                },
            },
        }

    @torch.no_grad()
    def restore_from_vault(
        self, streamer: ResumableBatchStreamer | None = None
    ) -> dict:
        """
        Attempts to restore the latest verified checkpoint from `AtomicCheckpointVault`
        (with automatic backup fallback on corruption).
        """
        loaded = self.vault.load_latest_valid_checkpoint()
        if not loaded["found"] or loaded["payload"] is None:
            return {
                "resumed": False,
                "source": None,
                "primary_corrupted": False,
                "resumed_global_step": 0,
                "resumed_epoch": 1,
                "resumed_batch_cursor": 0,
            }

        target_dev = self.config.resolve_torch_device()
        target_dtype = str(self.config.dtype)
        payload = loaded["payload"]
        self.model = SSENNProductionModel.from_checkpoint_state(payload["model_state"])
        cur_dev = self.model.model_device
        dev_mismatch = (cur_dev.type != target_dev.type) or (
            target_dev.index is not None and cur_dev.index != target_dev.index
        )
        if dev_mismatch or self.model.config.dtype != target_dtype:
            self.model.to_device_and_dtype(device=target_dev, dtype=target_dtype)
        self.config = self.model.config

        ts = payload.get("trainer_state", {})
        self.global_step = int(ts.get("global_step", 0))
        self.best_val_acc = float(ts.get("best_val_acc", -1.0))
        self.best_val_ppl = float(ts.get("best_val_ppl", float("inf")))
        self.sleep_phase_counter = int(ts.get("sleep_phase_counter", 1))
        self.preemption_saves = int(ts.get("preemption_saves", 0))
        self.corrupt_recoveries = int(ts.get("corrupt_recoveries", 0))
        if loaded["primary_corrupted"]:
            self.corrupt_recoveries += 1
        self.best_frozen_exports = int(ts.get("best_frozen_exports", 0))

        acc = ts.get("epoch_accum")
        if isinstance(acc, dict):
            self._current_epoch_accum = {
                "epoch": int(acc.get("epoch", 1)),
                "correct": int(acc.get("correct", 0)),
                "valid": int(acc.get("valid", 0)),
                "ce_sum": float(acc.get("ce_sum", 0.0)),
                "steps": int(acc.get("steps", 0)),
                "tokens": int(acc.get("tokens", 0)),
            }

        s_state = payload.get("streamer_state")
        if streamer is not None and s_state is not None:
            streamer.load_streamer_state(s_state)

        self.vault.append_telemetry(
            {
                "event": "checkpoint_resumed",
                "source": loaded["source"],
                "primary_corrupted": bool(loaded["primary_corrupted"]),
                "global_step": int(self.global_step),
                "epoch": int(streamer.current_epoch if streamer is not None else self._current_epoch_accum["epoch"]),
                "batch_cursor": int(streamer.batch_cursor if streamer is not None else 0),
            }
        )

        return {
            "resumed": True,
            "source": loaded["source"],
            "primary_corrupted": bool(loaded["primary_corrupted"]),
            "primary_error": loaded["primary_error"],
            "resumed_global_step": int(self.global_step),
            "resumed_epoch": int(streamer.current_epoch if streamer is not None else 1),
            "resumed_batch_cursor": int(streamer.batch_cursor if streamer is not None else 0),
        }

    @torch.no_grad()
    def _maybe_export_best_frozen(
        self, val_metrics: dict[str, float], epoch: int
    ) -> dict | None:
        """
        Exports a strictly compacted `Frozen W0 Only` checkpoint from a detached
        clone whenever validation accuracy reaches a new high (or ties with lower PPL),
        without freezing or disturbing the live training model.
        """
        val_acc = float(val_metrics["acc_pct"])
        val_ppl = float(val_metrics["ppl"])
        improved = (val_acc > self.best_val_acc + 1e-6) or (
            abs(val_acc - self.best_val_acc) <= 1e-6 and val_ppl < self.best_val_ppl
        )
        if not improved:
            return None

        self.best_val_acc = val_acc
        self.best_val_ppl = val_ppl
        if not self.keep_best_frozen:
            return None

        w0_state, freeze_stats = self.model.export_frozen_w0_state_nondestructive(map_location="cpu")
        w0_state["val_metrics"] = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "val_acc_pct": val_acc,
            "val_ppl": val_ppl,
            "frozen_live_mb": float(freeze_stats["after_live_mb"]),
        }
        save_info = self.vault.save_frozen_w0(w0_state)
        self.best_frozen_exports += 1
        self.vault.append_telemetry(
            {
                "event": "best_frozen_w0_exported",
                "epoch": int(epoch),
                "global_step": int(self.global_step),
                "val_acc_pct": val_acc,
                "val_ppl": val_ppl,
                "size_kb": float(save_info["size_kb"]),
            }
        )
        return save_info

    @torch.no_grad()
    def fit(
        self,
        train_dataset: MMapTokenDataset,
        val_batches: Sequence[tuple[torch.Tensor, torch.Tensor]] | MMapTokenDataset | None = None,
        epochs: int = 6,
        batch_size: int = 32,
        pad_id: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 1234,
        prefetch_batches: int = 2,
        auto_resume: bool = True,
        signal_handler: PreemptionSignalHandler | None = None,
        stop_after_steps: int | None = None,
        save_on_stop_after_steps: bool = False,
        on_step_callback: Callable[
            [int, int, int, "ElasticCloudTrainer", PreemptionSignalHandler], None
        ]
        | None = None,
    ) -> dict:
        """
        Runs resilient multi-epoch training over an `MMapTokenDataset` with
        automatic vault resume, periodic checkpoints, preemption emergency save,
        and JSONL telemetry logging.
        """
        streamer = ResumableBatchStreamer(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            base_seed=seed,
            prefetch_batches=prefetch_batches,
            start_epoch=1,
        )

        resume_info = {
            "resumed": False,
            "source": None,
            "primary_corrupted": False,
            "resumed_global_step": 0,
            "resumed_epoch": 1,
            "resumed_batch_cursor": 0,
        }
        if auto_resume:
            resume_info = self.restore_from_vault(streamer=streamer)

        start_epoch = int(streamer.current_epoch)
        last_save_wall = time.monotonic()
        t_fit_start = time.perf_counter()
        total_fit_tokens = 0
        status = "completed"
        last_train_acc = 0.0
        last_train_ce = 0.0
        checkpoint_saves = 0
        emergency_saves = 0
        epochs_history: list[dict] = []

        handler = signal_handler if signal_handler is not None else PreemptionSignalHandler(True)
        with handler:
            for ep in range(start_epoch, int(epochs) + 1):
                if (
                    self._current_epoch_accum["epoch"] == ep
                    and streamer.batch_cursor > 0
                ):
                    ep_correct = int(self._current_epoch_accum["correct"])
                    ep_valid = int(self._current_epoch_accum["valid"])
                    ep_ce_sum = float(self._current_epoch_accum["ce_sum"])
                    ep_steps = int(self._current_epoch_accum["steps"])
                    ep_tokens = int(self._current_epoch_accum["tokens"])
                    start_cursor = int(streamer.batch_cursor)
                else:
                    ep_correct = 0
                    ep_valid = 0
                    ep_ce_sum = 0.0
                    ep_steps = 0
                    ep_tokens = 0
                    start_cursor = 0
                    self._current_epoch_accum = {
                        "epoch": int(ep),
                        "correct": 0,
                        "valid": 0,
                        "ce_sum": 0.0,
                        "steps": 0,
                        "tokens": 0,
                    }

                interrupted_mid_epoch = False
                for batch_idx, x_b, y_b in streamer.iter_epoch(epoch=ep, start_cursor=start_cursor):
                    x_dev = x_b.to(device=self.model.model_device)
                    y_dev = y_b.to(device=self.model.model_device)
                    st = self.model.train_minibatch(x_dev, y_dev, pad_id=pad_id)

                    n_val_tok = int(st["valid_tokens"])
                    ep_correct += int(st["correct"])
                    ep_valid += n_val_tok
                    ep_ce_sum += float(st["ce_loss"]) * max(n_val_tok, 1)
                    ep_steps += 1
                    ep_tokens += n_val_tok
                    total_fit_tokens += n_val_tok
                    self.global_step += 1

                    self._current_epoch_accum = {
                        "epoch": int(ep),
                        "correct": ep_correct,
                        "valid": ep_valid,
                        "ce_sum": ep_ce_sum,
                        "steps": ep_steps,
                        "tokens": ep_tokens,
                    }

                    if on_step_callback is not None:
                        on_step_callback(self.global_step, ep, batch_idx, self, handler)

                    # 1. Periodic step or wall-clock atomic checkpoint
                    now_wall = time.monotonic()
                    need_step_save = (
                        self.checkpoint_every_steps > 0
                        and self.global_step % self.checkpoint_every_steps == 0
                    )
                    need_time_save = (
                        self.checkpoint_every_seconds > 0.0
                        and (now_wall - last_save_wall) >= self.checkpoint_every_seconds
                    )
                    if (need_step_save or need_time_save) and not handler.preemption_requested:
                        save_meta = self.vault.save_checkpoint(
                            self.build_checkpoint_payload(streamer)
                        )
                        last_save_wall = now_wall
                        checkpoint_saves += 1
                        self.vault.append_telemetry(
                            {
                                "event": "periodic_checkpoint",
                                "epoch": int(ep),
                                "batch_idx": int(batch_idx),
                                "next_batch_cursor": int(streamer.batch_cursor),
                                "global_step": int(self.global_step),
                                "save_ms": float(save_meta["save_ms"]),
                            }
                        )

                    # 2. Preemption signal watchdog (SIGTERM / SIGINT)
                    if handler.preemption_requested:
                        self.preemption_saves += 1
                        emergency_saves += 1
                        save_meta = self.vault.save_checkpoint(
                            self.build_checkpoint_payload(streamer)
                        )
                        self.vault.append_telemetry(
                            {
                                "event": "emergency_preemption_save",
                                "signal": handler.signal_name,
                                "epoch": int(ep),
                                "next_batch_cursor": int(streamer.batch_cursor),
                                "global_step": int(self.global_step),
                                "payload_kb": float(save_meta["payload_kb"]),
                            }
                        )
                        status = "preempted"
                        interrupted_mid_epoch = True
                        break

                    # 3. Deterministic hard-stop trigger for simulated fault injection
                    if stop_after_steps is not None and self.global_step >= int(stop_after_steps):
                        if save_on_stop_after_steps:
                            save_meta = self.vault.save_checkpoint(
                                self.build_checkpoint_payload(streamer)
                            )
                            checkpoint_saves += 1
                            self.vault.append_telemetry(
                                {
                                    "event": "stop_after_steps_save",
                                    "epoch": int(ep),
                                    "next_batch_cursor": int(streamer.batch_cursor),
                                    "global_step": int(self.global_step),
                                    "save_ms": float(save_meta["save_ms"]),
                                }
                            )
                        status = "stopped_mid_run"
                        interrupted_mid_epoch = True
                        break

                if interrupted_mid_epoch:
                    break

                # Clean full-epoch completion
                last_train_ce = ep_ce_sum / max(ep_valid, 1)
                last_train_acc = (float(ep_correct) / max(ep_valid, 1)) * 100.0
                growth_info = self.model.end_epoch(epoch=ep, mean_ce_loss=last_train_ce)

                has_val = val_batches is not None and len(val_batches) > 0
                val_metrics = (
                    evaluate_validation_batches(
                        self.model, val_batches, pad_id=pad_id, batch_size=batch_size
                    )
                    if has_val
                    else {"acc_pct": 0.0, "ce": 0.0, "ppl": 1.0, "valid_tokens": 0}
                )
                if has_val:
                    self._maybe_export_best_frozen(val_metrics, epoch=ep)

                if (
                    self.sleep_every_epochs > 0
                    and ep >= self.sleep_min_epoch
                    and (ep % self.sleep_every_epochs == 0)
                    and ep < int(epochs)
                ):
                    sleep_info = self.model.sleep_consolidate(phase_id=self.sleep_phase_counter)
                    self.sleep_phase_counter += 1
                    self.vault.append_telemetry(
                        {
                            "event": "sleep_consolidation",
                            "epoch": int(ep),
                            "phase_id": int(self.sleep_phase_counter - 1),
                            "merged_total": int(sleep_info.get("merged_total", 0)),
                            "pruned_total": int(sleep_info.get("pruned_total", 0)),
                            "locked_total": int(sleep_info.get("locked_total", 0)),
                        }
                    )

                streamer.advance_epoch_boundary(ep)
                self._current_epoch_accum = {
                    "epoch": int(ep + 1),
                    "correct": 0,
                    "valid": 0,
                    "ce_sum": 0.0,
                    "steps": 0,
                    "tokens": 0,
                }

                save_meta = self.vault.save_checkpoint(self.build_checkpoint_payload(streamer))
                last_save_wall = time.monotonic()
                checkpoint_saves += 1
                ep_record = {
                    "event": "epoch_end",
                    "epoch": int(ep),
                    "global_step": int(self.global_step),
                    "train_acc_pct": float(last_train_acc),
                    "train_ce": float(last_train_ce),
                    "val_acc_pct": float(val_metrics["acc_pct"]),
                    "val_ppl": float(val_metrics["ppl"]),
                    "active_dims": list(self.model.get_allocated_dimensions()),
                    "grew": bool(growth_info.get("grew", False)),
                    "save_ms": float(save_meta["save_ms"]),
                }
                self.vault.append_telemetry(ep_record)
                epochs_history.append(ep_record)

                if self.verbose:
                    print(
                        f"  [Ep {ep:02d}/{epochs:02d}] Step {self.global_step:04d} | "
                        f"Train {last_train_acc:.2f}% (CE {last_train_ce:.3f}) | "
                        f"Val {val_metrics['acc_pct']:.2f}% (PPL {val_metrics['ppl']:.2f})"
                    )

        elapsed_s = max(time.perf_counter() - t_fit_start, 1e-6)
        final_val = (
            evaluate_validation_batches(self.model, val_batches, pad_id=pad_id)
            if val_batches
            else {"acc_pct": 0.0, "ce": 0.0, "ppl": 1.0, "valid_tokens": 0}
        )
        return {
            "status": status,
            "resumed": bool(resume_info["resumed"]),
            "resume_source": resume_info["source"],
            "primary_corrupted": bool(resume_info["primary_corrupted"]),
            "resumed_global_step": int(resume_info["resumed_global_step"]),
            "resumed_epoch": int(resume_info["resumed_epoch"]),
            "resumed_batch_cursor": int(resume_info["resumed_batch_cursor"]),
            "final_global_step": int(self.global_step),
            "elapsed_s": elapsed_s,
            "tok_s": float(total_fit_tokens) / elapsed_s,
            "checkpoint_saves": int(checkpoint_saves),
            "emergency_saves": int(emergency_saves),
            "best_val_acc_pct": float(self.best_val_acc),
            "best_val_ppl": float(self.best_val_ppl),
            "final_val_acc_pct": float(final_val["acc_pct"]),
            "final_val_ppl": float(final_val["ppl"]),
            "last_train_acc_pct": float(last_train_acc),
            "last_train_ce": float(last_train_ce),
            "best_frozen_w0_exists": self.vault.best_frozen_path.exists(),
            "telemetry_records": len(self.vault.read_telemetry()),
            "epochs_history": epochs_history,
        }

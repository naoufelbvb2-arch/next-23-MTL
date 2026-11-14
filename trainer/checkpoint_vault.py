"""
trainer/checkpoint_vault.py
Crash-Proof Atomic Checkpoint Vault & Cloud/Colab Preemption Watchdog
for SS-ENN v3.0 Phase 3.3.

Provides:
  1. `AtomicCheckpointVault`:
     - Writes checkpoints into `.tmp` -> `flush()` -> `os.fsync()` -> rotates the
       prior validated `checkpoint_latest.pt` into `checkpoint_backup.pt` (also via
       atomic `.tmp` + `fsync` + `os.replace`) -> atomically swaps `.tmp` to
       `checkpoint_latest.pt`.
     - Embeds SHA-256 payload digest inside a lightweight envelope so truncation,
       zero-fill, or single-bit corruption in `checkpoint_latest.pt` triggers an
       automatic deterministic self-healing fallback to `checkpoint_backup.pt`.
     - Exports minimal deployment `best_frozen_w0.pt` atomically and appends
       structured JSONL telemetry to `training_telemetry.jsonl`.
  2. `PreemptionSignalHandler`:
     - Registers `SIGINT` and `SIGTERM` OS signal handlers (plus clean simulated
       preemption helper for automated tests) to trigger emergency atomic save.

Strictly Zero-Autograd.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Any

import hashlib
import torch


VAULT_ENVELOPE_MAGIC = "SSENN_ATOMIC_VAULT_V1"


def _fsync_parent_dir(path: Path) -> None:
    """Flushes parent directory metadata when supported by the OS (POSIX)."""
    if os.name == "nt":
        return
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _serialize_payload_with_checksum(payload: dict) -> dict:
    bio = io.BytesIO()
    torch.save(payload, bio)
    raw = bio.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "vault_magic": VAULT_ENVELOPE_MAGIC,
        "payload_sha256": digest,
        "payload_size_bytes": len(raw),
        "payload_bytes": raw,
    }


def _atomic_torch_save_envelope(target_path: Path, envelope: dict) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        torch.save(envelope, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)
    _fsync_parent_dir(target_path)


def _atomic_copy_file(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(f"{dst_path.name}.tmp.{os.getpid()}")
    with open(src_path, "rb") as src_f, open(tmp_path, "wb") as dst_f:
        shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)
        dst_f.flush()
        os.fsync(dst_f.fileno())
    os.replace(tmp_path, dst_path)
    _fsync_parent_dir(dst_path)


def _load_and_verify_envelope(path: Path) -> tuple[bool, dict | None, str | None]:
    """
    Loads a vaulted checkpoint envelope and checks both deserialization integrity
    and exact SHA-256 payload digest match.
    """
    if not path.exists():
        return False, None, f"missing:{path.name}"
    if path.stat().st_size == 0:
        return False, None, f"empty_file:{path.name}"
    try:
        env = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(env, dict) or env.get("vault_magic") != VAULT_ENVELOPE_MAGIC:
            return False, None, f"invalid_envelope_magic:{path.name}"
        raw = env.get("payload_bytes")
        expected_sha = env.get("payload_sha256")
        if not isinstance(raw, (bytes, bytearray)) or not isinstance(expected_sha, str):
            return False, None, f"malformed_envelope_fields:{path.name}"
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            return False, None, f"sha256_mismatch:{path.name}"
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return False, None, f"invalid_payload_type:{path.name}"
        return True, payload, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}:{exc}"


class AtomicCheckpointVault:
    """
    Crash-safe dual-generation checkpoint vault managing:
      - `checkpoint_latest.pt` (newest atomically committed checkpoint)
      - `checkpoint_backup.pt` (prior verified generation for self-healing fallback)
      - `best_frozen_w0.pt` (strictly compacted deployment W0 artifact)
      - `training_telemetry.jsonl` (append-only structured event log)
    """

    def __init__(self, vault_dir: str | Path) -> None:
        self.vault_dir = Path(vault_dir)
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.vault_dir / "checkpoint_latest.pt"
        self.backup_path = self.vault_dir / "checkpoint_backup.pt"
        self.best_frozen_path = self.vault_dir / "best_frozen_w0.pt"
        self.telemetry_path = self.vault_dir / "training_telemetry.jsonl"
        self._latest_verified: bool = False
        self._latest_verified_size: int = -1

    def save_checkpoint(self, payload: dict) -> dict:
        """
        Atomically writes `payload` into `checkpoint_latest.pt` while first
        preserving the existing valid `checkpoint_latest.pt` into
        `checkpoint_backup.pt` so a mid-swap crash or corruption never wipes all state.
        """
        t0 = time.perf_counter()
        envelope = _serialize_payload_with_checksum(payload)

        # Stage new checkpoint into temp file & sync BEFORE modifying latest/backup
        stage_tmp = self.vault_dir / f".stage_latest.tmp.{os.getpid()}"
        with open(stage_tmp, "wb") as f:
            torch.save(envelope, f)
            f.flush()
            os.fsync(f.fileno())

        # Promote prior healthy checkpoint_latest.pt -> checkpoint_backup.pt
        if self.latest_path.exists():
            try:
                cur_size = int(self.latest_path.stat().st_size)
            except OSError:
                cur_size = -1
            if self._latest_verified and cur_size == self._latest_verified_size:
                os.replace(self.latest_path, self.backup_path)
                _fsync_parent_dir(self.backup_path)
            else:
                ok_prev, _, _ = _load_and_verify_envelope(self.latest_path)
                if ok_prev:
                    _atomic_copy_file(self.latest_path, self.backup_path)

        os.replace(stage_tmp, self.latest_path)
        _fsync_parent_dir(self.latest_path)
        self._latest_verified = True
        self._latest_verified_size = int(self.latest_path.stat().st_size)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "latest_path": str(self.latest_path),
            "backup_exists": self.backup_path.exists(),
            "payload_sha256": envelope["payload_sha256"],
            "payload_kb": envelope["payload_size_bytes"] / 1024.0,
            "save_ms": elapsed_ms,
        }

    def load_latest_valid_checkpoint(self) -> dict:
        """
        Loads `checkpoint_latest.pt` if checksum and payload validate cleanly;
        otherwise automatically falls back to `checkpoint_backup.pt`.
        Returns a structured dict:
          `{"found": bool, "source": "latest"|"backup"|None,
            "primary_corrupted": bool, "primary_error": str|None, "payload": dict|None}`.
        """
        if not self.latest_path.exists() and not self.backup_path.exists():
            return {
                "found": False,
                "source": None,
                "primary_corrupted": False,
                "primary_error": None,
                "payload": None,
            }

        ok_latest, payload_latest, err_latest = _load_and_verify_envelope(self.latest_path)
        if ok_latest and payload_latest is not None:
            return {
                "found": True,
                "source": "latest",
                "primary_corrupted": False,
                "primary_error": None,
                "payload": payload_latest,
            }

        primary_corrupted = self.latest_path.exists()
        ok_backup, payload_backup, err_backup = _load_and_verify_envelope(self.backup_path)
        if ok_backup and payload_backup is not None:
            self.append_telemetry(
                {
                    "event": "corrupt_checkpoint_fallback",
                    "primary_error": err_latest,
                    "recovered_from": "checkpoint_backup.pt",
                }
            )
            return {
                "found": True,
                "source": "backup",
                "primary_corrupted": primary_corrupted,
                "primary_error": err_latest,
                "payload": payload_backup,
            }

        raise IOError(
            f"All checkpoints in vault {self.vault_dir} failed integrity check "
            f"(latest={err_latest}, backup={err_backup})"
        )

    def save_frozen_w0(self, frozen_state: dict) -> dict:
        """Atomically persists compacted `Frozen W0 Only` artifact (`best_frozen_w0.pt`)."""
        envelope = _serialize_payload_with_checksum(frozen_state)
        _atomic_torch_save_envelope(self.best_frozen_path, envelope)
        return {
            "path": str(self.best_frozen_path),
            "size_kb": float(self.best_frozen_path.stat().st_size) / 1024.0,
            "sha256": envelope["payload_sha256"],
        }

    def load_frozen_w0(self) -> dict:
        """Loads and verifies `best_frozen_w0.pt`."""
        return self.load_frozen_w0_file(self.best_frozen_path)

    @staticmethod
    def load_frozen_w0_file(path: str | Path) -> dict:
        """
        Loads a `Frozen W0 Only` checkpoint from `path`, transparently verifying
        the SHA-256 vault envelope when present or loading a standalone raw
        `export_frozen_w0_state()` dictionary directly.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Frozen W0 checkpoint not found: {p}")
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and obj.get("vault_magic") == VAULT_ENVELOPE_MAGIC:
            ok, payload, err = _load_and_verify_envelope(p)
            if not ok or payload is None:
                raise IOError(f"Failed to verify vaulted frozen W0 {p}: {err}")
            return payload
        if isinstance(obj, dict) and "w0_tensors" in obj and "config" in obj:
            return obj
        raise ValueError(f"Unrecognized Frozen W0 checkpoint schema at {p}")

    def append_telemetry(self, record: dict[str, Any]) -> None:
        """Appends a timestamped JSON record cleanly to `training_telemetry.jsonl`."""
        entry = {"timestamp": time.time(), **record}
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(self.telemetry_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def read_telemetry(self) -> list[dict]:
        """Reads all telemetry JSONL records recorded in the vault."""
        if not self.telemetry_path.exists():
            return []
        rows: list[dict] = []
        with open(self.telemetry_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
        return rows

    def corrupt_file_for_test(
        self,
        which: str = "latest",
        mode: str = "truncate",
    ) -> None:
        """
        Deterministic fault-injection helper (for benchmark & test verification):
        `mode='truncate'` truncates file to 32 bytes; `mode='bitflip'` flips bytes
        near the middle of the file.
        """
        self._latest_verified = False
        self._latest_verified_size = -1
        target = self.latest_path if which == "latest" else self.backup_path
        if not target.exists():
            raise FileNotFoundError(f"Target checkpoint does not exist: {target}")
        data = bytearray(target.read_bytes())
        if mode == "truncate":
            truncated = bytes(data[: min(32, max(8, len(data) // 10))])
            _atomic_write_bytes(target, truncated)
        elif mode == "bitflip":
            if len(data) < 64:
                data[0] ^= 0xFF
            else:
                mid = len(data) // 2
                for offset in range(mid, min(len(data), mid + 16)):
                    data[offset] ^= 0xAA
            _atomic_write_bytes(target, bytes(data))
        else:
            raise ValueError(f"Unsupported fault injection mode: {mode!r}")


def _atomic_write_bytes(target_path: Path, data: bytes) -> None:
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


class PreemptionSignalHandler:
    """
    Cloud/Colab graceful interruption watchdog catching `SIGINT` & `SIGTERM`.

    Signal callbacks only flip an atomic flag (`preemption_requested = True`) so
    the in-flight `train_minibatch()` completes deterministically before the
    trainer saves an emergency checkpoint and exits cleanly.
    """

    def __init__(self, register_os_signals: bool = True) -> None:
        self.register_os_signals = bool(register_os_signals)
        self.preemption_requested: bool = False
        self.signal_received: int | None = None
        self.signal_name: str | None = None
        self._prev_handlers: dict[int, Any] = {}

    def _handle_signal(self, signum: int, frame: Any) -> None:
        self.preemption_requested = True
        self.signal_received = int(signum)
        try:
            self.signal_name = signal.Signals(signum).name
        except Exception:
            self.signal_name = f"SIG_{signum}"

    def trigger_simulated_sigterm(self) -> None:
        """Deterministically triggers preemption state across any platform."""
        sigterm = getattr(signal, "SIGTERM", 15)
        self.preemption_requested = True
        self.signal_received = int(sigterm)
        self.signal_name = "SIGTERM"

    def raise_os_sigterm(self) -> bool:
        """
        Attempts to dispatch a real OS `SIGTERM` via `signal.raise_signal` when
        registered on the main thread; falls back to simulated flag if unavailable.
        """
        sigterm = getattr(signal, "SIGTERM", None)
        if (
            sigterm is not None
            and hasattr(signal, "raise_signal")
            and threading.current_thread() is threading.main_thread()
            and sigterm in self._prev_handlers
        ):
            signal.raise_signal(sigterm)
            return True
        self.trigger_simulated_sigterm()
        return False

    def __enter__(self) -> "PreemptionSignalHandler":
        if (
            self.register_os_signals
            and threading.current_thread() is threading.main_thread()
        ):
            for sig_attr in ("SIGINT", "SIGTERM"):
                sig_val = getattr(signal, sig_attr, None)
                if sig_val is not None:
                    try:
                        prev = signal.getsignal(sig_val)
                        signal.signal(sig_val, self._handle_signal)
                        self._prev_handlers[sig_val] = prev
                    except (ValueError, OSError):
                        pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for sig_val, prev in self._prev_handlers.items():
            try:
                signal.signal(sig_val, prev)
            except (ValueError, OSError):
                pass
        self._prev_handlers.clear()

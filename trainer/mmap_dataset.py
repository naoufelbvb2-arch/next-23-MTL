"""
trainer/mmap_dataset.py
Memory-Mapped Binary Shard Dataset & Deterministic Mid-Epoch Resumable Streamer
for SS-ENN v3.0 Phase 3.3.

Design guarantees:
  1. Binary shards (`uint16` for V <= 65,535, `uint32` otherwise) written
     atomically alongside `.meta.json` containing SHA-256 checksum & geometry.
  2. Read-only `numpy.memmap(mode='r')` access with O(1) resident RAM overhead
     regardless of corpus scale (MB to multi-GB corpora).
  3. Bounded background prefetch queue (`threading.Thread` + `queue.Queue`)
     keeping memory bounded while hiding disk I/O latency.
  4. Deterministic per-epoch RNG seeding (`base_seed + epoch * 1000003`) and exact
     `get_streamer_state()` / `load_streamer_state()` cursor serialization so
     mid-epoch kills resume on the exact next batch with zero skipped or
     duplicated batches (`0.00e+00` bitwise training parity).

Strictly Zero-Autograd.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch


SHARD_FORMAT_VERSION = "ssenn_mmap_shard_1.0"


def _atomic_write_bytes(target_path: Path, data: bytes) -> None:
    """Writes bytes atomically via temp file + flush + os.fsync + os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def _compute_file_sha256(file_path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Computes SHA-256 of a file using O(1) streaming buffer memory."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class BinaryShardBuilder:
    """
    Constructs packed fixed-width token shards (`.bin` + `.meta.json`) ready for
    O(1) RAM memory-mapped streaming in `MMapTokenDataset`.
    """

    @staticmethod
    def select_token_dtype(vocab_size: int) -> tuple[str, np.dtype]:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
        if vocab_size <= 65536:
            return "uint16", np.dtype(np.uint16)
        return "uint32", np.dtype(np.uint32)

    @classmethod
    def build_from_sequences(
        cls,
        sequences: Sequence[Sequence[int]] | np.ndarray | torch.Tensor,
        vocab_size: int,
        seq_len: int,
        output_dir: str | Path,
        shard_name: str = "train",
        pad_id: int = 0,
    ) -> dict:
        """
        Packs pre-tokenized rows into a binary memory-mappable shard where each
        row has length `row_stride = seq_len + 1` so inputs `x = row[:-1]` and
        next-token shifted targets `y = row[1:]`.
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dtype_name, np_dtype = cls.select_token_dtype(vocab_size)
        row_stride = int(seq_len) + 1

        if isinstance(sequences, (np.ndarray, torch.Tensor)):
            arr = (
                sequences.detach().cpu().numpy()
                if isinstance(sequences, torch.Tensor)
                else sequences
            )
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array of sequences, got shape {arr.shape}")
            num_rows, src_width = arr.shape
            if num_rows == 0:
                raise ValueError("Cannot build a binary shard from 0 sequences.")
            min_tok = int(arr.min())
            max_tok = int(arr.max())
            max_allowed = int(vocab_size) - 1
            if min_tok < 0 or max_tok > max_allowed:
                raise ValueError(
                    f"Token ids must lie in [0, {max_allowed}], got [{min_tok}, {max_tok}]"
                )
            packed = np.full((num_rows, row_stride), int(pad_id), dtype=np_dtype)
            copy_w = min(src_width, row_stride)
            packed[:, :copy_w] = arr[:, :copy_w].astype(np_dtype, copy=False)
            total_valid_tokens = int(np.count_nonzero(packed[:, 1:] != int(pad_id)))
        else:
            raw_rows = sequences
            num_rows = len(raw_rows)
            if num_rows == 0:
                raise ValueError("Cannot build a binary shard from 0 sequences.")

            packed = np.full((num_rows, row_stride), int(pad_id), dtype=np_dtype)
            max_allowed = int(vocab_size) - 1
            total_valid_tokens = 0
            for idx, seq in enumerate(raw_rows):
                if len(seq) < row_stride:
                    row_vals = list(seq) + [int(pad_id)] * (row_stride - len(seq))
                else:
                    row_vals = list(seq[:row_stride])
                for v in row_vals:
                    if v < 0 or v > max_allowed:
                        raise ValueError(
                            f"Token id {v} at row {idx} is out of range [0, {max_allowed}]"
                        )
                packed[idx, :] = np.asarray(row_vals, dtype=np_dtype)
                total_valid_tokens += sum(1 for tok in row_vals[1:] if tok != int(pad_id))

        bin_path = out_dir / f"{shard_name}.bin"
        meta_path = out_dir / f"{shard_name}.meta.json"
        bin_tmp = bin_path.with_name(f"{bin_path.name}.tmp.{os.getpid()}")

        with open(bin_tmp, "wb") as f:
            f.write(packed.tobytes(order="C"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(bin_tmp, bin_path)

        sha256 = _compute_file_sha256(bin_path)
        bin_size_bytes = int(bin_path.stat().st_size)

        meta = {
            "format_version": SHARD_FORMAT_VERSION,
            "shard_name": shard_name,
            "dtype": dtype_name,
            "itemsize_bytes": int(np_dtype.itemsize),
            "vocab_size": int(vocab_size),
            "seq_len": int(seq_len),
            "row_stride": row_stride,
            "num_sequences": int(num_rows),
            "total_stored_tokens": int(num_rows * row_stride),
            "total_valid_tokens": int(total_valid_tokens),
            "pad_id": int(pad_id),
            "bin_file": bin_path.name,
            "bin_size_bytes": bin_size_bytes,
            "sha256": sha256,
        }
        _atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode("utf-8"))
        return meta

    @classmethod
    def build_from_stories(
        cls,
        stories: Sequence[str],
        tokenizer,
        seq_len: int,
        output_dir: str | Path,
        shard_name: str = "train",
    ) -> dict:
        """
        Tokenizes narrative stories identically to `build_minibatch_tensors`
        (`add_bos=True`, `add_eos=True`, padded/trimmed to `seq_len + 1`) and
        writes a binary shard + `.meta.json`.
        """
        sequences: list[list[int]] = []
        row_stride = int(seq_len) + 1
        pad_id = int(tokenizer.pad_id)
        for story in stories:
            ids = tokenizer.encode(story, add_bos=True, add_eos=True)
            if len(ids) < row_stride:
                padded = ids + [pad_id] * (row_stride - len(ids))
            else:
                padded = ids[:row_stride]
            sequences.append(padded)
        return cls.build_from_sequences(
            sequences=sequences,
            vocab_size=int(tokenizer.vocab_size),
            seq_len=int(seq_len),
            output_dir=output_dir,
            shard_name=shard_name,
            pad_id=pad_id,
        )

    @classmethod
    def build_from_flat_tokens(
        cls,
        flat_tokens: Sequence[int] | np.ndarray | torch.Tensor,
        seq_len: int,
        vocab_size: int,
        output_dir: str | Path,
        shard_name: str = "train",
        pad_id: int = 0,
    ) -> dict:
        """
        Chunks a contiguous 1D token stream directly into non-overlapping
        `row_stride = seq_len + 1` binary shard windows without extra copies.
        """
        arr = (
            flat_tokens.detach().cpu().numpy()
            if isinstance(flat_tokens, torch.Tensor)
            else np.asarray(flat_tokens, dtype=np.int64)
        )
        row_stride = int(seq_len) + 1
        n_rows = int(arr.size // row_stride)
        if n_rows == 0:
            raise ValueError(
                f"Need at least {row_stride} tokens to emit 1 window, got {arr.size}"
            )
        reshaped = arr[: n_rows * row_stride].reshape(n_rows, row_stride)
        return cls.build_from_sequences(
            sequences=reshaped,
            vocab_size=vocab_size,
            seq_len=seq_len,
            output_dir=output_dir,
            shard_name=shard_name,
            pad_id=pad_id,
        )


class MMapTokenDataset:
    """
    Read-only `numpy.memmap` token shard reader maintaining O(1) resident RAM
    footprint regardless of binary shard size.
    """

    def __init__(
        self,
        shard_path: str | Path,
        shard_name: str = "train",
        verify_checksum: bool = False,
    ) -> None:
        p = Path(shard_path)
        if p.is_dir():
            meta_path = p / f"{shard_name}.meta.json"
        elif p.name.endswith(".meta.json"):
            meta_path = p
        elif p.name.endswith(".bin"):
            meta_path = p.with_suffix(".meta.json")
        else:
            meta_path = p / f"{shard_name}.meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Shard metadata file not found: {meta_path}")

        self.meta_path: Path = meta_path.resolve()
        self.meta: dict = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.meta.get("format_version") != SHARD_FORMAT_VERSION:
            raise ValueError(f"Unsupported shard format: {self.meta.get('format_version')}")

        self.bin_path: Path = self.meta_path.parent / self.meta["bin_file"]
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary shard file not found: {self.bin_path}")

        dtype_str = self.meta["dtype"]
        if dtype_str not in ("uint16", "uint32"):
            raise ValueError(f"Unsupported binary shard dtype: {dtype_str!r}")
        self.np_dtype = np.dtype(dtype_str)

        self.vocab_size: int = int(self.meta["vocab_size"])
        self.seq_len: int = int(self.meta["seq_len"])
        self.row_stride: int = int(self.meta["row_stride"])
        self.num_sequences: int = int(self.meta["num_sequences"])
        self.total_stored_tokens: int = int(self.meta["total_stored_tokens"])
        self.total_valid_tokens: int = int(self.meta.get("total_valid_tokens", self.total_stored_tokens))
        self.pad_id: int = int(self.meta.get("pad_id", 0))

        expected_bytes = self.num_sequences * self.row_stride * int(self.np_dtype.itemsize)
        actual_bytes = int(self.bin_path.stat().st_size)
        if actual_bytes != expected_bytes:
            raise IOError(
                f"Corrupted shard size on disk ({actual_bytes} B != expected {expected_bytes} B): {self.bin_path}"
            )

        if verify_checksum and not self.verify_integrity():
            raise IOError(f"SHA-256 integrity mismatch for binary shard: {self.bin_path}")

        self._mmap: np.memmap | None = np.memmap(
            self.bin_path,
            dtype=self.np_dtype,
            mode="r",
            shape=(self.num_sequences, self.row_stride),
        )

    def verify_integrity(self) -> bool:
        """Streams the `.bin` shard in 1 MB chunks to verify its SHA-256 digest."""
        return _compute_file_sha256(self.bin_path) == self.meta.get("sha256")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mmap is None:
            raise RuntimeError("MMapTokenDataset has been closed.")
        row = np.asarray(self._mmap[idx], dtype=np.int64).copy()
        t = torch.from_numpy(row)
        return t[:-1], t[1:]

    def get_batch(self, indices: Sequence[int] | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Reads a batch of row indices directly from memmap into `[B, seq_len]` tensors."""
        if self._mmap is None:
            raise RuntimeError("MMapTokenDataset has been closed.")
        idx_arr = np.asarray(indices, dtype=np.int64)
        rows = np.asarray(self._mmap[idx_arr], dtype=np.int64).copy()
        t = torch.from_numpy(rows)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def disk_size_mb(self) -> float:
        return float(self.bin_path.stat().st_size) / (1024.0 * 1024.0)

    def metadata_ram_bytes(self) -> int:
        """Returns resident Python object & metadata byte footprint (strictly O(1))."""
        total = sys.getsizeof(self) + sys.getsizeof(self.meta) + sys.getsizeof(self._mmap)
        for k, v in self.meta.items():
            total += sys.getsizeof(k) + sys.getsizeof(v)
        return int(total)

    def close(self) -> None:
        """Releases the OS memory mapping cleanly (critical for Windows file handles)."""
        if self._mmap is not None:
            inner = getattr(self._mmap, "_mmap", None)
            self._mmap = None
            if inner is not None:
                try:
                    inner.close()
                except Exception:
                    pass

    def __enter__(self) -> "MMapTokenDataset":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class ResumableBatchStreamer:
    """
    Deterministic epoch batch streamer backed by `MMapTokenDataset` and a
    bounded background prefetcher thread.

    Cursor invariant:
      - `self.current_epoch`: active epoch (1-indexed).
      - `self.batch_cursor`: 0-based index of the NEXT batch to be trained.
      - Before yielding `(batch_idx, x_b, y_b)`, `batch_cursor` advances to
        `batch_idx + 1` once pulled; thus checkpointing right after
        `train_minibatch(x_b, y_b)` records the exact next batch index without
        repeating or skipping a single sample upon restore in a fresh process.
    """

    def __init__(
        self,
        dataset: MMapTokenDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        base_seed: int = 1234,
        prefetch_batches: int = 2,
        start_epoch: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if prefetch_batches < 0:
            raise ValueError(f"prefetch_batches must be >= 0, got {prefetch_batches}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.base_seed = int(base_seed)
        self.prefetch_batches = int(prefetch_batches)

        self.current_epoch: int = int(start_epoch)
        self.batch_cursor: int = 0
        self.epoch_rng_seed: int = self._compute_epoch_seed(self.current_epoch)
        self._cached_epoch: int | None = None
        self._cached_batches: list[np.ndarray] = []

    def _compute_epoch_seed(self, epoch: int) -> int:
        return int(self.base_seed) + int(epoch) * 1000003

    def _build_epoch_batches(self, epoch: int) -> list[np.ndarray]:
        if self._cached_epoch == epoch and self._cached_batches:
            return self._cached_batches

        n_samples = len(self.dataset)
        if self.shuffle:
            g = torch.Generator(device="cpu")
            g.manual_seed(self._compute_epoch_seed(epoch))
            perm = torch.randperm(n_samples, generator=g).numpy()
        else:
            perm = np.arange(n_samples, dtype=np.int64)

        batches: list[np.ndarray] = []
        for start in range(0, n_samples, self.batch_size):
            end = start + self.batch_size
            if end > n_samples and self.drop_last:
                break
            batches.append(perm[start:end])

        self._cached_epoch = int(epoch)
        self._cached_batches = batches
        return batches

    def num_batches_per_epoch(self, epoch: int = 1) -> int:
        return len(self._build_epoch_batches(epoch))

    def get_streamer_state(self) -> dict:
        """Returns deterministic cursor metadata safe to embed inside checkpoints."""
        return {
            "epoch": int(self.current_epoch),
            "batch_cursor": int(self.batch_cursor),
            "epoch_rng_seed": int(self.epoch_rng_seed),
            "batch_size": int(self.batch_size),
            "shuffle": bool(self.shuffle),
            "drop_last": bool(self.drop_last),
            "base_seed": int(self.base_seed),
            "num_sequences": len(self.dataset),
        }

    def load_streamer_state(self, state: dict) -> None:
        """Restores exact epoch permutation and mid-epoch batch cursor."""
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError(
                f"Streamer batch_size mismatch: ckpt={state['batch_size']} vs cur={self.batch_size}"
            )
        if int(state["num_sequences"]) != len(self.dataset):
            raise ValueError(
                f"Dataset length mismatch: ckpt={state['num_sequences']} vs cur={len(self.dataset)}"
            )
        self.base_seed = int(state["base_seed"])
        self.shuffle = bool(state["shuffle"])
        self.drop_last = bool(state["drop_last"])
        self.current_epoch = int(state["epoch"])
        self.batch_cursor = int(state["batch_cursor"])
        self.epoch_rng_seed = int(state["epoch_rng_seed"])
        self._cached_epoch = None
        self._build_epoch_batches(self.current_epoch)

    def advance_epoch_boundary(self, completed_epoch: int) -> None:
        """Marks `completed_epoch` finished and resets cursor cleanly to next epoch batch 0."""
        self.current_epoch = int(completed_epoch) + 1
        self.batch_cursor = 0
        self.epoch_rng_seed = self._compute_epoch_seed(self.current_epoch)
        self._cached_epoch = None

    def iter_epoch(
        self,
        epoch: int | None = None,
        start_cursor: int | None = None,
    ) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        """
        Yields `(batch_idx, input_ids, target_ids)` for `epoch` starting at
        `start_cursor`, backed by a bounded prefetch worker when `prefetch_batches > 0`.
        """
        ep = self.current_epoch if epoch is None else int(epoch)
        self.current_epoch = ep
        self.epoch_rng_seed = self._compute_epoch_seed(ep)
        epoch_batches = self._build_epoch_batches(ep)

        cursor = self.batch_cursor if start_cursor is None else int(start_cursor)
        self.batch_cursor = cursor
        total_batches = len(epoch_batches)
        if cursor >= total_batches:
            return

        if self.prefetch_batches == 0:
            for b_idx in range(cursor, total_batches):
                x_b, y_b = self.dataset.get_batch(epoch_batches[b_idx])
                self.batch_cursor = b_idx + 1
                yield b_idx, x_b, y_b
            return

        q_buf: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        stop_ev = threading.Event()
        sentinel = object()

        def _worker() -> None:
            try:
                for b_idx in range(cursor, total_batches):
                    if stop_ev.is_set():
                        break
                    x_b, y_b = self.dataset.get_batch(epoch_batches[b_idx])
                    while not stop_ev.is_set():
                        try:
                            q_buf.put((b_idx, x_b, y_b), timeout=0.05)
                            break
                        except queue.Full:
                            continue
                while not stop_ev.is_set():
                    try:
                        q_buf.put(sentinel, timeout=0.05)
                        break
                    except queue.Full:
                        continue
            except BaseException as exc:  # pragma: no cover - surfaced to consumer
                try:
                    q_buf.put(exc, timeout=0.2)
                except queue.Full:
                    pass

        worker = threading.Thread(target=_worker, name=f"ssenn-prefetch-ep{ep}", daemon=True)
        worker.start()
        try:
            while True:
                item = q_buf.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                b_idx, x_b, y_b = item
                self.batch_cursor = b_idx + 1
                yield b_idx, x_b, y_b
        finally:
            stop_ev.set()
            worker.join(timeout=2.0)

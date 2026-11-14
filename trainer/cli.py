"""
trainer/cli.py
Unified Command-Line Interface (`python -m trainer.cli <subcommand>`) for
SS-ENN v3.0 Phase 3.4.

Subcommands:
  1. `estimate`     : Zero-allocation analytical VRAM/RAM footprint & GPU tier estimator.
  2. `prepare-data` : Text/corpus tokenization + SHA-256 verified binary `.bin` shard builder.
  3. `train`        : Fault-tolerant elastic training / auto-resume with Atomic Vault & SIGTERM guard.
  4. `generate`     : O(1) stateful cache autoregressive rollout from compacted `best_frozen_w0.pt`.

Strictly Zero-Autograd.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.english_dataset import build_english_tinystories_corpus  # noqa: E402
from engine.tokenizer import (  # noqa: E402
    EnglishSubwordTokenizer,
    MultilingualByteBPETokenizer,
    load_tokenizer,
)
from models.config_v3 import SSENNConfig  # noqa: E402
from models.model_v3_production import SSENNProductionModel  # noqa: E402
from trainer.checkpoint_vault import AtomicCheckpointVault, PreemptionSignalHandler  # noqa: E402
from trainer.colab_orchestrator import ElasticCloudTrainer  # noqa: E402
from trainer.mmap_dataset import BinaryShardBuilder, MMapTokenDataset  # noqa: E402


def _recommend_hardware(total_train_with_kahan_gb: float, max_train_params: int) -> str:
    if total_train_with_kahan_gb <= 1.0 and max_train_params <= 50_000_000:
        return "CPU / Colab T4 16GB"
    if total_train_with_kahan_gb <= 12.5:
        return "Colab T4 16GB (or L4 24GB)"
    if total_train_with_kahan_gb <= 20.5:
        return "Colab L4 24GB (or A100 40GB)"
    if total_train_with_kahan_gb <= 36.0:
        return "Colab A100 40GB (or A100 80GB)"
    if total_train_with_kahan_gb <= 72.0:
        return "Colab A100 80GB (or Cloud TPU v5p)"
    return "Multi-GPU 80GB+ / Cloud TPU v5p Pod"


def _add_model_config_args(p: argparse.ArgumentParser, *, default_vocab: int | None) -> None:
    p.add_argument(
        "--vocab-size",
        type=int,
        default=default_vocab,
        help="Vocabulary size V (for train, inferred from shard metadata if omitted).",
    )
    p.add_argument("--num-layers", type=int, default=12, help="Number of SS-ENN spring layers.")
    p.add_argument("--d-init", type=int, default=512, help="Initial active subspace width.")
    p.add_argument("--d-max", type=int, default=1024, help="Maximum neurogenesis width ceiling.")
    p.add_argument("--delta-d", type=int, default=128, help="Neurogenesis expansion step (must align with block-size).")
    p.add_argument("--block-size", type=int, default=16, help="Contiguous capsule block size B_block.")
    p.add_argument("--top-k-blocks", type=int, default=8, help="Active sparse blocks per forward pass.")
    p.add_argument("--max-seq-len", type=int, default=1024, help="Maximum causal context length T_max.")
    p.add_argument("--attn-top-k", type=int, default=12, help="Causal attention Top-K slots.")
    p.add_argument("--salient-anchor-slots", type=int, default=2, help="Salient entity anchor slots.")
    p.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
        help="Weight tensor precision.",
    )
    p.add_argument(
        "--kahan-mode",
        choices=("full", "anchor_k", "off"),
        default="full",
        help="FP32 Kahan residual mode: full (w,w0,k), anchor_k (w0,k; -33.3%% RAM), or off.",
    )
    p.add_argument("--device", type=str, default="cpu", help="Target device (cpu, cuda, cuda:0, xla).")
    p.add_argument("--seed", type=int, default=42, help="Deterministic base PRNG seed.")


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B ({n:,})"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M ({n:,})"
    return f"{n:,}"


def cmd_estimate(args: argparse.Namespace) -> int:
    torch.set_grad_enabled(False)
    cfg = SSENNConfig(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        d_init=args.d_init,
        d_max=args.d_max,
        delta_d=args.delta_d,
        block_size=args.block_size,
        top_k_blocks=args.top_k_blocks,
        max_seq_len=args.max_seq_len,
        attn_top_k=args.attn_top_k,
        salient_anchor_slots=args.salient_anchor_slots,
        dtype=args.dtype,
        kahan_mode=args.kahan_mode,
        device=args.device,
        seed=args.seed,
    )
    fp = cfg.estimate_memory_footprint()
    hw = _recommend_hardware(float(fp["total_train_with_kahan_gb"]), int(fp["max_train_params"]))
    result = {**fp, "recommended_hardware": hw}

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("=" * 88)
    print("SS-ENN v3.0 Analytical Footprint Estimator (Zero-Allocation)")
    print("=" * 88)
    print(
        f"  Geometry       : L={cfg.num_layers} | d_init={cfg.d_init} -> d_max={cfg.d_max} "
        f"(delta_d={cfg.delta_d}, B={cfg.block_size}, K_blk={cfg.top_k_blocks})"
    )
    print(
        f"  Context & Dtype: V={cfg.vocab_size:,} | T_max={cfg.max_seq_len} | "
        f"dtype={cfg.dtype} | kahan_mode={fp['kahan_mode']} | target={cfg.device}"
    )
    print("-" * 88)
    print(f"  Init Active Params      : {_format_params(int(fp['init_active_params']))}")
    print(f"  Init Active Memory      : {fp['init_active_mb']:>10.2f} MB")
    print(f"  Max Train Params        : {_format_params(int(fp['max_train_params']))}")
    print(
        f"  Max Train Weights       : {fp['max_train_mb']:>10.2f} MB  "
        f"({fp['max_train_gb']:.3f} GB)"
    )
    print(
        f"  Factorized DFA Memory   : {fp['dfa_mb']:>10.2f} MB  "
        f"(vs Naive [L,D,V] {fp['naive_dfa_mb']:.2f} MB -> {fp['dfa_memory_savings_pct']:.2f}% saved)"
    )
    print(
        f"  Kahan FP32 Residuals    : {fp['kahan_max_overhead_mb']:>10.2f} MB  "
        f"({fp['kahan_max_overhead_mb'] / 1024.0:.3f} GB | {fp['kahan_savings_vs_full_pct']:.1f}% vs full)"
    )
    print(
        f"  Total Train + Kahan     : {fp['total_train_with_kahan_mb']:>10.2f} MB  "
        f"({fp['total_train_with_kahan_gb']:.3f} GB)"
    )
    print(
        f"  Frozen W0 Deployment    : {fp['frozen_w0_mb']:>10.2f} MB  "
        f"({fp['frozen_w0_gb']:.3f} GB | {_format_params(int(fp['frozen_w0_params']))})"
    )
    print("-" * 88)
    print(f"  Recommended Accelerator : {hw}")
    print("=" * 88)
    return 0


def _split_raw_text_documents(raw_text: str) -> list[str]:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text.strip()) if s.strip()]
    if len(sentences) >= 2:
        return sentences
    words = raw_text.strip().split()
    if len(words) < 4:
        return [raw_text.strip()] if raw_text.strip() else []
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def cmd_prepare_data(args: argparse.Namespace) -> int:
    torch.set_grad_enabled(False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.builtin_tinystories:
        corpus = build_english_tinystories_corpus(
            target_min_tokens=args.target_min_tokens,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        train_stories = list(corpus["train_stories"])
        val_stories = list(corpus["val_stories"])
    else:
        if not args.input_file:
            raise ValueError("Provide either --input-file PATH or --builtin-tinystories.")
        in_path = Path(args.input_file)
        if not in_path.exists():
            raise FileNotFoundError(f"Input text file not found: {in_path}")
        docs = _split_raw_text_documents(in_path.read_text(encoding="utf-8"))
        if len(docs) < 2:
            raise ValueError(f"Input corpus {in_path} has too little text to form train/val shards.")
        rng = random.Random(args.seed)
        rng.shuffle(docs)
        n_train = max(1, min(len(docs) - 1, int(round(len(docs) * float(args.train_ratio)))))
        train_stories = docs[:n_train]
        val_stories = docs[n_train:]

    if getattr(args, "tokenizer_type", "english_subword") == "multilingual_bpe":
        tokenizer = MultilingualByteBPETokenizer(max_vocab_size=args.vocab_size)
    else:
        tokenizer = EnglishSubwordTokenizer(max_vocab_size=args.vocab_size)
    tokenizer.fit(train_stories + val_stories, pad_to_max_vocab=not args.no_pad_vocab)
    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(tok_path)

    train_meta = BinaryShardBuilder.build_from_stories(
        stories=train_stories,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        output_dir=out_dir,
        shard_name="train_shard",
    )
    val_meta = BinaryShardBuilder.build_from_stories(
        stories=val_stories,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        output_dir=out_dir,
        shard_name="val_shard",
    )

    print("=" * 88)
    print("SS-ENN v3.0 Binary Shard Preparation Complete")
    print("=" * 88)
    print(f"  Output Dir     : {out_dir.resolve()}")
    print(f"  Tokenizer      : {tok_path.name} (vocab_size={tokenizer.vocab_size})")
    print(
        f"  Train Shard    : {train_meta['bin_file']} | seqs={train_meta['num_sequences']:,} | "
        f"seq_len={train_meta['seq_len']} | dtype={train_meta['dtype']} | "
        f"tokens={train_meta['total_stored_tokens']:,} | sha256={train_meta['sha256'][:16]}..."
    )
    print(
        f"  Val Shard      : {val_meta['bin_file']} | seqs={val_meta['num_sequences']:,} | "
        f"seq_len={val_meta['seq_len']} | dtype={val_meta['dtype']} | "
        f"tokens={val_meta['total_stored_tokens']:,} | sha256={val_meta['sha256'][:16]}..."
    )
    print("=" * 88)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    torch.set_grad_enabled(False)
    data_dir = Path(args.data_dir)
    vault_dir = Path(args.vault_dir)
    vault_dir.mkdir(parents=True, exist_ok=True)

    tok_path = Path(args.tokenizer) if args.tokenizer else (data_dir / "tokenizer.json")
    if tok_path.exists():
        tokenizer = load_tokenizer(tok_path)
        shutil.copy2(tok_path, vault_dir / "tokenizer.json")
    else:
        tokenizer = None

    with (
        MMapTokenDataset(data_dir, shard_name=args.train_shard, verify_checksum=True) as train_ds,
        MMapTokenDataset(data_dir, shard_name=args.val_shard, verify_checksum=True) as val_ds,
    ):
        vocab_size = int(args.vocab_size) if args.vocab_size is not None else int(train_ds.vocab_size)
        if vocab_size != train_ds.vocab_size:
            raise ValueError(
                f"Configured vocab_size ({vocab_size}) != shard vocab_size ({train_ds.vocab_size})"
            )
        if val_ds.seq_len != train_ds.seq_len:
            raise ValueError(
                f"Train shard seq_len ({train_ds.seq_len}) != val shard seq_len ({val_ds.seq_len})"
            )
        effective_seq_len = int(train_ds.seq_len)
        if int(args.max_seq_len) < effective_seq_len:
            print(
                f"[CLI Notice] Raising --max-seq-len from {args.max_seq_len} to "
                f"{effective_seq_len} to match shard seq_len.",
                flush=True,
            )
        max_seq_len = int(max(args.max_seq_len, effective_seq_len))
        pad_id = int(train_ds.pad_id)
        bos_id = tokenizer.bos_id if tokenizer is not None else 1
        eos_id = tokenizer.eos_id if tokenizer is not None else 2

        cfg = SSENNConfig(
            vocab_size=vocab_size,
            num_layers=args.num_layers,
            d_init=args.d_init,
            d_max=args.d_max,
            delta_d=args.delta_d,
            block_size=args.block_size,
            top_k_blocks=args.top_k_blocks,
            max_seq_len=max_seq_len,
            attn_top_k=args.attn_top_k,
            salient_anchor_slots=args.salient_anchor_slots,
            dtype=args.dtype,
            kahan_mode=args.kahan_mode,
            device=args.device,
            seed=args.seed,
        )

        if args.sleep_min_epoch < 0:
            raise ValueError(f"--sleep-min-epoch must be >= 0, got {args.sleep_min_epoch}")
        trainer = ElasticCloudTrainer(
            config=cfg,
            vault_dir=vault_dir,
            checkpoint_every_steps=args.ckpt_every_steps,
            checkpoint_every_seconds=args.ckpt_every_seconds,
            keep_best_frozen=True,
            sleep_every_epochs=args.sleep_every_epochs,
            sleep_min_epoch=args.sleep_min_epoch,
            verbose=not args.quiet,
        )

        has_prior_ckpt = (
            AtomicCheckpointVault(vault_dir).latest_path.exists()
            or AtomicCheckpointVault(vault_dir).backup_path.exists()
        )
        if not (args.auto_resume and has_prior_ckpt):
            trainer.calibrate_salience_from_dataset(
                train_ds, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id
            )

        sig_handler = PreemptionSignalHandler(register_os_signals=True)
        res = trainer.fit(
            train_dataset=train_ds,
            val_batches=val_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            pad_id=pad_id,
            shuffle=True,
            seed=args.seed + 1000,
            prefetch_batches=args.prefetch_batches,
            auto_resume=args.auto_resume,
            signal_handler=sig_handler,
            stop_after_steps=args.stop_after_steps,
        )

    print("=" * 88)
    print("SS-ENN v3.0 Elastic Cloud Training Summary")
    print("=" * 88)
    print(
        f"  Status          : {res['status']} | Final Step: {res['final_global_step']} | "
        f"Throughput: {res['tok_s']:,.1f} tok/s"
    )
    print(
        f"  Sequence & Sleep: shard_seq_len={effective_seq_len} | max_seq_len={max_seq_len} | "
        f"sleep_every={args.sleep_every_epochs} (min_ep={args.sleep_min_epoch})"
    )
    print(
        f"  Resume State    : resumed={res['resumed']} (source={res['resume_source']}, "
        f"corrupted={res['primary_corrupted']})"
    )
    print(
        f"  Validation      : Final Val Acc={res['final_val_acc_pct']:.2f}% (PPL {res['final_val_ppl']:.2f}) | "
        f"Best Val Acc={res['best_val_acc_pct']:.2f}% (PPL {res['best_val_ppl']:.2f})"
    )
    print(
        f"  Vault Artifacts : best_frozen_w0={res['best_frozen_w0_exists']} | "
        f"telemetry_records={res['telemetry_records']} | dir={vault_dir.resolve()}"
    )
    print("=" * 88)
    return 0


def _resolve_frozen_model_and_tokenizer(args: argparse.Namespace) -> tuple[SSENNProductionModel, EnglishSubwordTokenizer]:
    ckpt_path: Path | None = Path(args.checkpoint) if args.checkpoint else None
    vault_dir: Path | None = Path(args.vault_dir) if args.vault_dir else None

    if ckpt_path is None:
        if vault_dir is None:
            raise ValueError("Provide either --checkpoint PATH or --vault-dir PATH.")
        candidate = vault_dir / "best_frozen_w0.pt"
        if candidate.exists():
            ckpt_path = candidate

    if ckpt_path is not None:
        raw_state = AtomicCheckpointVault.load_frozen_w0_file(ckpt_path)
        model = SSENNProductionModel.from_frozen_w0_state(raw_state)
        search_dirs = [ckpt_path.parent]
    else:
        vault = AtomicCheckpointVault(vault_dir)
        loaded = vault.load_latest_valid_checkpoint()
        if not loaded["found"] or loaded["payload"] is None:
            raise FileNotFoundError(f"No valid checkpoint found in vault {vault_dir}")
        model = SSENNProductionModel.from_checkpoint_state(loaded["payload"]["model_state"])
        model.production_freeze(prune_dormant=True)
        search_dirs = [vault_dir]

    if args.device and str(args.device) != str(model.model_device):
        model.to_device_and_dtype(device=args.device, dtype=model.config.dtype)

    tok_path: Path | None = Path(args.tokenizer) if args.tokenizer else None
    if tok_path is None:
        for d in search_dirs:
            cand = d / "tokenizer.json"
            if cand.exists():
                tok_path = cand
                break
    if tok_path is None or not tok_path.exists():
        raise FileNotFoundError(
            "Tokenizer file not found. Pass --tokenizer PATH pointing to tokenizer.json."
        )
    tokenizer = load_tokenizer(tok_path)
    return model, tokenizer


def cmd_generate(args: argparse.Namespace) -> int:
    torch.set_grad_enabled(False)
    model, tokenizer = _resolve_frozen_model_and_tokenizer(args)

    prompt_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    t0 = time.perf_counter()
    out_ids = model.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_id=tokenizer.eos_id,
        enable_refractory_gate=not args.disable_refractory,
        use_fast_cache=not args.slow_cache,
    )
    elapsed_s = max(time.perf_counter() - t0, 1e-6)
    gen_tokens = max(0, len(out_ids) - len(prompt_ids))
    tok_s = float(gen_tokens) / elapsed_s
    full_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    print("=" * 88)
    print("SS-ENN v3.0 Frozen W0 O(1) Stateful Cache Generation")
    print("=" * 88)
    print(
        f"  Cache Mode      : {'O(1) Stateful Cache' if not args.slow_cache else 'O(T^2) Recomputation'} | "
        f"Device: {model.model_device} | Frozen={model.frozen}"
    )
    print(f"  Prompt          : {args.prompt}")
    print(f"  Generated Text  : {full_text}")
    print(
        f"  Speed           : {gen_tokens} new tokens in {elapsed_s * 1000.0:.2f} ms "
        f"({tok_s:,.1f} tok/s)"
    )
    print("=" * 88)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trainer.cli",
        description="SS-ENN v3.0 Unified Production CLI (Zero-Autograd)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. estimate
    p_est = sub.add_parser(
        "estimate",
        help="Analytical zero-allocation VRAM/RAM footprint & hardware estimator.",
    )
    _add_model_config_args(p_est, default_vocab=32000)
    p_est.add_argument("--json", action="store_true", help="Emit JSON output instead of table.")
    p_est.set_defaults(func=cmd_estimate)

    # 2. prepare-data
    p_prep = sub.add_parser(
        "prepare-data",
        help="Tokenize text or built-in corpus into binary uint16/uint32 .bin shards.",
    )
    p_prep.add_argument("--input-file", type=str, default=None, help="Path to UTF-8 raw text file.")
    p_prep.add_argument(
        "--builtin-tinystories",
        action="store_true",
        help="Use built-in connected English TinyStories corpus.",
    )
    p_prep.add_argument("--out-dir", type=str, required=True, help="Output shard & tokenizer directory.")
    p_prep.add_argument(
        "--tokenizer-type",
        type=str,
        default="english_subword",
        choices=("english_subword", "multilingual_bpe"),
        help="Tokenizer engine: legacy english_subword or lossless MultilingualByteBPETokenizer.",
    )
    p_prep.add_argument("--vocab-size", type=int, default=2048, help="Tokenizer max vocabulary size.")
    p_prep.add_argument("--seq-len", type=int, default=36, help="Causal training sequence length T.")
    p_prep.add_argument("--train-ratio", type=float, default=0.90, help="Train split fraction.")
    p_prep.add_argument(
        "--target-min-tokens",
        type=int,
        default=20000,
        help="Minimum tokens when generating built-in TinyStories corpus.",
    )
    p_prep.add_argument("--no-pad-vocab", action="store_true", help="Do not pad vocab up to --vocab-size.")
    p_prep.add_argument("--seed", type=int, default=42, help="Split permutation seed.")
    p_prep.set_defaults(func=cmd_prepare_data)

    # 3. train
    p_train = sub.add_parser(
        "train",
        help="Launch or resume fault-tolerant SS-ENN training with Atomic Vault.",
    )
    _add_model_config_args(p_train, default_vocab=None)
    p_train.set_defaults(num_layers=6, d_init=64, d_max=128, delta_d=16, top_k_blocks=2, max_seq_len=64, attn_top_k=8, dtype="float32")
    p_train.add_argument("--data-dir", type=str, required=True, help="Directory holding .bin shards & tokenizer.json.")
    p_train.add_argument("--vault-dir", type=str, required=True, help="Directory for AtomicCheckpointVault outputs.")
    p_train.add_argument("--tokenizer", type=str, default=None, help="Optional explicit path to tokenizer.json.")
    p_train.add_argument("--train-shard", type=str, default="train_shard", help="Training shard basename.")
    p_train.add_argument("--val-shard", type=str, default="val_shard", help="Validation shard basename.")
    p_train.add_argument("--epochs", type=int, default=14, help="Total target training epochs.")
    p_train.add_argument("--batch-size", type=int, default=32, help="Minibatch size B.")
    p_train.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Optional check against shard seq_len (auto-inferred from shard meta.json when omitted).",
    )
    p_train.add_argument("--prefetch-batches", type=int, default=2, help="Background prefetch queue depth.")
    p_train.add_argument("--ckpt-every-steps", type=int, default=20, help="Periodic checkpoint interval in steps.")
    p_train.add_argument("--ckpt-every-seconds", type=float, default=0.0, help="Periodic checkpoint interval in seconds.")
    p_train.add_argument(
        "--sleep-every-epochs",
        type=int,
        default=0,
        help="Run dual-stage offline sleep consolidation every N epochs (default: 0 disables).",
    )
    p_train.add_argument(
        "--sleep-min-epoch",
        type=int,
        default=0,
        help="Minimum epoch required before periodic sleep consolidation triggers (default: 0).",
    )
    p_train.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume from checkpoint_latest.pt / checkpoint_backup.pt (default: True).",
    )
    p_train.add_argument("--stop-after-steps", type=int, default=None, help="Optional hard step budget cap.")
    p_train.add_argument("--quiet", action="store_true", help="Suppress per-epoch progress lines.")
    p_train.set_defaults(func=cmd_train)

    # 4. generate
    p_gen = sub.add_parser(
        "generate",
        help="Run O(1) stateful cache text generation from Frozen W0 checkpoint.",
    )
    p_gen.add_argument("--checkpoint", type=str, default=None, help="Path to best_frozen_w0.pt or frozen state.")
    p_gen.add_argument("--vault-dir", type=str, default=None, help="Vault directory holding best_frozen_w0.pt.")
    p_gen.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.json.")
    p_gen.add_argument("--prompt", type=str, required=True, help="Input text prompt.")
    p_gen.add_argument("--max-new-tokens", type=int, default=30, help="Maximum new tokens to decode.")
    p_gen.add_argument("--device", type=str, default="cpu", help="Inference device (cpu, cuda).")
    p_gen.add_argument("--disable-refractory", action="store_true", help="Disable inference refractory gate.")
    p_gen.add_argument("--slow-cache", action="store_true", help="Use O(T^2) recomputation instead of O(1) cache.")
    p_gen.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and args.seq_len is not None:
        # Validate requested seq_len matches prepared binary shard when explicitly supplied
        with MMapTokenDataset(args.data_dir, shard_name=args.train_shard, verify_checksum=False) as ds_check:
            if int(args.seq_len) != ds_check.seq_len:
                raise ValueError(
                    f"--seq-len ({args.seq_len}) does not match shard seq_len ({ds_check.seq_len})"
                )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

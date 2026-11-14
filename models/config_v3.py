"""
models/config_v3.py
Phase 3.2: Explicit Manual Configuration & Pre-Allocation Memory Footprint Estimator
for the Clean Unified SS-ENN v3.0 Production Model (`SSENNProductionModel`).

Design Principles:
  - Pure manual numeric configuration: no hidden black-box scale presets inside
    the model or config constructor. Users explicitly specify architectural numbers
    (`vocab_size`, `num_layers`, `d_init`, `d_max`, `delta_d`, `block_size`,
    `top_k_blocks`, `device`, `dtype`, physics rates).
  - Strict physical validation before allocation (`validate()`): catches invalid
    block divisibility, out-of-range sparse Top-K counts, or unsupported hardware/dtype.
  - Analytical zero-allocation memory estimator (`estimate_memory_footprint()`):
    computes exact parameter counts & MB/GB across d_init, d_max, FactorizedHighwayDFA,
    Kahan FP32 compensator overhead (on BF16/FP16), and post-freeze `Frozen W0 Only`.

Zero autograd / zero backward().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import torch


_VALID_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_VALID_DEVICE_PREFIXES: tuple[str, ...] = ("cpu", "cuda", "xla", "mps")


@dataclass
class SSENNConfig:
    """Explicit manual configuration for `SSENNProductionModel`."""

    # Core structural geometry (must be explicitly configured)
    vocab_size: int = 2048
    num_layers: int = 6
    d_init: int = 64
    d_max: int = 128
    delta_d: int = 16
    block_size: int = 16
    top_k_blocks: int = 2
    layer_top_k_blocks: list[int] | None = None
    max_seq_len: int = 512
    attn_top_k: int = 8
    salient_anchor_slots: int = 2
    head_scale: float = 1.0
    negative_slope: float = 0.10

    # Hardware target & numerical precision
    device: str = "cpu"
    dtype: str = "float32"
    use_kahan_compensator: bool | None = None
    kahan_mode: str = "full"  # "full" (w, w0, k), "anchor_k" (w0, k only; -33.3% RAM), or "off"

    # Spring physics & plasticity rates
    lr_w: float = 0.08
    lr_w0: float = 0.015
    gamma_restore: float = 0.10
    beta_k: float = 0.025

    lr_head_w: float = 0.28
    lr_head_0: float = 0.040
    gamma_head: float = 0.020
    beta_k_head: float = 0.025

    lr_attn_w: float = 0.05
    lr_attn_0: float = 0.010
    gamma_attn: float = 0.10
    beta_k_attn: float = 0.020

    # Highway normalization & factorized feedback alignment
    alpha_res: float = 0.50
    target_rms: float = 1.20
    lr_dfa: float = 0.015
    dfa_anchor_blend: float = 0.65

    # Sparse block competition, stiffness spectrum & neurogenesis
    block_balance_strength: float = 0.70
    stiff_mult_low: float = 0.45
    stiff_mult_high: float = 1.60
    stiff_ceil_low: float = 0.28
    stiff_ceil_high: float = 0.85

    neurogenesis_err_thresh: float = 0.04
    neurogenesis_sat_thresh: float = 0.70
    neurogenesis_stiff_thresh: float = 0.0005
    neurogenesis_cooldown: int = 3

    # Dual-stage sleep consolidation & synaptic recycling
    sleep_merge_threshold: float = 0.85
    sleep_prune_ratio: float = 0.25
    sleep_lock_min_k: float = 0.88
    sleep_lock_max_k: float = 0.96

    # Two-tier inference refractory gate
    refractory_adj_penalty: float = 4.5
    refractory_loop_penalty: float = 6.0

    # Reproducibility seed
    seed: int = 42

    def __post_init__(self) -> None:
        if self.layer_top_k_blocks is not None:
            self.layer_top_k_blocks = [int(x) for x in self.layer_top_k_blocks]
        if self.kahan_mode not in ("full", "anchor_k", "off"):
            raise ValueError(
                f"Unsupported kahan_mode '{self.kahan_mode}'. Choose from ('full', 'anchor_k', 'off')"
            )
        if self.use_kahan_compensator is None:
            self.use_kahan_compensator = (self.dtype in ("bfloat16", "float16")) and (
                self.kahan_mode != "off"
            )
        elif not self.use_kahan_compensator:
            if self.kahan_mode == "anchor_k":
                raise ValueError("kahan_mode='anchor_k' conflicts with use_kahan_compensator=False")
            self.kahan_mode = "off"
        elif self.kahan_mode == "off":
            self.use_kahan_compensator = False
        self.validate()

    def validate(self) -> None:
        """Validates all geometric, structural, precision, and physical constraints."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.d_init <= 0 or self.d_max <= 0:
            raise ValueError(f"d_init ({self.d_init}) and d_max ({self.d_max}) must be > 0")
        if self.delta_d < 0:
            raise ValueError(f"delta_d must be >= 0, got {self.delta_d}")
        if self.d_init > self.d_max:
            raise ValueError(f"d_init ({self.d_init}) must be <= d_max ({self.d_max})")

        if self.d_init % self.block_size != 0:
            raise ValueError(
                f"d_init ({self.d_init}) must be divisible by block_size ({self.block_size})"
            )
        if self.d_max % self.block_size != 0:
            raise ValueError(
                f"d_max ({self.d_max}) must be divisible by block_size ({self.block_size})"
            )
        if self.delta_d % self.block_size != 0:
            raise ValueError(
                f"delta_d ({self.delta_d}) must be divisible by block_size ({self.block_size})"
            )

        n_blocks_init = self.d_init // self.block_size
        if self.top_k_blocks <= 0 or self.top_k_blocks > n_blocks_init:
            raise ValueError(
                f"top_k_blocks ({self.top_k_blocks}) must be in [1, d_init // block_size = {n_blocks_init}]"
            )

        if self.layer_top_k_blocks is not None:
            if len(self.layer_top_k_blocks) != self.num_layers:
                raise ValueError(
                    f"layer_top_k_blocks length ({len(self.layer_top_k_blocks)}) "
                    f"must equal num_layers ({self.num_layers})"
                )
            for idx, k_blk in enumerate(self.layer_top_k_blocks):
                if k_blk <= 0 or k_blk > n_blocks_init:
                    raise ValueError(
                        f"layer_top_k_blocks[{idx}]={k_blk} must be in [1, {n_blocks_init}]"
                    )

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")
        if self.attn_top_k <= 1:
            raise ValueError(f"attn_top_k must be >= 2, got {self.attn_top_k}")
        if self.salient_anchor_slots < 1 or self.salient_anchor_slots >= self.attn_top_k:
            raise ValueError(
                f"salient_anchor_slots ({self.salient_anchor_slots}) must be in "
                f"[1, attn_top_k - 1 = {self.attn_top_k - 1}]"
            )

        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}'. Choose from {tuple(_VALID_DTYPES.keys())}"
            )
        if self.kahan_mode not in ("full", "anchor_k", "off"):
            raise ValueError(
                f"Unsupported kahan_mode '{self.kahan_mode}'. Choose from ('full', 'anchor_k', 'off')"
            )
        dev_base = str(self.device).split(":", 1)[0].lower()
        if dev_base not in _VALID_DEVICE_PREFIXES:
            raise ValueError(
                f"Unsupported device '{self.device}'. Prefix must be one of {_VALID_DEVICE_PREFIXES}"
            )

        if self.lr_w <= 0 or self.lr_w0 < 0 or self.gamma_restore < 0 or self.beta_k < 0:
            raise ValueError("Spring plasticity rates must satisfy lr_w > 0 and lr_w0, gamma, beta_k >= 0")
        if not (0.0 <= self.dfa_anchor_blend <= 1.0):
            raise ValueError(f"dfa_anchor_blend must be in [0, 1], got {self.dfa_anchor_blend}")
        if not (0.0 <= self.sleep_merge_threshold <= 1.0):
            raise ValueError(
                f"sleep_merge_threshold must be in [0, 1], got {self.sleep_merge_threshold}"
            )
        if not (0.0 < self.sleep_lock_min_k <= self.sleep_lock_max_k < 1.0):
            raise ValueError("sleep_lock_min_k and sleep_lock_max_k must satisfy 0 < min <= max < 1")

    @property
    def torch_dtype(self) -> torch.dtype:
        return _VALID_DTYPES[self.dtype]

    @property
    def dtype_bytes(self) -> int:
        return 4 if self.dtype == "float32" else 2

    @property
    def uses_kahan_compensator(self) -> bool:
        return bool(
            self.use_kahan_compensator
            and self.kahan_mode != "off"
            and self.dtype in ("bfloat16", "float16")
        )

    @property
    def effective_kahan_mode(self) -> str:
        if not self.uses_kahan_compensator:
            return "off"
        return self.kahan_mode

    def resolve_torch_device(self) -> torch.device:
        return torch.device(self.device)

    def get_layer_top_k_blocks(self) -> list[int]:
        if self.layer_top_k_blocks is not None:
            return list(self.layer_top_k_blocks)
        return [int(self.top_k_blocks) for _ in range(self.num_layers)]

    def get_hierarchical_stiff_schedule(self) -> list[tuple[float, float]]:
        schedule: list[tuple[float, float]] = []
        denom = float(max(self.num_layers - 1, 1))
        for l_idx in range(self.num_layers):
            rho = float(l_idx) / denom
            mult = self.stiff_mult_low + (self.stiff_mult_high - self.stiff_mult_low) * rho
            ceil = self.stiff_ceil_low + (self.stiff_ceil_high - self.stiff_ceil_low) * rho
            schedule.append((mult, ceil))
        return schedule

    def estimate_memory_footprint(self) -> dict[str, float | int | str | bool]:
        """
        Pure analytical pre-allocation footprint estimator (no tensors created).
        Breaks down parameter counts and RAM/VRAM usage for:
          - Initial active slice (`d_init`)
          - Full training allocation (`d_max`, Zero-[D,V] buffers)
          - FactorizedHighwayDFA vs naive [L, D, V] DFA savings
          - FP32 Kahan residual compensator overhead (when dtype is BF16/FP16)
          - Final deployment state (`Frozen W0 Only`, with w/k/DFA/Kahan stripped)
        """
        V = int(self.vocab_size)
        L = int(self.num_layers)
        Di = int(self.d_init)
        Dm = int(self.d_max)
        T = int(self.max_seq_len)
        b = int(self.dtype_bytes)
        mb = 1024.0 * 1024.0
        gb = mb * 1024.0

        # Non-spring structural frontend buffers
        def _frontend_non_spring(d: int) -> int:
            return (V * d) + (T * d) + (3 * d * d)

        # Spring triplets (w, w0, k): each element holds 3 tensors in `dtype`
        def _spring_base_elements(d: int) -> int:
            attn_qkv = 3 * d * d
            hidden = L * d * d
            head = V * d
            return attn_qkv + hidden + head

        # Compact highway->attention router (`attn_dfa`: [d, d]) + FactorizedHighwayDFA (L * [d, d])
        def _factorized_dfa_elements(d: int) -> int:
            return (L + 1) * d * d

        init_frontend = _frontend_non_spring(Di)
        init_spring_w = _spring_base_elements(Di)
        init_dfa = _factorized_dfa_elements(Di)
        init_active_params = init_frontend + (3 * init_spring_w) + init_dfa

        max_frontend = _frontend_non_spring(Dm)
        max_spring_w = _spring_base_elements(Dm)
        max_dfa = _factorized_dfa_elements(Dm)
        max_train_params = max_frontend + (3 * max_spring_w) + max_dfa

        # Kahan compensator holds 3x FP32 ("full"), 2x FP32 ("anchor_k", -33.3%), or 0 ("off")
        eff_kahan = self.effective_kahan_mode
        kahan_mult = 3 if eff_kahan == "full" else (2 if eff_kahan == "anchor_k" else 0)
        kahan_init_bytes = kahan_mult * init_spring_w * 4
        kahan_max_bytes = kahan_mult * max_spring_w * 4
        kahan_full_max_bytes = (3 * max_spring_w * 4) if self.dtype in ("bfloat16", "float16") else 0
        kahan_savings_vs_full_pct = (
            (1.0 - (float(kahan_max_bytes) / float(kahan_full_max_bytes))) * 100.0
            if kahan_full_max_bytes > 0
            else 0.0
        )

        # Naive dense DFA comparison: [L, Dm, V] + attention [Dm, V]
        naive_dfa_params = (L + 1) * Dm * V
        dfa_savings_pct = (1.0 - (float(max_dfa) / float(max(naive_dfa_params, 1)))) * 100.0

        # Frozen Production W0 Only (destroys w, k, FactorizedHighwayDFA, attn_dfa, Kahan)
        frozen_at_init_params = init_frontend + init_spring_w
        frozen_at_max_params = max_frontend + max_spring_w

        return {
            "dtype": self.dtype,
            "dtype_bytes": b,
            "kahan_enabled": self.uses_kahan_compensator,
            "kahan_mode": eff_kahan,
            "kahan_multiplier": kahan_mult,
            "kahan_savings_vs_full_pct": kahan_savings_vs_full_pct,
            "init_active_params": init_active_params,
            "init_active_mb": (init_active_params * b) / mb,
            "max_train_params": max_train_params,
            "max_train_mb": (max_train_params * b) / mb,
            "max_train_gb": (max_train_params * b) / gb,
            "dfa_params": max_dfa,
            "dfa_mb": (max_dfa * b) / mb,
            "naive_dfa_params": naive_dfa_params,
            "naive_dfa_mb": (naive_dfa_params * b) / mb,
            "dfa_memory_savings_pct": dfa_savings_pct,
            "kahan_init_overhead_mb": kahan_init_bytes / mb,
            "kahan_max_overhead_mb": kahan_max_bytes / mb,
            "total_train_with_kahan_mb": ((max_train_params * b) + kahan_max_bytes) / mb,
            "total_train_with_kahan_gb": ((max_train_params * b) + kahan_max_bytes) / gb,
            "frozen_w0_at_init_params": frozen_at_init_params,
            "frozen_w0_at_init_mb": (frozen_at_init_params * b) / mb,
            "frozen_w0_params": frozen_at_max_params,
            "frozen_w0_mb": (frozen_at_max_params * b) / mb,
            "frozen_w0_gb": (frozen_at_max_params * b) / gb,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SSENNConfig":
        return cls(**dict(data))

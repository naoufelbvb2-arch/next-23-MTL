"""
models/model_v3_production.py
Phase 3.2: Clean Unified Production SS-ENN v3.0 Architecture (`SSENNProductionModel`).

Unified Zero-Autograd Production Design:
  - Strictly configured via explicit manual numbers (`SSENNConfig`).
  - Built directly on fused hardware kernels (`engine/kernels.py`):
      * Zero-Allocation In-Place Plasticity (`fused_spring_update_inplace`)
      * Active-Block Union Mini-Batch Kernels (`batch_active_block_union_update`)
      * Indexed Block Highway Forward (`indexed_block_activate_highway`)
      * Compact O(L * D^2) Feedback Alignment (`FactorizedHighwayDFA` + `[D, D]` `attn_dfa`)
      * ZERO `[D, V]` `DFARouter` or `B_attn` buffers anywhere
      * Mixed-Precision FP32 Kahan Compensator (`MixedPrecisionCompensator`) with
        deterministic string keys (`"layer_i"`, `"attn_q/k/v"`, `"output_head"`) for
        100% bitwise checkpoint save/resume parity
      * Stateful O(1) KV + EMA Capacitor + Refractory Cache (`SpringInferenceCache`)
  - Includes full continual learning lifecycle:
      * Clipped Log-Salience & Salience-Gated EMA Capacitor
      * Dual-Horizon Sparse Causal Attention + ALiBi + Lag-1/Lag-2 context
      * Contiguous Block-Sparse Elastic Capsules (Load-Balanced Top-K blocks)
      * Depth-Scaled Residual Highway (`alpha_res / sqrt(L)`) + Local Active RMSNorm
      * Frustration-Driven Dynamic Neurogenesis (on-demand deterministic row init)
      * Dual-Stage Offline Sleep Consolidation, Synaptic Recycling & Armor Locking
      * Critic-Free Three-Factor Reward Alignment
      * Production Freeze + Physical Dormant Pruning + Frozen W0 Export / Reload

Strictly zero autograd / zero backward().
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.kernels import (
    FactorizedHighwayDFA,
    MixedPrecisionCompensator,
    SpringInferenceCache,
    SpringScratchBuffer,
    batch_active_block_union_update,
    fused_spring_update_inplace,
    indexed_block_activate_highway,
    normalize_token_sequences,
)
from .config_v3 import SSENNConfig, _VALID_DTYPES


def _deterministic_uniform_rows(
    rows: int,
    cols: int,
    bound: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generates deterministic Kaiming-uniform weights on demand without permanent pool RAM."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    t = torch.empty((rows, cols), dtype=torch.float32, device="cpu").uniform_(-bound, bound, generator=gen)
    return t.to(device=device, dtype=dtype)


class _ProductionBlockLayer(nn.Module):
    """Unified block-sparse spring capsule layer plugging directly into `engine.kernels`."""

    def __init__(
        self,
        d_max: int,
        d_init: int,
        delta_d: int,
        block_size: int,
        negative_slope: float,
        eta_w: float,
        eta_0: float,
        gamma: float,
        beta_k: float,
        layer_id: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        skip_train_alloc: bool = False,
        compacted_out: int | None = None,
        compacted_in: int | None = None,
    ) -> None:
        super().__init__()
        self.d_max = int(d_max)
        self.d_init = int(d_init)
        self.d_active = int(compacted_out if compacted_out is not None else d_init)
        self.delta_d = int(delta_d)
        self.block_size = int(block_size)
        self.negative_slope = float(negative_slope)
        self.eta_w = float(eta_w)
        self.eta_0 = float(eta_0)
        self.gamma = float(gamma)
        self.beta_k = float(beta_k)
        self.layer_id = int(layer_id)
        self.seed = int(seed)
        self.last_growth_epoch: int = -100
        self._last_winner_k: float = 0.0
        self._compacted_in_dim: int | None = compacted_in

        # Structural tracking vectors (compact 1D float32)
        active_mask = torch.zeros(self.d_active if compacted_out is not None else self.d_max, device=device, dtype=torch.float32)
        active_mask[: self.d_active] = 1.0
        self.register_buffer("active_mask", active_mask)
        n_track = active_mask.shape[0]
        self.register_buffer("win_counts", torch.zeros(n_track, device=device, dtype=torch.float32))
        self.register_buffer("epoch_win_counts", torch.zeros(n_track, device=device, dtype=torch.float32))

        if skip_train_alloc:
            out_d = int(compacted_out or self.d_active)
            in_d = int(compacted_in or self.d_active)
            self.register_buffer("w", None)
            self.register_buffer("w0", torch.zeros((out_d, in_d), device=device, dtype=dtype))
            self.register_buffer("k", None)
        else:
            bound = 1.0 / math.sqrt(float(self.d_max))
            w_init = torch.zeros((self.d_max, self.d_max), device=device, dtype=dtype)
            w_init[: self.d_init, :] = _deterministic_uniform_rows(
                self.d_init, self.d_max, bound, seed=self.seed + 11 * self.layer_id, device=device, dtype=dtype
            )
            self.register_buffer("w", w_init)
            self.register_buffer("w0", w_init.clone())
            self.register_buffer("k", torch.zeros((self.d_max, self.d_max), device=device, dtype=dtype))

        self._last_a_prev: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None
        self._last_mask: torch.Tensor | None = None
        self._last_a: torch.Tensor | None = None

    @torch.no_grad()
    def grow_active(self, wake_count: int, salt: int) -> int:
        if self.w is None or wake_count <= 0 or self.d_active >= self.d_max:
            return 0
        old_d = self.d_active
        new_d = min(self.d_max, old_d + wake_count)
        actual = new_d - old_d
        if actual <= 0:
            return 0
        bound = 1.0 / math.sqrt(float(self.d_max))
        rows = _deterministic_uniform_rows(
            actual,
            self.d_max,
            bound,
            seed=self.seed + 100003 * salt + 7919 * self.layer_id + old_d,
            device=self.w.device,
            dtype=self.w.dtype,
        )
        self.w[old_d:new_d, :].copy_(rows)
        self.w0[old_d:new_d, :].copy_(rows)
        self.k[old_d:new_d, :].zero_()
        self.active_mask[old_d:new_d] = 1.0
        self.d_active = new_d
        return actual


class SSENNProductionModel(nn.Module):
    """Clean Unified SS-ENN v3.0 Production Network driven by explicit `SSENNConfig`."""

    # Buffers whose dtype follows config.dtype (float32 / bfloat16 / float16)
    _WEIGHT_BUFFER_NAMES = (
        "token_embedding",
        "pos_embedding",
        "alibi_pos_encoding",
        "alibi_causal_weights",
        "cap_proj",
        "lag1_proj",
        "lag2_proj",
        "q_w",
        "q_w0",
        "q_k",
        "k_w",
        "k_w0",
        "k_k",
        "v_w",
        "v_w0",
        "v_k",
        "attn_dfa",
        "head_w",
        "head_w0",
        "head_k",
        "head_w0_bedrock",
    )

    def __init__(
        self,
        config: SSENNConfig,
        _skip_train_alloc: bool = False,
        _frozen_dims: tuple[int, list[int]] | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = copy.deepcopy(config)

        # Structural geometry
        self.vocab_size = int(config.vocab_size)
        self.num_layers = int(config.num_layers)
        self.d_init = int(config.d_init)
        self.d_max = int(config.d_max)
        self.delta_d = int(config.delta_d)
        self.block_size = int(config.block_size)
        self.top_k_blocks = int(config.top_k_blocks)
        self.layer_top_k_blocks = config.get_layer_top_k_blocks()
        self.stiff_schedule = config.get_hierarchical_stiff_schedule()
        self.max_seq_len = int(config.max_seq_len)
        self.attn_top_k = int(config.attn_top_k)
        self.salient_anchor_slots = int(config.salient_anchor_slots)
        self.head_scale = float(config.head_scale)
        self.negative_slope = float(config.negative_slope)

        device = config.resolve_torch_device()
        w_dtype = config.torch_dtype

        self.frozen: bool = bool(_skip_train_alloc)
        self._is_training_step: bool = False
        self._in_fast_generate: bool = False
        self.current_epoch: int = 0
        self.reward_baseline: float = 0.0
        self.attn_armor_locked: bool = False
        self.has_multiword_entities: bool = False
        self.attn_v_topk: int = min(8, self.d_init)

        # Kernel scratch & FP32 Kahan residual compensator (deterministic string keys)
        self._kernel_scratch: SpringScratchBuffer | None = None if _skip_train_alloc else SpringScratchBuffer()
        self.compensator: MixedPrecisionCompensator | None = (
            None
            if _skip_train_alloc
            else MixedPrecisionCompensator(mode=config.effective_kahan_mode)
        )

        self._rejuv_gen = torch.Generator(device="cpu")
        self._rejuv_gen.manual_seed(config.seed + 777)

        # Tracking buffers shared by training, sleep, and inference
        self.register_buffer("token_salience", torch.ones(self.vocab_size, device=device, dtype=torch.float32))
        self.register_buffer("capacitor_alpha", torch.zeros(self.vocab_size, device=device, dtype=torch.float32))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(config.seed + 202)

        n_blocks_max = max(1, self.d_max // self.block_size)
        self.block_win_counts: list[torch.Tensor] = [
            torch.zeros(n_blocks_max, device=device, dtype=torch.float32) for _ in range(self.num_layers)
        ]
        self.epoch_block_win_counts: list[torch.Tensor] = [
            torch.zeros(n_blocks_max, device=device, dtype=torch.float32) for _ in range(self.num_layers)
        ]
        self._last_epoch_block_wins: list[torch.Tensor] = [
            torch.zeros(n_blocks_max, device=device, dtype=torch.float32) for _ in range(self.num_layers)
        ]
        self.block_usage_ema: list[torch.Tensor] = [
            torch.ones(n_blocks_max, device=device, dtype=torch.float32) / float(n_blocks_max)
            for _ in range(self.num_layers)
        ]
        self.armor_locked_masks: list[torch.Tensor] = [
            torch.zeros(self.d_max, device=device, dtype=torch.bool) for _ in range(self.num_layers)
        ]
        self.register_buffer("head_armor_mask", torch.zeros(self.vocab_size, device=device, dtype=torch.bool))
        self.register_buffer("locked_excl_vocab_mask", torch.zeros(self.vocab_size, device=device, dtype=torch.bool))
        self.register_buffer("phase_active_vocab_mask", torch.ones(self.vocab_size, device=device, dtype=torch.bool))
        self.register_buffer("phase_seen_targets", torch.zeros(self.vocab_size, device=device, dtype=torch.float32))
        self.register_buffer("vocab_phase_counts", torch.zeros(self.vocab_size, device=device, dtype=torch.float32))

        if _skip_train_alloc and _frozen_dims is not None:
            f_eff, f_layer_outs = _frozen_dims
            self._alloc_frozen_skeleton(f_eff, f_layer_outs, device, w_dtype)
        else:
            self._alloc_full_training_buffers(device, w_dtype, gen)

        self.rejuvenated_slots: list[list[int]] = [[] for _ in range(self.num_layers)]
        self.total_recycled_cells_per_layer: list[int] = [0 for _ in range(self.num_layers)]
        self.current_phase_slots: list[list[int]] = [[] for _ in range(self.num_layers)]
        self.growth_log: list[dict] = []
        self.sleep_logs: list[dict] = []
        self.alignment_logs: list[dict] = []
        self._last_layer_raw_rms: list[float] = [0.0 for _ in range(self.num_layers)]
        self._last_layer_normed_rms: list[float] = [0.0 for _ in range(self.num_layers)]
        self._last_v_prev: torch.Tensor | None = None
        self._last_v_z: torch.Tensor | None = None
        self._last_v_mask: torch.Tensor | None = None
        self._last_q_prev: torch.Tensor | None = None

        for p in self.parameters():
            p.requires_grad_(False)

    def _alloc_full_training_buffers(
        self,
        device: torch.device,
        w_dtype: torch.dtype,
        gen: torch.Generator,
    ) -> None:
        cfg = self.config
        V, D, T = self.vocab_size, self.d_max, self.max_seq_len

        # Frontend embeddings & ALiBi window
        emb = torch.randn((V, D), generator=gen, dtype=torch.float32)
        emb = F.normalize(emb, p=2, dim=-1) * math.sqrt(D / 3.0)
        self.register_buffer("token_embedding", emb.to(device=device, dtype=w_dtype))

        pos = torch.randn((T, D), generator=gen, dtype=torch.float32)
        pos = F.normalize(pos, p=2, dim=-1) * math.sqrt(D / 6.0)
        self.register_buffer("pos_embedding", pos.to(device=device, dtype=w_dtype))

        pos_enc = torch.zeros((T, D), dtype=torch.float32)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(1000.0) / float(D)))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = F.normalize(pos_enc, p=2, dim=-1) * math.sqrt(D / 5.0)
        self.register_buffer("alibi_pos_encoding", pos_enc.to(device=device, dtype=w_dtype))

        causal_tril = torch.tril(torch.ones((T, T), dtype=torch.float32))
        idx = torch.arange(T)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0).float()
        alibi_w = (0.68 ** dist) * causal_tril
        alibi_w[:, 0] += 0.30 * causal_tril[:, 0]
        if T > 1:
            alibi_w[:, 1] += 0.20 * causal_tril[:, 1]
        alibi_w = alibi_w / alibi_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("alibi_causal_weights", alibi_w.to(device=device, dtype=w_dtype))

        # Capacitor & local lag structural projections
        scale_d = 1.0 / math.sqrt(float(D))
        for name in ("cap_proj", "lag1_proj", "lag2_proj"):
            proj = torch.randn((D, D), generator=gen, dtype=torch.float32) * scale_d
            self.register_buffer(name, proj.to(device=device, dtype=w_dtype))

        # Q, K, V Attention Spring triplets + compact [D_max, D_max] attention DFA adapter
        bound_d = 1.0 / math.sqrt(float(D))
        for offset, prefix in enumerate(("q", "k", "v"), start=1):
            w_init = _deterministic_uniform_rows(D, D, bound_d, seed=cfg.seed + 202 + offset, device=device, dtype=w_dtype)
            self.register_buffer(f"{prefix}_w", w_init)
            self.register_buffer(f"{prefix}_w0", w_init.clone())
            self.register_buffer(f"{prefix}_k", torch.zeros((D, D), device=device, dtype=w_dtype))

        eye = torch.eye(D, dtype=torch.float32)
        noise = torch.randn((D, D), generator=gen, dtype=torch.float32) * scale_d
        attn_dfa = F.normalize(0.80 * eye + 0.20 * noise, p=2, dim=-1, eps=1e-6)
        self.register_buffer("attn_dfa", attn_dfa.to(device=device, dtype=w_dtype))

        # Block-Sparse Elastic Spring Layers
        self.spring_layers = nn.ModuleList([
            _ProductionBlockLayer(
                d_max=D,
                d_init=self.d_init,
                delta_d=self.delta_d,
                block_size=self.block_size,
                negative_slope=self.negative_slope,
                eta_w=cfg.lr_w,
                eta_0=cfg.lr_w0,
                gamma=cfg.gamma_restore,
                beta_k=cfg.beta_k,
                layer_id=l_idx + 1,
                seed=cfg.seed + 10 * (l_idx + 1),
                device=device,
                dtype=w_dtype,
            )
            for l_idx in range(self.num_layers)
        ])

        # Factorized Highway DFA (Zero [D, V] matrices)
        self.factorized_dfa: FactorizedHighwayDFA | None = FactorizedHighwayDFA(
            num_layers=self.num_layers,
            d_max=D,
            vocab_size=V,
            eta_b=cfg.lr_dfa,
            dfa_anchor_blend=cfg.dfa_anchor_blend,
            seed=cfg.seed + 500,
        ).to(device=device, dtype=w_dtype)

        # Output Head Spring triplet [V, D_max] (active [:d_init] populated, [d_init:] zeroed)
        head_init = torch.zeros((V, D), device=device, dtype=w_dtype)
        head_init[:, : self.d_init] = _deterministic_uniform_rows(
            V, self.d_init, bound_d, seed=cfg.seed + 999, device=device, dtype=w_dtype
        )
        self.register_buffer("head_w", head_init)
        self.register_buffer("head_w0", head_init.clone())
        self.register_buffer("head_k", torch.zeros((V, D), device=device, dtype=w_dtype))

        # Continual bedrock snapshots (for sleep consolidation)
        self.register_buffer("head_w0_bedrock", None)
        self.layer_w0_bedrock: list[torch.Tensor] = []

    def _alloc_frozen_skeleton(
        self,
        d_eff: int,
        layer_outs: list[int],
        device: torch.device,
        w_dtype: torch.dtype,
    ) -> None:
        V, T = self.vocab_size, self.max_seq_len
        self.register_buffer("token_embedding", torch.zeros((V, d_eff), device=device, dtype=w_dtype))
        self.register_buffer("pos_embedding", torch.zeros((T, d_eff), device=device, dtype=w_dtype))
        self.register_buffer("alibi_pos_encoding", torch.zeros((T, d_eff), device=device, dtype=w_dtype))
        self.register_buffer("alibi_causal_weights", torch.zeros((T, T), device=device, dtype=w_dtype))
        for name in ("cap_proj", "lag1_proj", "lag2_proj", "q_w0", "k_w0", "v_w0"):
            self.register_buffer(name, torch.zeros((d_eff, d_eff), device=device, dtype=w_dtype))
        for name in ("q_w", "q_k", "k_w", "k_k", "v_w", "v_k", "attn_dfa", "head_w", "head_k", "head_w0_bedrock"):
            self.register_buffer(name, None)

        d_last = layer_outs[-1]
        self.register_buffer("head_w0", torch.zeros((V, d_last), device=device, dtype=w_dtype))
        self.factorized_dfa = None
        self.layer_w0_bedrock = []
        cfg = self.config
        self.spring_layers = nn.ModuleList([
            _ProductionBlockLayer(
                d_max=self.d_max,
                d_init=self.d_init,
                delta_d=self.delta_d,
                block_size=self.block_size,
                negative_slope=self.negative_slope,
                eta_w=cfg.lr_w,
                eta_0=cfg.lr_w0,
                gamma=cfg.gamma_restore,
                beta_k=cfg.beta_k,
                layer_id=l_idx + 1,
                seed=cfg.seed + 10 * (l_idx + 1),
                device=device,
                dtype=w_dtype,
                skip_train_alloc=True,
                compacted_out=layer_outs[l_idx],
                compacted_in=d_eff,
            )
            for l_idx in range(self.num_layers)
        ])

    @property
    def model_device(self) -> torch.device:
        return self.token_embedding.device

    @property
    def weight_dtype(self) -> torch.dtype:
        return self.token_embedding.dtype

    @torch.no_grad()
    def _effective_dims(self) -> tuple[int, int]:
        """Returns `(d_eff, d_last)` matching current active/compacted feature dimensions."""
        d_eff = max(layer.w0.shape[1] if (self.frozen and layer._compacted_in_dim is not None) else layer.d_active for layer in self.spring_layers)
        if self.frozen and self.spring_layers[-1]._compacted_in_dim is not None:
            d_last = self.spring_layers[-1].w0.shape[0]
        else:
            d_last = self.spring_layers[-1].d_active
        return int(d_eff), int(d_last)

    @torch.no_grad()
    def get_active_dimensions(self) -> list[int]:
        dims = []
        for l_idx, layer in enumerate(self.spring_layers):
            if self.frozen:
                dims.append(int(layer.w0.shape[0]))
            else:
                dims.append(int(layer.d_active - len(self.rejuvenated_slots[l_idx])))
        return dims

    @torch.no_grad()
    def get_allocated_dimensions(self) -> list[int]:
        return [int(layer.w0.shape[0] if self.frozen else layer.d_active) for layer in self.spring_layers]

    @torch.no_grad()
    def to_device_and_dtype(
        self,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> "SSENNProductionModel":
        """Explicitly moves weights/counters to target device & converts weights cleanly to target dtype."""
        target_dev = torch.device(device) if device is not None else self.model_device
        if dtype is None:
            target_dtype_str = self.config.dtype
        elif isinstance(dtype, torch.dtype):
            inv = {v: k for k, v in _VALID_DTYPES.items()}
            if dtype not in inv:
                raise ValueError(f"Unsupported torch.dtype {dtype}")
            target_dtype_str = inv[dtype]
        else:
            if dtype not in _VALID_DTYPES:
                raise ValueError(f"Unsupported dtype '{dtype}'")
            target_dtype_str = dtype
        target_dtype = _VALID_DTYPES[target_dtype_str]

        # Validate updated config cleanly while preserving explicit kahan_mode
        self.config.device = str(target_dev)
        self.config.dtype = target_dtype_str
        self.config.use_kahan_compensator = (
            target_dtype_str in ("bfloat16", "float16")
        ) and (self.config.kahan_mode != "off")
        self.config.validate()

        for bname in self._WEIGHT_BUFFER_NAMES:
            buf = getattr(self, bname, None)
            if buf is not None:
                setattr(self, bname, buf.to(device=target_dev, dtype=target_dtype))

        for bname in ("token_salience", "capacitor_alpha", "phase_seen_targets", "vocab_phase_counts"):
            buf = getattr(self, bname, None)
            if buf is not None:
                setattr(self, bname, buf.to(device=target_dev, dtype=torch.float32))

        for bname in ("head_armor_mask", "locked_excl_vocab_mask", "phase_active_vocab_mask"):
            buf = getattr(self, bname, None)
            if buf is not None:
                setattr(self, bname, buf.to(device=target_dev, dtype=torch.bool))

        for layer in self.spring_layers:
            for lname in ("w", "w0", "k"):
                t = getattr(layer, lname, None)
                if t is not None:
                    setattr(layer, lname, t.to(device=target_dev, dtype=target_dtype))
            for lname in ("active_mask", "win_counts", "epoch_win_counts"):
                t = getattr(layer, lname, None)
                if t is not None:
                    setattr(layer, lname, t.to(device=target_dev, dtype=torch.float32))

        if self.factorized_dfa is not None:
            self.factorized_dfa = self.factorized_dfa.to(device=target_dev, dtype=target_dtype)

        self.block_win_counts = [t.to(device=target_dev, dtype=torch.float32) for t in self.block_win_counts]
        self.epoch_block_win_counts = [t.to(device=target_dev, dtype=torch.float32) for t in self.epoch_block_win_counts]
        self._last_epoch_block_wins = [t.to(device=target_dev, dtype=torch.float32) for t in self._last_epoch_block_wins]
        self.block_usage_ema = [t.to(device=target_dev, dtype=torch.float32) for t in self.block_usage_ema]
        self.armor_locked_masks = [t.to(device=target_dev, dtype=torch.bool) for t in self.armor_locked_masks]
        self.layer_w0_bedrock = [t.to(device=target_dev, dtype=target_dtype) for t in self.layer_w0_bedrock]

        if self.compensator is not None:
            eff_mode = self.config.effective_kahan_mode
            self.compensator.mode = eff_mode
            if eff_mode == "off":
                self.compensator._res_w.clear()
                self.compensator._res_w0.clear()
                self.compensator._res_k.clear()
            elif eff_mode == "anchor_k":
                self.compensator._res_w.clear()
            for store in (self.compensator._res_w, self.compensator._res_w0, self.compensator._res_k):
                for ck in list(store.keys()):
                    store[ck] = store[ck].to(device=target_dev, dtype=torch.float32)

        if self._kernel_scratch is not None:
            self._kernel_scratch = SpringScratchBuffer()

        self._reset_step_cache()
        return self

    def _reset_step_cache(self) -> None:
        self._last_v_prev = None
        self._last_v_z = None
        self._last_v_mask = None
        self._last_q_prev = None
        for layer in self.spring_layers:
            layer._last_a_prev = None
            layer._last_z = None
            layer._last_mask = None
            layer._last_a = None

    @staticmethod
    @torch.no_grad()
    def _apply_subspace_rmsnorm(
        x_sub: torch.Tensor,
        target_rms: float = 1.20,
        return_stats: bool = False,
    ) -> tuple[torch.Tensor, float, float]:
        x_f = x_sub if x_sub.dtype == torch.float32 else x_sub.float()
        raw_rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        normed = (x_f * (target_rms / raw_rms)).to(x_sub.dtype)
        if return_stats:
            raw_mean = float(raw_rms.mean().item())
            norm_mean = float(torch.sqrt(normed.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).mean().item())
            return normed, raw_mean, norm_mean
        return normed, 0.0, 0.0

    @torch.no_grad()
    def calibrate_salience_from_tokens(
        self,
        token_sequences,
        pad_id: int = 0,
        bos_id: int | None = 1,
        eos_id: int | None = 2,
    ) -> dict[str, float]:
        """Law 1: Clipped Log-Salience s(v) in [0.70, 1.80] & EMA Capacitor alpha(v)."""
        norm_seqs = normalize_token_sequences(token_sequences)
        pad_tok = int(pad_id)
        bos_tok = int(bos_id) if bos_id is not None else None
        eos_tok = int(eos_id) if eos_id is not None else None
        special_ids = {t for t in (pad_tok, bos_tok, eos_tok) if t is not None}

        counts = torch.zeros(self.vocab_size, dtype=torch.float32)
        doc_freq = torch.zeros(self.vocab_size, dtype=torch.float32)
        burst_freq = torch.zeros(self.vocab_size, dtype=torch.float32)
        unigram_cross_burst = torch.zeros(self.vocab_size, dtype=torch.float32)
        bigram_df: dict[tuple[int, int], float] = {}
        bigram_cross_burst: dict[tuple[int, int], float] = {}
        n_docs = max(len(norm_seqs), 1)

        for seq in norm_seqs:
            seq_counts: dict[int, int] = {}
            uni_pos: dict[int, list[int]] = {}
            bi_pos: dict[tuple[int, int], list[int]] = {}
            prev_tid = -1
            for pos, raw_tok in enumerate(seq):
                tid = int(raw_tok)
                if 0 <= tid < self.vocab_size and tid not in special_ids:
                    counts[tid] += 1.0
                    seq_counts[tid] = seq_counts.get(tid, 0) + 1
                    uni_pos.setdefault(tid, []).append(pos)
                    if prev_tid >= 0:
                        bi_pos.setdefault((prev_tid, tid), []).append(pos)
                    prev_tid = tid
                else:
                    prev_tid = -1
            for tid, c in seq_counts.items():
                doc_freq[tid] += 1.0
                if c >= 2:
                    burst_freq[tid] += 1.0
                plist = uni_pos[tid]
                if len(plist) >= 2 and (plist[-1] - plist[0]) >= 4:
                    unigram_cross_burst[tid] += 1.0
            for bg, plist in bi_pos.items():
                bigram_df[bg] = bigram_df.get(bg, 0.0) + 1.0
                if len(plist) >= 2 and (plist[-1] - plist[0]) >= 4:
                    bigram_cross_burst[bg] = bigram_cross_burst.get(bg, 0.0) + 1.0

        phrase_burst_score = torch.zeros(self.vocab_size, dtype=torch.float32)
        df_thresh = 0.30 * float(n_docs)
        for (u, v), b_df in bigram_df.items():
            if b_df > 0 and doc_freq[u] < df_thresh and doc_freq[v] < df_thresh:
                ratio = bigram_cross_burst.get((u, v), 0.0) / b_df
                if ratio >= 0.45:
                    if ratio > float(phrase_burst_score[u].item()):
                        phrase_burst_score[u] = ratio
                    if ratio > float(phrase_burst_score[v].item()):
                        phrase_burst_score[v] = ratio

        total_tokens = float(counts.sum().item())
        active_vocab = counts > 0
        if total_tokens > 0 and active_vocab.any():
            inv_log_tf = torch.log((total_tokens + 1.0) / (counts + 1.0))
            idf = torch.log((float(n_docs) + 1.0) / (doc_freq + 1.0))
            selective_mask = (doc_freq < df_thresh).float()
            burst_ratio = (burst_freq / (doc_freq + 1.0)) * selective_mask
            raw_info = 0.50 * inv_log_tf + 0.40 * idf + 1.50 * burst_ratio + 2.20 * phrase_burst_score
            min_val = float(raw_info[active_vocab].min().item())
            max_val = float(raw_info[active_vocab].max().item())
            span = max(max_val - min_val, 1e-6)
            salience = (0.70 + 1.10 * ((raw_info - min_val) / span)).clamp(0.70, 1.80)
            salience = torch.where(active_vocab, salience, torch.ones_like(salience))
        else:
            salience = torch.ones(self.vocab_size, dtype=torch.float32)

        if 0 <= pad_tok < self.vocab_size:
            salience[pad_tok] = 0.0
        if bos_tok is not None and 0 <= bos_tok < self.vocab_size:
            salience[bos_tok] = 0.70
        if eos_tok is not None and 0 <= eos_tok < self.vocab_size:
            salience[eos_tok] = 0.80
        self.token_salience.copy_(salience.to(self.model_device))

        phrase_mask = phrase_burst_score >= 0.45
        single_mask = (
            (unigram_cross_burst / (doc_freq + 1e-6) >= 0.65)
            & (doc_freq < (0.25 * float(n_docs)))
            & (salience >= 1.45)
        )
        self.has_multiword_entities = bool(phrase_mask.any().item())

        rel_sal = ((salience - 0.70) / 1.10).clamp(0.0, 1.0)
        alpha = torch.where(salience >= 1.45, 0.38 + 0.12 * rel_sal, 0.004 * (rel_sal ** 2))
        if 0 <= pad_tok < self.vocab_size:
            alpha[pad_tok] = 0.0
        if bos_tok is not None and 0 <= bos_tok < self.vocab_size:
            alpha[bos_tok] = 0.0
        self.capacitor_alpha.copy_(alpha.to(self.model_device))

        seen_sal = salience[active_vocab]
        return {
            "min_salience": float(seen_sal.min().item()),
            "max_salience": float(seen_sal.max().item()),
            "mean_salience": float(seen_sal.mean().item()),
        }

    def _qkv_weights(self, d_eff: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.frozen:
            return self.q_w0[:d_eff, :d_eff], self.k_w0[:d_eff, :d_eff], self.v_w0[:d_eff, :d_eff]
        return self.q_w[:d_eff, :d_eff], self.k_w[:d_eff, :d_eff], self.v_w[:d_eff, :d_eff]

    @torch.no_grad()
    def _project_v_sparse(
        self,
        x_flat: torch.Tensor,
        v_weight: torch.Tensor,
        store_cache: bool,
    ) -> torch.Tensor:
        z_v = F.linear(x_flat, v_weight)
        d_eff = z_v.shape[-1]
        k_v = min(self.attn_v_topk, d_eff)
        if 0 < k_v < d_eff:
            _, top_idx = torch.topk(z_v.abs(), k=k_v, dim=-1)
            mask_v = torch.zeros_like(z_v)
            mask_v.scatter_(-1, top_idx, 1.0)
        else:
            mask_v = torch.ones_like(z_v)
        a_v = F.leaky_relu(z_v, negative_slope=self.negative_slope) * mask_v
        if store_cache and not self.frozen:
            self._last_v_prev = x_flat.detach()
            self._last_v_z = z_v.detach()
            self._last_v_mask = mask_v.detach()
        return x_flat + a_v

    @torch.no_grad()
    def _apply_dual_horizon_topk(
        self,
        scores: torch.Tensor,
        sal_keys: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Shared Dual-Horizon Sparse Causal Top-K Selection used identically in
        `forward()` and `forward_step_cached()` to guarantee 100.00% parity.
        `scores` shape: `[B, Q, K]`, `sal_keys` shape: `[B, 1, K]`.
        """
        k_len = scores.shape[-1]
        if self.attn_top_k >= k_len:
            return scores

        k_anchor = min(self.salient_anchor_slots, k_len)
        k_local = min(max(1, self.attn_top_k - k_anchor), k_len)
        sal_rank = 4.0 * sal_keys + 0.15 * scores.float()
        if causal_mask is not None:
            sal_rank = sal_rank.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

        sparse_scores = torch.full_like(scores, float("-inf"))
        _, anchor_idx = torch.topk(sal_rank, k=k_anchor, dim=-1)
        sparse_scores.scatter_(-1, anchor_idx, scores.gather(-1, anchor_idx))
        _, local_idx = torch.topk(scores, k=k_local, dim=-1)
        sparse_scores.scatter_(-1, local_idx, scores.gather(-1, local_idx))
        return sparse_scores

    @torch.no_grad()
    def _build_vectorized_context(self, input_ids: torch.Tensor, d_eff: int) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        tok_emb = self.token_embedding[input_ids, :d_eff]
        pos_emb = self.pos_embedding[:seq_len, :d_eff].unsqueeze(0)
        x = tok_emb + pos_emb
        x_flat = x.reshape(bsz * seq_len, d_eff)

        q_w_sub, k_w_sub, v_w_sub = self._qkv_weights(d_eff)
        q = F.linear(x_flat, q_w_sub).reshape(bsz, seq_len, d_eff)
        k = F.linear(x_flat, k_w_sub).reshape(bsz, seq_len, d_eff)
        store_attn_cache = self._is_training_step and (not self.frozen)
        v = self._project_v_sparse(x_flat, v_w_sub, store_cache=store_attn_cache).reshape(bsz, seq_len, d_eff)
        if store_attn_cache:
            self._last_q_prev = x_flat.detach()

        scale = 1.0 / math.sqrt(float(d_eff))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        key_sal = self.token_salience[input_ids].unsqueeze(1)  # [B, 1, T]
        scores = scores + (1.35 * (key_sal - 1.0)).to(scores.dtype)

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
        scores = self._apply_dual_horizon_topk(scores, key_sal, causal_mask=causal_mask)

        attn_weights = F.softmax(scores.float(), dim=-1) * causal_mask.unsqueeze(0).float()
        attended = torch.matmul(attn_weights.to(v.dtype), v) + x

        # Salience-Gated EMA Context Capacitor
        alphas = self.capacitor_alpha[input_ids].unsqueeze(-1).to(tok_emb.dtype)
        c_states = torch.empty_like(tok_emb)
        c_prev = torch.zeros((bsz, d_eff), device=tok_emb.device, dtype=tok_emb.dtype)
        for t in range(seq_len):
            a_t = alphas[:, t, :]
            c_prev = (1.0 - a_t) * c_prev + a_t * tok_emb[:, t, :]
            c_states[:, t, :] = c_prev

        w_alibi = self.alibi_causal_weights[:seq_len, :seq_len]
        x_pos = tok_emb + self.alibi_pos_encoding[:seq_len, :d_eff].unsqueeze(0)
        alibi_ctx = torch.matmul(w_alibi.unsqueeze(0), x_pos)
        cap_ctx = F.linear(c_states, self.cap_proj[:d_eff, :d_eff])

        lag1 = torch.zeros_like(tok_emb)
        if seq_len > 1:
            lag1[:, 1:, :] = F.linear(tok_emb[:, :-1, :], self.lag1_proj[:d_eff, :d_eff])
        lag2 = torch.zeros_like(tok_emb)
        if seq_len > 2:
            lag2[:, 2:, :] = F.linear(tok_emb[:, :-2, :], self.lag2_proj[:d_eff, :d_eff])

        return attended + alibi_ctx + 1.65 * cap_ctx + lag1 + lag2

    @torch.no_grad()
    def _forward_block_layer(
        self,
        l_idx: int,
        layer: _ProductionBlockLayer,
        a_prev_eff: torch.Tensor,
        in_fast_gen: bool = False,
    ) -> torch.Tensor:
        in_dim = a_prev_eff.shape[1]
        if self.frozen:
            w_sub = layer.w0[:, :in_dim] if layer.w0.shape[1] != in_dim else layer.w0
            z_act = F.linear(a_prev_eff, w_sub)
            d_sub = z_act.shape[1]
        else:
            d_sub = layer.d_active
            z_act = F.linear(a_prev_eff, layer.w[:d_sub, :in_dim])

        n_tok = z_act.shape[0]
        b_size = self.block_size
        n_blocks = max(1, d_sub // b_size)
        k_blk = min(self.layer_top_k_blocks[l_idx], n_blocks)

        z_blocks = z_act.view(n_tok, n_blocks, b_size)
        block_energy = torch.sqrt(z_blocks.float().pow(2).mean(dim=-1) + 1e-6)

        if (
            self._is_training_step
            and (not self.frozen)
            and k_blk < n_blocks
            and self.config.block_balance_strength > 0.0
        ):
            ema = self.block_usage_ema[l_idx][:n_blocks]
            rel_u = ema / ema.mean().clamp(min=1e-6)
            rank_score = block_energy * (rel_u + 0.10).pow(-self.config.block_balance_strength)
        else:
            rank_score = block_energy

        res_scale = self.config.alpha_res / math.sqrt(float(self.num_layers))
        if self.frozen and k_blk < n_blocks:
            _, top_b_idx = torch.topk(rank_score, k=k_blk, dim=-1)
            raw_out = indexed_block_activate_highway(
                z_act=z_act,
                a_prev_eff=a_prev_eff,
                top_b_idx=top_b_idx,
                block_size=b_size,
                neg_slope=layer.negative_slope,
                res_scale=res_scale,
                use_highway=True,
            )
            if d_sub == in_dim:
                out, raw_rms, normed_rms = self._apply_subspace_rmsnorm(
                    raw_out, target_rms=self.config.target_rms, return_stats=not in_fast_gen
                )
            else:
                out_sub, raw_rms, normed_rms = self._apply_subspace_rmsnorm(
                    raw_out[:, :d_sub], target_rms=self.config.target_rms, return_stats=not in_fast_gen
                )
                out = raw_out
                out[:, :d_sub] = out_sub
            self._last_layer_raw_rms[l_idx] = raw_rms
            self._last_layer_normed_rms[l_idx] = normed_rms
            return out

        if k_blk < n_blocks:
            _, top_b_idx = torch.topk(rank_score, k=k_blk, dim=-1)
            idx_exp = top_b_idx.unsqueeze(-1).expand(-1, -1, b_size)
            z_win = torch.gather(z_blocks, dim=1, index=idx_exp)
            a_win = F.leaky_relu(z_win, negative_slope=layer.negative_slope)
            a_sparse = torch.zeros_like(z_blocks).scatter_(1, idx_exp, a_win).reshape(n_tok, d_sub)
            block_mask = torch.zeros((n_tok, n_blocks), device=z_act.device, dtype=torch.float32)
            block_mask.scatter_(1, top_b_idx, 1.0)
        else:
            a_sparse = F.leaky_relu(z_act, negative_slope=layer.negative_slope)
            block_mask = torch.ones((n_tok, n_blocks), device=z_act.device, dtype=torch.float32)

        if self._is_training_step and (not self.frozen):
            batch_b_wins = block_mask.sum(dim=0)
            self.block_win_counts[l_idx][:n_blocks].add_(batch_b_wins)
            self.epoch_block_win_counts[l_idx][:n_blocks].add_(batch_b_wins)
            ema_slice = self.block_usage_ema[l_idx][:n_blocks]
            ema_slice.mul_(0.88).add_(0.12 * (batch_b_wins / max(float(n_tok), 1.0)))

            cell_wins = batch_b_wins.repeat_interleave(b_size)
            layer.win_counts[:d_sub].add_(cell_wins)
            layer.epoch_win_counts[:d_sub].add_(cell_wins)

            layer._last_a_prev = a_prev_eff.detach()
            layer._last_z = z_act.detach()
            layer._last_mask = block_mask.detach()
            layer._last_a = a_sparse.detach()

        need_stats = (not self._is_training_step) and (not in_fast_gen)
        if d_sub == in_dim:
            raw_out = a_prev_eff + res_scale * a_sparse
            out, raw_rms, normed_rms = self._apply_subspace_rmsnorm(
                raw_out, target_rms=self.config.target_rms, return_stats=need_stats
            )
        else:
            raw_out = a_prev_eff.clone()
            raw_out[:, :d_sub].add_(a_sparse, alpha=res_scale)
            out_sub, raw_rms, normed_rms = self._apply_subspace_rmsnorm(
                raw_out[:, :d_sub], target_rms=self.config.target_rms, return_stats=need_stats
            )
            out = raw_out
            out[:, :d_sub] = out_sub

        self._last_layer_raw_rms[l_idx] = raw_rms
        self._last_layer_normed_rms[l_idx] = normed_rms
        return out

    @torch.no_grad()
    def _forward_hidden_and_head(
        self,
        a_ctx: torch.Tensor,
        d_last: int,
        in_fast_gen: bool = False,
        cached_head_w0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, _, _ = self._apply_subspace_rmsnorm(a_ctx, target_rms=self.config.target_rms, return_stats=False)
        for l_idx, layer in enumerate(self.spring_layers):
            a = self._forward_block_layer(l_idx, layer, a, in_fast_gen=in_fast_gen)

        if self.frozen:
            h_w0 = cached_head_w0 if cached_head_w0 is not None else self.head_w0
            in_dim = h_w0.shape[1]
            a_head = a[:, :in_dim] if a.shape[1] != in_dim else a
            logits = F.linear(a_head, h_w0)
        else:
            a_head = a[:, :d_last] if a.shape[1] != d_last else a
            logits = F.linear(a_head, self.head_w[:, :d_last])
        if self.head_scale != 1.0:
            logits = logits * self.head_scale
        return logits, a_head

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor | list[int]) -> torch.Tensor:
        single_seq = False
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.model_device)
            single_seq = True
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0).to(self.model_device)
            single_seq = True
        else:
            input_ids = input_ids.to(self.model_device)

        bsz, seq_len = input_ids.shape
        d_eff, d_last = self._effective_dims()
        ctx = self._build_vectorized_context(input_ids, d_eff)
        a_ctx = ctx.reshape(bsz * seq_len, d_eff)
        logits_flat, _ = self._forward_hidden_and_head(a_ctx, d_last=d_last, in_fast_gen=False)
        logits = logits_flat.reshape(bsz, seq_len, self.vocab_size)
        if single_seq:
            return logits[0, -1]
        return logits

    @torch.no_grad()
    def _update_attention_springs(
        self,
        e_highway: torch.Tensor,
        d_eff: int,
        n_total: float,
    ) -> None:
        if self.attn_armor_locked:
            return
        if self._last_v_prev is None or self._last_v_z is None or self._last_v_mask is None:
            return
        cfg = self.config
        d_last = e_highway.shape[1]
        attn_err = F.linear(e_highway, self.attn_dfa[:d_eff, :d_last])
        x_flat = self._last_v_prev
        v_z = self._last_v_z
        v_mask = self._last_v_mask

        g_prime = torch.ones_like(v_z)
        g_prime.masked_fill_(v_z <= 0.0, self.negative_slope)
        v_delta = attn_err * g_prime * v_mask
        drv_v = torch.matmul(v_delta.t(), x_flat).div_(max(n_total, 1.0))

        fused_spring_update_inplace(
            w=self.v_w[:d_eff, :d_eff],
            w0=self.v_w0[:d_eff, :d_eff],
            k=self.v_k[:d_eff, :d_eff],
            driving_signal=drv_v,
            eta_w=cfg.lr_attn_w,
            eta_0=cfg.lr_attn_0,
            gamma=cfg.gamma_attn,
            beta_k_eff=cfg.beta_k_attn,
            stiff_ceil=1.0,
            scratch=self._kernel_scratch,
            compensator=self.compensator,
            comp_key="attn_v",
        )

        qk_err = attn_err * 0.25
        drv_qk = torch.matmul(qk_err.t(), x_flat).div_(max(n_total, 1.0))
        for prefix in ("q", "k"):
            fused_spring_update_inplace(
                w=getattr(self, f"{prefix}_w")[:d_eff, :d_eff],
                w0=getattr(self, f"{prefix}_w0")[:d_eff, :d_eff],
                k=getattr(self, f"{prefix}_k")[:d_eff, :d_eff],
                driving_signal=drv_qk,
                eta_w=cfg.lr_attn_w,
                eta_0=cfg.lr_attn_0,
                gamma=cfg.gamma_attn,
                beta_k_eff=cfg.beta_k_attn,
                stiff_ceil=1.0,
                scratch=self._kernel_scratch,
                compensator=self.compensator,
                comp_key=f"attn_{prefix}",
            )

    @torch.no_grad()
    def train_minibatch(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        pad_id: int = 0,
    ) -> dict[str, float]:
        """Single Zero-Autograd vectorized minibatch plasticity step using fused kernels."""
        if self.frozen:
            raise RuntimeError("Model is frozen; training is disabled.")

        input_ids = input_ids.to(self.model_device)
        target_ids = target_ids.to(self.model_device)
        bsz, seq_len = input_ids.shape
        n_total = bsz * seq_len
        d_eff, d_last = self._effective_dims()

        self._is_training_step = True
        try:
            ctx = self._build_vectorized_context(input_ids, d_eff)
            logits_flat, a_head = self._forward_hidden_and_head(ctx.reshape(n_total, d_eff), d_last=d_last)
        finally:
            self._is_training_step = False

        targets_flat = target_ids.reshape(n_total)
        inputs_flat = input_ids.reshape(n_total)
        valid_rows = targets_flat != pad_id
        n_valid = float(valid_rows.sum().item())
        if n_valid == 0.0:
            return {"ce_loss": 0.0, "mse_loss": 0.0, "correct": 0, "valid_tokens": 0}

        valid_mask = valid_rows.float()
        valid_targets = targets_flat[valid_rows]
        self.phase_seen_targets.scatter_add_(0, valid_targets, torch.ones_like(valid_targets, dtype=torch.float32))

        # Compute softmax & cross-entropy in float32 cleanly across FP32/BF16/FP16
        probs = F.softmax(logits_flat.float(), dim=-1)
        tgt_probs = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        ce_loss = float((-torch.log(tgt_probs[valid_rows].clamp(min=1e-30))).mean().item())
        correct = int(((probs.argmax(dim=-1) == targets_flat) & valid_rows).sum().item())

        target_sal = self.token_salience[targets_flat]
        input_sal = self.token_salience[inputs_flat]
        step_salience = (0.65 * target_sal + 0.35 * input_sal).clamp(0.70, 1.80)

        # Zero-allocation in-place residual error: probs <- (y_onehot - probs) * mask * salience
        probs.neg_()
        probs.scatter_add_(1, targets_flat.unsqueeze(1), valid_mask.unsqueeze(1))
        if n_valid < float(n_total):
            probs.mul_(valid_mask.unsqueeze(-1))
        mse_loss = float((probs.norm().pow(2) * 0.5).item() / n_valid)
        probs.mul_(step_salience.unsqueeze(-1))
        salient_err = probs if probs.dtype == self.weight_dtype else probs.to(self.weight_dtype)

        maturity_gain = 2.0 if self.current_epoch <= 2 else min(280.0, 32.0 * float(self.current_epoch - 1))
        depth_dfa_scale = ((4.0 / float(max(self.num_layers, 4))) ** 0.25) if self.num_layers > 4 else 1.0

        # Project vocab error ONCE into highway space [N, d_last]
        head_w_active = self.head_w[:, :d_last]
        e_highway = self.factorized_dfa.project_vocab_to_highway(salient_err, head_w_active)

        # Update Attention Q/K/V springs via compact [D, D] highway adapter
        self._update_attention_springs(e_highway, d_eff=d_eff, n_total=float(n_total))

        # Update Hidden Block-Sparse Layers + Factorized Highway Adapters
        for l_idx, layer in enumerate(self.spring_layers):
            d_act = layer.d_active
            delta_l = self.factorized_dfa.broadcast_layer_error(
                layer_idx=l_idx,
                e_highway=e_highway,
                d_act=d_act,
                depth_scale=depth_dfa_scale,
            )
            stiff_mult, stiff_ceil = self.stiff_schedule[l_idx]
            k_blk = min(self.layer_top_k_blocks[l_idx], max(1, d_act // self.block_size))
            batch_active_block_union_update(
                layer=layer,
                projected_error=delta_l,
                n_valid=n_valid,
                maturity_stiff_gain=maturity_gain,
                block_size=self.block_size,
                k_blk=k_blk,
                stiff_mult=stiff_mult,
                stiff_ceil=stiff_ceil,
                scratch=self._kernel_scratch,
                compensator=self.compensator,
                comp_key=f"layer_{l_idx}",
            )

            if self.rejuvenated_slots[l_idx]:
                dorm_idx = torch.tensor(self.rejuvenated_slots[l_idx], dtype=torch.long, device=self.model_device)
                layer.w[dorm_idx, :] = 0.0
                layer.w0[dorm_idx, :] = 0.0
                layer.k[dorm_idx, :] = 0.0
            if self.armor_locked_masks[l_idx][:d_act].any() and l_idx < len(self.layer_w0_bedrock):
                locked = self.armor_locked_masks[l_idx][:d_act]
                bed = self.layer_w0_bedrock[l_idx][:d_act][locked, :d_eff]
                layer.w0[:d_act][locked, :d_eff].copy_(bed)
                layer.w[:d_act][locked, :d_eff].copy_(bed)

            if layer._last_a is not None:
                unlocked_gate = (~self.armor_locked_masks[l_idx][:d_act]).float().unsqueeze(1)
                self.factorized_dfa.adapt_layer(
                    layer_idx=l_idx,
                    a_act=layer._last_a[:, :d_act],
                    e_highway=e_highway,
                    n_valid=n_valid,
                    unlocked_row_gate=unlocked_gate,
                )

        # In-place batch-averaged output head update
        scale_factor = float(n_total) / max(n_valid, 1.0)
        salient_err.mul_(scale_factor)
        driving_head = torch.matmul(salient_err.t(), a_head).div_(float(max(n_total, 1)))
        cfg = self.config
        fused_spring_update_inplace(
            w=self.head_w[:, :d_last],
            w0=self.head_w0[:, :d_last],
            k=self.head_k[:, :d_last],
            driving_signal=driving_head,
            eta_w=cfg.lr_head_w,
            eta_0=cfg.lr_head_0,
            gamma=cfg.gamma_head,
            beta_k_eff=cfg.beta_k_head,
            stiff_ceil=1.0,
            active_row_gate=None,
            cubic_shield=False,
            scratch=self._kernel_scratch,
            compensator=self.compensator,
            comp_key="output_head",
        )

        self._reset_step_cache()
        return {
            "ce_loss": ce_loss,
            "mse_loss": mse_loss,
            "correct": correct,
            "valid_tokens": int(n_valid),
        }

    @torch.no_grad()
    def step_neurogenesis(self, mean_ce_loss: float | None = None) -> dict:
        """Frustration-driven waking of `delta_d` dormant cells with deterministic on-demand rows."""
        if self.frozen or self.delta_d <= 0:
            return {"woke_total": 0, "dims": self.get_allocated_dimensions()}

        cfg = self.config
        woke_total = 0
        old_max_d = max(layer.d_active for layer in self.spring_layers)
        bound_d = 1.0 / math.sqrt(float(self.d_max))

        for l_idx, layer in enumerate(self.spring_layers):
            if layer.d_active >= self.d_max:
                continue
            if (self.current_epoch - layer.last_growth_epoch) < cfg.neurogenesis_cooldown:
                continue

            d_act = layer.d_active
            n_blocks = max(1, d_act // self.block_size)
            ep_wins = self.epoch_block_win_counts[l_idx][:n_blocks]
            sat_ratio = float((ep_wins > 0).float().mean().item()) if ep_wins.numel() > 0 else 0.0
            mean_k = float(layer.k[:d_act, :d_act].float().mean().item())
            err_val = float(mean_ce_loss) if mean_ce_loss is not None else 1.0

            triggered = (
                (err_val >= cfg.neurogenesis_err_thresh)
                and (
                    (sat_ratio >= cfg.neurogenesis_sat_thresh)
                    or (mean_k >= cfg.neurogenesis_stiff_thresh)
                )
            )
            if triggered:
                woke = layer.grow_active(self.delta_d, salt=self.current_epoch + 1)
                if woke > 0:
                    layer.last_growth_epoch = self.current_epoch
                    woke_total += woke

        new_max_d = max(layer.d_active for layer in self.spring_layers)
        if new_max_d > old_max_d:
            cols_w = _deterministic_uniform_rows(
                self.vocab_size,
                new_max_d - old_max_d,
                bound_d,
                seed=cfg.seed + 999000 + old_max_d,
                device=self.model_device,
                dtype=self.weight_dtype,
            )
            self.head_w[:, old_max_d:new_max_d].copy_(cols_w)
            self.head_w0[:, old_max_d:new_max_d].copy_(cols_w)
            self.head_k[:, old_max_d:new_max_d].zero_()

        dims = self.get_allocated_dimensions()
        if woke_total > 0:
            self.growth_log.append({
                "epoch": self.current_epoch,
                "woke_total": woke_total,
                "dims": dims,
            })
        return {"woke_total": woke_total, "dims": dims}

    @torch.no_grad()
    def end_epoch(self, epoch: int | None = None, mean_ce_loss: float | None = None) -> dict:
        self.current_epoch = int(epoch) if epoch is not None else (self.current_epoch + 1)
        growth_info = self.step_neurogenesis(mean_ce_loss=mean_ce_loss)
        for l_idx, layer in enumerate(self.spring_layers):
            self._last_epoch_block_wins[l_idx].copy_(self.epoch_block_win_counts[l_idx])
            self.epoch_block_win_counts[l_idx].zero_()
            layer.epoch_win_counts.zero_()
        return growth_info

    @torch.no_grad()
    def sleep_consolidate(
        self,
        phase_id: int = 1,
        merge_cosine_threshold: float | None = None,
        prune_ratio: float | None = None,
    ) -> dict:
        """
        Dual-Stage Offline Sleep Consolidation, Low-Utility Pruning & Armor Locking:
          1. High-Cosine Synaptic Fusion + Low-Utility Pruning -> `rejuvenated_slots`
          2. Snap `w <- w0` & lock consolidated rows (`k in [lock_min, lock_max]`)
        """
        if self.frozen:
            raise RuntimeError("Cannot sleep-consolidate a frozen model.")

        cfg = self.config
        m_thresh = cfg.sleep_merge_threshold if merge_cosine_threshold is None else float(merge_cosine_threshold)
        p_ratio = cfg.sleep_prune_ratio if prune_ratio is None else float(prune_ratio)
        d_eff, _ = self._effective_dims()

        while len(self.layer_w0_bedrock) < self.num_layers:
            self.layer_w0_bedrock.append(
                torch.zeros((self.d_max, self.d_max), device=self.model_device, dtype=self.weight_dtype)
            )
        if self.head_w0_bedrock is None:
            self.head_w0_bedrock = torch.zeros(
                (self.vocab_size, self.d_max), device=self.model_device, dtype=self.weight_dtype
            )

        merged_per_layer: list[int] = []
        pruned_per_layer: list[int] = []
        locked_per_layer: list[int] = []

        for l_idx, layer in enumerate(self.spring_layers):
            d_act = layer.d_active
            wins = layer.win_counts[:d_act].float()
            unlocked = ~self.armor_locked_masks[l_idx][:d_act]
            cand = [j for j in range(d_act) if unlocked[j] and (j not in self.rejuvenated_slots[l_idx])]

            for j in cand:
                layer.w0[j, :d_eff].copy_(layer.w[j, :d_eff])
                layer.w0[j, d_eff:] = 0.0

            w0_act = layer.w0[:d_act, :d_eff].float()
            row_norms = w0_act.norm(p=2, dim=-1)
            plastic_idx = [j for j in cand if row_norms[j] > 1e-7]

            util_ranked = []
            for j in plastic_idx:
                r_norm = float(row_norms[j].item())
                k_mean = float(layer.k[j, :d_eff].float().mean().item())
                util = r_norm * (k_mean + 0.05) * math.log(2.0 + float(wins[j].item()))
                util_ranked.append((util, j))
            util_ranked.sort(key=lambda x: x[0])

            merged = 0
            merge_pool = [j for _, j in util_ranked[:12]] if len(util_ranked) >= 12 else plastic_idx
            if len(merge_pool) >= 4:
                p_t = torch.tensor(merge_pool, dtype=torch.long, device=self.model_device)
                w0_sub = F.normalize(w0_act[p_t], p=2, dim=-1)
                cos_mat = torch.matmul(w0_sub, w0_sub.t())
                cos_mat.fill_diagonal_(-1.0)
                tri_u, tri_v = torch.triu_indices(len(merge_pool), len(merge_pool), offset=1, device=self.model_device)
                scores = cos_mat[tri_u, tri_v]
                used: set[int] = set()
                for p_idx in torch.argsort(scores, descending=True).tolist():
                    if merged >= 3:
                        break
                    if float(scores[p_idx].item()) < min(m_thresh, 0.15):
                        break
                    u = merge_pool[int(tri_u[p_idx].item())]
                    v = merge_pool[int(tri_v[p_idx].item())]
                    if u in used or v in used:
                        continue
                    u_norm = float(row_norms[u].item())
                    fused = 0.70 * w0_act[u] + 0.30 * w0_act[v]
                    f_norm = float(fused.norm(p=2).item())
                    if f_norm > 1e-7:
                        scaled = (fused * (u_norm / f_norm)).to(layer.w0.dtype)
                        layer.w0[u, :d_eff].copy_(scaled)
                        layer.w[u, :d_eff].copy_(scaled)
                    layer.w[v, :].zero_()
                    layer.w0[v, :].zero_()
                    layer.k[v, :].zero_()
                    layer.win_counts[v] = 0.0
                    self.rejuvenated_slots[l_idx].append(v)
                    used.update((u, v))
                    merged += 1

            # Stage 2: Prune lowest-utility remaining plastic slots into dormant reserve
            rem = [j for _, j in util_ranked if j not in set(self.rejuvenated_slots[l_idx])]
            n_prune = min(len(rem), int(round(len(rem) * p_ratio))) if len(rem) >= 4 else 0
            for v in rem[:n_prune]:
                layer.w[v, :].zero_()
                layer.w0[v, :].zero_()
                layer.k[v, :].zero_()
                layer.win_counts[v] = 0.0
                self.rejuvenated_slots[l_idx].append(v)

            self.total_recycled_cells_per_layer[l_idx] += merged + n_prune

            # Stage 3: Snapshot bedrock & lock surviving phase cells
            active_surv = [j for j in cand if j not in set(self.rejuvenated_slots[l_idx])]
            if active_surv:
                s_idx = torch.tensor(active_surv, dtype=torch.long, device=self.model_device)
                self.layer_w0_bedrock[l_idx][s_idx, :d_eff].copy_(layer.w0[s_idx, :d_eff])
                k_lock = torch.linspace(
                    cfg.sleep_lock_min_k,
                    cfg.sleep_lock_max_k,
                    steps=len(active_surv),
                    device=self.model_device,
                    dtype=layer.k.dtype,
                ).unsqueeze(-1)
                layer.k[s_idx, :d_eff].copy_(k_lock.expand(-1, d_eff))
                self.armor_locked_masks[l_idx][s_idx] = True

            merged_per_layer.append(merged)
            pruned_per_layer.append(n_prune)
            locked_per_layer.append(len(active_surv))

        seen_mask = self.phase_seen_targets > 0
        if seen_mask.any():
            self.head_w0[seen_mask, :d_eff].lerp_(self.head_w[seen_mask, :d_eff], 0.50)
            self.head_w[seen_mask, :d_eff].copy_(self.head_w0[seen_mask, :d_eff])
            self.head_k[seen_mask, :d_eff].clamp_(min=cfg.sleep_lock_min_k, max=cfg.sleep_lock_max_k)
            self.head_armor_mask[seen_mask] = True
            self.head_w0_bedrock[seen_mask, :d_eff].copy_(self.head_w0[seen_mask, :d_eff])
            self.phase_seen_targets.zero_()

        record = {
            "phase_id": int(phase_id),
            "merged_per_layer": merged_per_layer,
            "pruned_per_layer": pruned_per_layer,
            "recycled_total": sum(merged_per_layer) + sum(pruned_per_layer),
            "locked_per_layer": locked_per_layer,
            "active_dims": self.get_active_dimensions(),
        }
        self.sleep_logs.append(record)
        return record

    @torch.no_grad()
    def align_with_reward_modulation(
        self,
        seq_ids: torch.Tensor,
        prefix_lens: torch.Tensor | list[int] | int = 1,
        rewards: torch.Tensor | list[float] | float = 0.0,
        pad_id: int = 0,
        gamma_elig: float = 0.96,
        eta_melt: float = 0.82,
        eta_lock: float = 0.25,
        eta_head_w: float = 5.0,
        eta_head_0: float = 0.65,
    ) -> dict[str, float]:
        """Critic-Free Three-Factor Reward-Modulated Spring Alignment on Continuation Decisions."""
        if self.frozen:
            raise RuntimeError("Model is frozen; post-training alignment is disabled.")

        seq_ids = seq_ids.to(self.model_device)
        in_ids = seq_ids[:, :-1]
        tgt_ids = seq_ids[:, 1:]
        bsz, seq_len = in_ids.shape
        d_eff, d_last = self._effective_dims()

        if isinstance(rewards, (int, float)):
            r_t = torch.full((bsz,), float(rewards), device=self.model_device, dtype=torch.float32)
        elif isinstance(rewards, torch.Tensor):
            r_t = rewards.to(device=self.model_device, dtype=torch.float32)
        else:
            r_t = torch.tensor(rewards, device=self.model_device, dtype=torch.float32)

        r_adv = (r_t - self.reward_baseline).clamp(-1.5, 1.5)
        self.reward_baseline = max(-0.30, min(0.30, 0.90 * self.reward_baseline + 0.10 * float(r_t.mean().item())))

        if isinstance(prefix_lens, int):
            p_list = [prefix_lens] * bsz
        elif isinstance(prefix_lens, torch.Tensor):
            p_list = [int(x) for x in prefix_lens.tolist()]
        else:
            p_list = [int(x) for x in prefix_lens]

        horizon_w = torch.zeros((bsz, seq_len), device=self.model_device, dtype=torch.float32)
        for b in range(bsz):
            p = max(1, min(p_list[b], seq_len))
            t_idxs = torch.nonzero(tgt_ids[b] != pad_id, as_tuple=False).squeeze(-1)
            dec = [int(t.item()) for t in t_idxs if int(t.item()) >= (p - 1)]
            if dec:
                t_last = dec[-1]
                for t in dec:
                    sal = float(self.token_salience[int(tgt_ids[b, t].item())].item())
                    horizon_w[b, t] = (gamma_elig ** float(t_last - t)) * max(sal, 0.70)

        w_flat = horizon_w.reshape(-1)
        valid = (w_flat > 0.0) & (tgt_ids.reshape(-1) != pad_id)
        n_dec = int(valid.sum().item())
        if n_dec == 0:
            self._reset_step_cache()
            return {"valid_decisions": 0, "ce_loss": 0.0}

        ctx = self._build_vectorized_context(in_ids, d_eff)
        logits_flat, a_head = self._forward_hidden_and_head(ctx.reshape(bsz * seq_len, d_eff), d_last=d_last)
        probs = F.softmax(logits_flat.float(), dim=-1)
        tgt_flat = tgt_ids.reshape(-1)
        p_chosen = probs.gather(1, tgt_flat.unsqueeze(1)).squeeze(1)
        ce_loss = float((-torch.log(p_chosen[valid].clamp(min=1e-30))).mean().item())

        r_flat = r_adv.unsqueeze(1).expand(bsz, seq_len).reshape(-1) * w_flat
        pos_scale = torch.where((r_flat > 0.0) & valid, r_flat, torch.zeros_like(r_flat)).unsqueeze(-1)
        neg_scale = torch.where((r_flat < 0.0) & valid, -r_flat, torch.zeros_like(r_flat)).unsqueeze(-1)

        y_onehot = F.one_hot(tgt_flat, num_classes=self.vocab_size).float()
        err_total = ((y_onehot - probs) * pos_scale) - ((y_onehot - probs) * neg_scale * 1.15)
        denom_h = float(max(n_dec, 1))
        w_h, w0_h, k_h = self.head_w[:, :d_last], self.head_w0[:, :d_last], self.head_k[:, :d_last]

        # Selective melting on rejected rows
        neg_active = r_adv < 0.0
        if neg_active.any():
            mean_r_neg = float((-r_adv[neg_active]).mean().item())
            elig_neg = torch.matmul((y_onehot * neg_scale * p_chosen.clamp(min=0.10).unsqueeze(-1)).t(), a_head) / denom_h
            max_e = max(float(elig_neg.abs().max().item()), 1e-6)
            norm_neg = (elig_neg.abs() / max_e).clamp(0.0, 1.0)
            k_h.mul_(1.0 - (norm_neg > 0.05).to(k_h.dtype) * (eta_melt * mean_r_neg * norm_neg).clamp(0.0, 0.92).to(k_h.dtype))

        mean_a2 = max(float(a_head.float().pow(2).sum(dim=-1).mean().item()), 1.0)
        drv_head = (torch.matmul(err_total.t(), a_head.float()) / (denom_h * mean_a2)).to(w_h.dtype)
        fused_spring_update_inplace(
            w=w_h,
            w0=w0_h,
            k=k_h,
            driving_signal=drv_head,
            eta_w=eta_head_w,
            eta_0=eta_head_0,
            gamma=self.config.gamma_head,
            beta_k_eff=0.0,
            stiff_ceil=0.92,
            cubic_shield=False,
            scratch=self._kernel_scratch,
            compensator=self.compensator,
            comp_key="output_head",
        )

        pos_active = r_adv > 0.0
        if pos_active.any():
            mean_r_pos = float(r_adv[pos_active].mean().item())
            elig_pos = torch.matmul((y_onehot * pos_scale).t(), a_head.float()) / denom_h
            max_ep = max(float(elig_pos.abs().max().item()), 1e-6)
            norm_pos = (elig_pos.abs() / max_ep).clamp(0.0, 1.0)
            k_h.add_((eta_lock * mean_r_pos * norm_pos).to(k_h.dtype) * (1.0 - k_h)).clamp_(0.0, 0.92)

        self._reset_step_cache()
        stats = {"valid_decisions": n_dec, "ce_loss": ce_loss, "reward_baseline": self.reward_baseline}
        self.alignment_logs.append(stats)
        return stats

    @torch.no_grad()
    def init_inference_cache(
        self,
        batch_size: int = 1,
        max_seq_len: int | None = None,
    ) -> SpringInferenceCache:
        """Pre-allocates an O(1) stateful cache matching current effective dimensions & dtype."""
        d_eff, d_last = self._effective_dims()
        req_len = int(max_seq_len if max_seq_len is not None else self.max_seq_len)
        cache = SpringInferenceCache(
            batch_size=batch_size,
            max_seq_len=min(req_len, self.max_seq_len),
            d_max=d_eff,
            context_dim=d_eff,
            device=self.model_device,
            dtype=self.weight_dtype,
        )
        cache._cached_d_eff = d_eff
        cache._cached_d_last = d_last
        cache._cached_head_w = self.head_w0 if self.frozen else self.head_w[:, :d_last]
        return cache

    @torch.no_grad()
    def forward_step_cached(
        self,
        token_id: int,
        cache: SpringInferenceCache,
        compute_logits: bool = True,
    ) -> torch.Tensor | None:
        """
        O(1) per-token incremental decoding mirroring `_build_vectorized_context`
        & `_forward_hidden_and_head` bit-for-bit across prompt and generation steps.
        """
        t = cache.step
        if t >= cache.max_seq_len:
            raise RuntimeError(f"Sequence step {t} exceeds cache.max_seq_len ({cache.max_seq_len}).")

        d_eff = getattr(cache, "_cached_d_eff", None)
        d_last = getattr(cache, "_cached_d_last", None)
        if d_eff is None or d_last is None:
            d_eff, d_last = self._effective_dims()

        tok_emb = self.token_embedding[token_id, :d_eff].unsqueeze(0)  # [1, d_eff]
        pos_emb = self.pos_embedding[t, :d_eff].unsqueeze(0)          # [1, d_eff]
        x_t = tok_emb + pos_emb

        q_w_sub, k_w_sub, v_w_sub = self._qkv_weights(d_eff)
        k_t = F.linear(x_t, k_w_sub)
        v_t = self._project_v_sparse(x_t, v_w_sub, store_cache=False)

        cache.k_cache[:, t, :d_eff].copy_(k_t)
        cache.v_cache[:, t, :d_eff].copy_(v_t)
        sal_t = float(self.token_salience[token_id].item())
        cache.sal_cache[:, t].fill_(sal_t)
        cache.record_token_ngrams(token_id)

        a_cap = float(self.capacitor_alpha[token_id].item())
        if a_cap != 0.0:
            cache.c_state.mul_(1.0 - a_cap).add_(tok_emb, alpha=a_cap)

        x_pos_t = tok_emb + self.alibi_pos_encoding[t, :d_eff].unsqueeze(0)
        cache.alibi_u.mul_(0.68).add_(x_pos_t)
        cache.alibi_denom_base = cache.alibi_denom_base * 0.68 + 1.0
        if t == 0:
            cache.x_pos_0.copy_(x_pos_t)
        elif t == 1:
            cache.x_pos_1.copy_(x_pos_t)

        logits_out = None
        if compute_logits:
            k_len = t + 1
            q_t = F.linear(x_t, q_w_sub)  # [1, d_eff]
            k_pre = cache.k_cache[:, :k_len, :d_eff]
            v_pre = cache.v_cache[:, :k_len, :d_eff]
            sal_pre = cache.sal_cache[:, :k_len].unsqueeze(1)  # [1, 1, k_len]

            scale = 1.0 / math.sqrt(float(d_eff))
            scores = torch.bmm(q_t.unsqueeze(1), k_pre.transpose(1, 2)) * scale  # [1, 1, k_len]
            scores = scores + (1.35 * (sal_pre - 1.0)).to(scores.dtype)
            scores = self._apply_dual_horizon_topk(scores, sal_pre, causal_mask=None)
            attn_w = F.softmax(scores.float(), dim=-1).to(v_pre.dtype)
            attended = torch.bmm(attn_w, v_pre).squeeze(1) + x_t

            if t == 0:
                alibi_num = cache.alibi_u + 0.30 * cache.x_pos_0
                denom = cache.alibi_denom_base + 0.30
            else:
                alibi_num = cache.alibi_u + 0.30 * cache.x_pos_0 + 0.20 * cache.x_pos_1
                denom = cache.alibi_denom_base + 0.50
            alibi_ctx = alibi_num / max(denom, 1e-6)
            cap_ctx = F.linear(cache.c_state, self.cap_proj[:d_eff, :d_eff])
            lag1 = F.linear(cache.prev_emb_1, self.lag1_proj[:d_eff, :d_eff]) if t >= 1 else torch.zeros_like(tok_emb)
            lag2 = F.linear(cache.prev_emb_2, self.lag2_proj[:d_eff, :d_eff]) if t >= 2 else torch.zeros_like(tok_emb)

            a_ctx = attended + alibi_ctx + 1.65 * cap_ctx + lag1 + lag2
            cached_hw = getattr(cache, "_cached_head_w", None)
            logits_step, _ = self._forward_hidden_and_head(
                a_ctx, d_last=d_last, in_fast_gen=True, cached_head_w0=cached_hw
            )
            logits_out = logits_step[0]

        cache.prev_emb_2.copy_(cache.prev_emb_1)
        cache.prev_emb_1.copy_(tok_emb)
        cache.step += 1
        return logits_out

    def _apply_refractory_gate_logits(
        self,
        logits: torch.Tensor,
        seq: list[int],
        cache: SpringInferenceCache | None = None,
    ) -> torch.Tensor:
        if not seq:
            return logits
        prev1 = seq[-1]
        if 0 <= prev1 < self.vocab_size:
            logits[prev1] = float("-inf")

        if len(seq) >= 3:
            prev2, prev3 = seq[-2], seq[-3]
            if cache is not None:
                bigram_counts = cache.bigram_next.get(prev1, {})
                trigram_counts = cache.trigram_next.get((prev2, prev1), {})
                fourgram_counts = cache.fourgram_next.get((prev3, prev2, prev1), {})
            else:
                bigram_counts = {}
                trigram_counts = {}
                fourgram_counts = {}
                for i in range(len(seq) - 1):
                    if seq[i] == prev1:
                        nxt = seq[i + 1]
                        bigram_counts[nxt] = bigram_counts.get(nxt, 0) + 1
                for i in range(len(seq) - 2):
                    if seq[i] == prev2 and seq[i + 1] == prev1:
                        nxt = seq[i + 2]
                        trigram_counts[nxt] = trigram_counts.get(nxt, 0) + 1
                for i in range(len(seq) - 3):
                    if seq[i] == prev3 and seq[i + 1] == prev2 and seq[i + 2] == prev1:
                        nxt = seq[i + 3]
                        fourgram_counts[nxt] = fourgram_counts.get(nxt, 0) + 1

            for cand_id, b_cnt in bigram_counts.items():
                if not (0 <= cand_id < self.vocab_size):
                    continue
                sal = float(self.token_salience[cand_id].item())
                t_cnt = trigram_counts.get(cand_id, 0)
                fg_cnt = fourgram_counts.get(cand_id, 0)

                if fg_cnt >= 1:
                    logits[cand_id] -= 2.5 * self.config.refractory_loop_penalty * float(fg_cnt)
                if sal >= 0.95 and (t_cnt >= 2 or b_cnt >= 2):
                    logits[cand_id] -= 2.0 * self.config.refractory_loop_penalty * float(max(t_cnt, b_cnt))
            for cand_id, fg_cnt in fourgram_counts.items():
                if cand_id not in bigram_counts and 0 <= cand_id < self.vocab_size and fg_cnt >= 1:
                    logits[cand_id] -= 2.5 * self.config.refractory_loop_penalty * float(fg_cnt)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 30,
        eos_id: int = 2,
        enable_refractory_gate: bool = True,
        use_fast_cache: bool = True,
    ) -> list[int]:
        out = list(prompt_ids)
        if not out:
            return out

        max_steps = min(self.max_seq_len, len(out) + max_new_tokens)
        steps_to_gen = max(0, min(max_new_tokens, self.max_seq_len - len(out)))
        if steps_to_gen == 0:
            return out

        if use_fast_cache:
            self._in_fast_generate = True
            try:
                cache = self.init_inference_cache(batch_size=1, max_seq_len=max_steps)
                for idx in range(len(out) - 1):
                    self.forward_step_cached(out[idx], cache, compute_logits=False)
                logits = self.forward_step_cached(out[-1], cache, compute_logits=True)
                for step_i in range(steps_to_gen):
                    l_work = logits.float().clone() if logits.dtype != torch.float32 else logits.clone()
                    if enable_refractory_gate:
                        l_work = self._apply_refractory_gate_logits(l_work, out, cache=cache)
                    nxt = int(torch.argmax(l_work).item())
                    out.append(nxt)
                    if nxt == eos_id or step_i + 1 >= steps_to_gen:
                        break
                    logits = self.forward_step_cached(nxt, cache, compute_logits=True)
            finally:
                self._in_fast_generate = False
            return out

        for _ in range(steps_to_gen):
            logits = self.forward(out)
            l_work = logits.float().clone() if logits.dtype != torch.float32 else logits.clone()
            if enable_refractory_gate:
                l_work = self._apply_refractory_gate_logits(l_work, out, cache=None)
            nxt = int(torch.argmax(l_work).item())
            out.append(nxt)
            if nxt == eos_id:
                break
        return out

    @torch.no_grad()
    def count_parameters_breakdown(self) -> dict[str, int | float | list[int] | str]:
        live_buffers = [b for b in self.buffers() if b is not None]
        live_params = sum(int(b.numel()) for b in live_buffers)
        live_bytes = sum(int(b.numel()) * int(b.element_size()) for b in live_buffers)
        kahan_bytes = 0
        if self.compensator is not None:
            for store in (self.compensator._res_w, self.compensator._res_w0, self.compensator._res_k):
                for t in store.values():
                    kahan_bytes += int(t.numel()) * int(t.element_size())
        mb = 1024.0 * 1024.0
        return {
            "frozen": self.frozen,
            "dtype": self.config.dtype,
            "device": str(self.model_device),
            "active_dims": self.get_active_dimensions(),
            "allocated_dims": self.get_allocated_dimensions(),
            "live_buffer_params": live_params,
            "live_buffer_mb": live_bytes / mb,
            "kahan_residual_mb": kahan_bytes / mb,
            "total_live_mb": (live_bytes + kahan_bytes) / mb,
        }

    @torch.no_grad()
    def production_freeze(self, prune_dormant: bool = True) -> dict:
        """
        Consolidates plastic anchors into `W_0`, physically prunes dormant subspace
        when `prune_dormant=True`, and destroys `w`, `k`, `FactorizedHighwayDFA`,
        `attn_dfa`, bedrock, and Kahan residual compensators (`Frozen W0 Only`).
        """
        before = self.count_parameters_breakdown()
        if not self.frozen:
            d_eff, d_last = self._effective_dims()
            # Final Elastic Settlement: fold the converged equilibrium stretch (w - w0)
            # into anchor w0 on unlocked, active rows before tearing down training buffers.
            if not self.attn_armor_locked:
                for p in ("q", "k", "v"):
                    w_t = getattr(self, f"{p}_w")[:d_eff, :d_eff]
                    w0_t = getattr(self, f"{p}_w0")[:d_eff, :d_eff]
                    k_t = getattr(self, f"{p}_k")[:d_eff, :d_eff]
                    settle_alpha = (0.95 + 0.05 * (k_t.float() / 0.85).clamp(0.0, 1.0)).to(w0_t.dtype)
                    w0_t.add_((w_t - w0_t) * settle_alpha)

            for l_idx, layer in enumerate(self.spring_layers):
                d_act = layer.d_active
                settle_mask = ~self.armor_locked_masks[l_idx][:d_act].clone()
                if self.rejuvenated_slots[l_idx]:
                    settle_mask[self.rejuvenated_slots[l_idx]] = False
                if settle_mask.any():
                    w_s = layer.w[:d_act, :d_eff][settle_mask]
                    w0_s = layer.w0[:d_act, :d_eff][settle_mask]
                    k_s = layer.k[:d_act, :d_eff][settle_mask]
                    settle_alpha = (0.95 + 0.05 * (k_s.float() / 0.85).clamp(0.0, 1.0)).to(layer.w0.dtype)
                    layer.w0[:d_act, :d_eff][settle_mask] = w0_s + (w_s - w0_s) * settle_alpha

            unlocked_vocab = ~self.locked_excl_vocab_mask
            if unlocked_vocab.any():
                hw = self.head_w[unlocked_vocab, :d_last]
                hw0 = self.head_w0[unlocked_vocab, :d_last]
                hk = self.head_k[unlocked_vocab, :d_last]
                settle_alpha = (0.95 + 0.05 * (hk.float() / 0.85).clamp(0.0, 1.0)).to(hw0.dtype)
                self.head_w0[unlocked_vocab, :d_last] = hw0 + (hw - hw0) * settle_alpha

            if prune_dormant:
                # Compact frontend projections & embeddings cleanly to [:d_eff]
                self.register_buffer("token_embedding", self.token_embedding[:, :d_eff].contiguous())
                self.register_buffer("pos_embedding", self.pos_embedding[:, :d_eff].contiguous())
                self.register_buffer("alibi_pos_encoding", self.alibi_pos_encoding[:, :d_eff].contiguous())
                for name in ("cap_proj", "lag1_proj", "lag2_proj", "q_w0", "k_w0", "v_w0"):
                    t = getattr(self, name)
                    self.register_buffer(name, t[:d_eff, :d_eff].contiguous())

                for l_idx, layer in enumerate(self.spring_layers):
                    d_act = layer.d_active
                    layer.register_buffer("w0", layer.w0[:d_act, :d_eff].contiguous())
                    layer._compacted_in_dim = d_eff
                    layer.register_buffer("active_mask", torch.ones(d_act, device=self.model_device, dtype=torch.float32))
                    layer.register_buffer("win_counts", layer.win_counts[:d_act].contiguous())
                    layer.register_buffer("epoch_win_counts", layer.epoch_win_counts[:d_act].contiguous())

                self.register_buffer("head_w0", self.head_w0[:, :d_last].contiguous())

            # Destroy training buffers completely
            for name in ("q_w", "q_k", "k_w", "k_k", "v_w", "v_k", "attn_dfa", "head_w", "head_k", "head_w0_bedrock"):
                self.register_buffer(name, None)
            for layer in self.spring_layers:
                layer.register_buffer("w", None)
                layer.register_buffer("k", None)
            self.factorized_dfa = None
            self.layer_w0_bedrock = []
            self.compensator = None
            self._kernel_scratch = None
            self._reset_step_cache()
            self.frozen = True

        after = self.count_parameters_breakdown()
        saved_pct = (
            (float(before["total_live_mb"]) - float(after["total_live_mb"]))
            / max(float(before["total_live_mb"]), 1e-9)
        ) * 100.0
        return {
            "frozen": True,
            "before_live_mb": before["total_live_mb"],
            "after_live_mb": after["total_live_mb"],
            "before_params": before["live_buffer_params"],
            "after_params": after["live_buffer_params"],
            "memory_saved_pct": saved_pct,
            "compacted_dims": self.get_allocated_dimensions(),
        }

    @torch.no_grad()
    def get_full_checkpoint_state(self) -> dict:
        """
        Exports a complete bitwise-faithful training or inference checkpoint state
        containing all structural tensors, layer win statistics, RNG state,
        and deterministic FP32 Kahan residual buffers.
        """
        state_tensors = {}
        for name, buf in self.named_buffers():
            if buf is not None:
                state_tensors[name] = buf.detach().clone()

        comp_state = {
            "mode": self.compensator.mode if self.compensator is not None else self.config.effective_kahan_mode,
            "_res_w": {},
            "_res_w0": {},
            "_res_k": {},
        }
        if self.compensator is not None:
            for sname in ("_res_w", "_res_w0", "_res_k"):
                src = getattr(self.compensator, sname)
                comp_state[sname] = {k: v.detach().clone() for k, v in src.items()}

        return {
            "format_version": "ssenn_v3_checkpoint_1.0",
            "config": self.config.to_dict(),
            "state_tensors": state_tensors,
            "compensator": comp_state,
            "layer_states": [
                {
                    "d_active": int(layer.d_active),
                    "last_growth_epoch": int(layer.last_growth_epoch),
                    "_last_winner_k": float(layer._last_winner_k),
                    "_compacted_in_dim": layer._compacted_in_dim,
                }
                for layer in self.spring_layers
            ],
            "trackers": {
                "block_win_counts": [t.detach().clone() for t in self.block_win_counts],
                "epoch_block_win_counts": [t.detach().clone() for t in self.epoch_block_win_counts],
                "_last_epoch_block_wins": [t.detach().clone() for t in self._last_epoch_block_wins],
                "block_usage_ema": [t.detach().clone() for t in self.block_usage_ema],
                "armor_locked_masks": [t.detach().clone() for t in self.armor_locked_masks],
                "layer_w0_bedrock": [t.detach().clone() for t in self.layer_w0_bedrock],
            },
            "metadata": {
                "frozen": bool(self.frozen),
                "current_epoch": int(self.current_epoch),
                "reward_baseline": float(self.reward_baseline),
                "attn_armor_locked": bool(self.attn_armor_locked),
                "has_multiword_entities": bool(self.has_multiword_entities),
                "attn_v_topk": int(self.attn_v_topk),
                "rejuvenated_slots": copy.deepcopy(self.rejuvenated_slots),
                "total_recycled_cells_per_layer": list(self.total_recycled_cells_per_layer),
                "current_phase_slots": copy.deepcopy(self.current_phase_slots),
                "growth_log": copy.deepcopy(self.growth_log),
                "sleep_logs": copy.deepcopy(self.sleep_logs),
                "alignment_logs": copy.deepcopy(self.alignment_logs),
                "rejuv_gen_state": self._rejuv_gen.get_state().clone(),
            },
        }

    @torch.no_grad()
    def load_full_checkpoint_state(self, ckpt: dict) -> None:
        """Restores full training/inference state cleanly with exact bitwise fidelity."""
        if ckpt.get("format_version") != "ssenn_v3_checkpoint_1.0":
            raise ValueError(f"Unsupported checkpoint format: {ckpt.get('format_version')}")

        meta = ckpt["metadata"]
        is_ckpt_frozen = bool(meta["frozen"])
        if is_ckpt_frozen and not self.frozen:
            # Ensure skeleton shapes match compacted buffers before copying
            f_eff = int(ckpt["state_tensors"]["token_embedding"].shape[1])
            f_outs = [int(ls["d_active"]) for ls in ckpt["layer_states"]]
            self._alloc_frozen_skeleton(f_eff, f_outs, self.model_device, self.weight_dtype)

        state_tensors: dict[str, torch.Tensor] = ckpt["state_tensors"]
        for name, _ in list(self.named_buffers()):
            if name not in state_tensors:
                # Clear module buffer to None if absent in checkpoint (e.g., frozen)
                parts = name.split(".")
                mod = self
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                mod.register_buffer(parts[-1], None)

        for name, val in state_tensors.items():
            parts = name.split(".")
            mod = self
            for p in parts[:-1]:
                mod = getattr(mod, p)
            attr = parts[-1]
            cur = getattr(mod, attr, None)
            val_moved = val.to(device=self.model_device)
            if cur is not None and cur.shape == val_moved.shape and cur.dtype == val_moved.dtype:
                cur.copy_(val_moved)
            else:
                mod.register_buffer(attr, val_moved.clone())

        if is_ckpt_frozen:
            self.factorized_dfa = None
            self.compensator = None
            self._kernel_scratch = None
        else:
            comp_state = ckpt.get("compensator", {})
            comp_mode = comp_state.get("mode", self.config.effective_kahan_mode)
            if self.compensator is None:
                self.compensator = MixedPrecisionCompensator(mode=comp_mode)
            else:
                self.compensator.mode = comp_mode
            for sname in ("_res_w", "_res_w0", "_res_k"):
                target_dict = getattr(self.compensator, sname)
                target_dict.clear()
                for k, v in comp_state.get(sname, {}).items():
                    target_dict[k] = v.to(device=self.model_device, dtype=torch.float32).clone()

        for layer, ls in zip(self.spring_layers, ckpt["layer_states"]):
            layer.d_active = int(ls["d_active"])
            layer.last_growth_epoch = int(ls["last_growth_epoch"])
            layer._last_winner_k = float(ls["_last_winner_k"])
            layer._compacted_in_dim = ls["_compacted_in_dim"]

        tr = ckpt["trackers"]
        self.block_win_counts = [t.to(device=self.model_device).clone() for t in tr["block_win_counts"]]
        self.epoch_block_win_counts = [t.to(device=self.model_device).clone() for t in tr["epoch_block_win_counts"]]
        self._last_epoch_block_wins = [t.to(device=self.model_device).clone() for t in tr["_last_epoch_block_wins"]]
        self.block_usage_ema = [t.to(device=self.model_device).clone() for t in tr["block_usage_ema"]]
        self.armor_locked_masks = [t.to(device=self.model_device).clone() for t in tr["armor_locked_masks"]]
        self.layer_w0_bedrock = [t.to(device=self.model_device).clone() for t in tr["layer_w0_bedrock"]]

        self.frozen = is_ckpt_frozen
        self.current_epoch = int(meta["current_epoch"])
        self.reward_baseline = float(meta["reward_baseline"])
        self.attn_armor_locked = bool(meta["attn_armor_locked"])
        self.has_multiword_entities = bool(meta["has_multiword_entities"])
        self.attn_v_topk = int(meta["attn_v_topk"])
        self.rejuvenated_slots = copy.deepcopy(meta["rejuvenated_slots"])
        self.total_recycled_cells_per_layer = list(meta["total_recycled_cells_per_layer"])
        self.current_phase_slots = copy.deepcopy(meta["current_phase_slots"])
        self.growth_log = copy.deepcopy(meta["growth_log"])
        self.sleep_logs = copy.deepcopy(meta["sleep_logs"])
        self.alignment_logs = copy.deepcopy(meta["alignment_logs"])
        self._rejuv_gen.set_state(meta["rejuv_gen_state"].cpu().clone())
        self._reset_step_cache()

    @classmethod
    @torch.no_grad()
    def from_checkpoint_state(cls, ckpt: dict) -> "SSENNProductionModel":
        cfg = SSENNConfig.from_dict(ckpt["config"])
        if ckpt["metadata"]["frozen"]:
            f_eff = int(ckpt["state_tensors"]["token_embedding"].shape[1])
            f_outs = [int(ls["d_active"]) for ls in ckpt["layer_states"]]
            model = cls(cfg, _skip_train_alloc=True, _frozen_dims=(f_eff, f_outs))
        else:
            model = cls(cfg)
        model.load_full_checkpoint_state(ckpt)
        return model

    @torch.no_grad()
    def export_frozen_w0_state(self) -> dict:
        """Exports strictly minimal `W_0`-only deployment artifacts (Zero w, k, DFA, Kahan)."""
        if not self.frozen:
            self.production_freeze(prune_dormant=True)

        d_eff, d_last = self._effective_dims()
        w0_tensors = {
            "token_embedding": self.token_embedding.detach().clone(),
            "pos_embedding": self.pos_embedding.detach().clone(),
            "alibi_pos_encoding": self.alibi_pos_encoding.detach().clone(),
            "alibi_causal_weights": self.alibi_causal_weights.detach().clone(),
            "cap_proj": self.cap_proj.detach().clone(),
            "lag1_proj": self.lag1_proj.detach().clone(),
            "lag2_proj": self.lag2_proj.detach().clone(),
            "q_w0": self.q_w0.detach().clone(),
            "k_w0": self.k_w0.detach().clone(),
            "v_w0": self.v_w0.detach().clone(),
            "head_w0": self.head_w0.detach().clone(),
            "token_salience": self.token_salience.detach().clone(),
            "capacitor_alpha": self.capacitor_alpha.detach().clone(),
            "layers_w0": [layer.w0.detach().clone() for layer in self.spring_layers],
        }
        return {
            "format_version": "ssenn_v3_frozen_w0_1.0",
            "config": self.config.to_dict(),
            "d_eff": int(d_eff),
            "d_last": int(d_last),
            "layer_d_active": [int(layer.w0.shape[0]) for layer in self.spring_layers],
            "attn_v_topk": int(self.attn_v_topk),
            "has_multiword_entities": bool(self.has_multiword_entities),
            "w0_tensors": w0_tensors,
        }

    @classmethod
    @torch.no_grad()
    def from_frozen_w0_state(cls, frozen_state: dict) -> "SSENNProductionModel":
        """Instantiates an inference-only `SSENNProductionModel` directly from Frozen W0 state."""
        if frozen_state.get("format_version") != "ssenn_v3_frozen_w0_1.0":
            raise ValueError(f"Unsupported frozen W0 format: {frozen_state.get('format_version')}")
        cfg = SSENNConfig.from_dict(frozen_state["config"])
        d_eff = int(frozen_state["d_eff"])
        layer_outs = [int(x) for x in frozen_state["layer_d_active"]]
        model = cls(cfg, _skip_train_alloc=True, _frozen_dims=(d_eff, layer_outs))
        dev = model.model_device
        w_dt = model.weight_dtype

        wt = frozen_state["w0_tensors"]
        for bname in (
            "token_embedding",
            "pos_embedding",
            "alibi_pos_encoding",
            "alibi_causal_weights",
            "cap_proj",
            "lag1_proj",
            "lag2_proj",
            "q_w0",
            "k_w0",
            "v_w0",
            "head_w0",
        ):
            getattr(model, bname).copy_(wt[bname].to(device=dev, dtype=w_dt))
        model.token_salience.copy_(wt["token_salience"].to(device=dev, dtype=torch.float32))
        model.capacitor_alpha.copy_(wt["capacitor_alpha"].to(device=dev, dtype=torch.float32))
        for layer, lw0 in zip(model.spring_layers, wt["layers_w0"]):
            layer.w0.copy_(lw0.to(device=dev, dtype=w_dt))
        model.attn_v_topk = int(frozen_state["attn_v_topk"])
        model.has_multiword_entities = bool(frozen_state["has_multiword_entities"])
        return model

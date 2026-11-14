"""
models/model_scaled.py
Million-Parameter Scale-Up of Self-Sizing Elastic Neural Networks (SS-ENN v2.0 Core Physics)
with Vectorized Mini-Batching ([Batch, Seq, D_max]) for high-throughput CPU execution.

Implements the 4 Permanent v2.0 Physical Laws:
  1. Law 1 (Clipped Log-Salience Weighting):
     s(v) in [0.70, 1.80] computed from inverse log token frequency so high-frequency
     function/hub words ('the', '.', 'afterward', 'then') do not monopolize springs,
     while content/entity nouns ('rabbit', 'bird', 'squirrel', 'oak') receive high salience.
  2. Law 2 (Salience-Gated EMA Context Capacitor + Dual-Horizon Top-K Causal Attention):
     - Continuous causal capacitor c_t = (1 - alpha_t) c_{t-1} + alpha_t * Embed(x_t)
       where alpha_t is gated by token salience s(x_t)^3, preserving entity identity
       across sentence boundaries and hub transitions ('. afterward', '. then').
     - Dual-Horizon Top-K Causal Attention (Future Leakage = 0.0): splits K_attn into
       (K - 2) local syntactic slots + 2 Salient Anchor Slots tracking the highest
       salience-weighted tokens in the causal prefix.
  3. Law 3 (Winner-Count Normalized Spring & Maturity-Gated Stiffening):
     - Normalizes each cell j's batch update by its actual active winner count N_wins(j)
       rather than total B * T, preventing mini-batch gradient/stiffness dilution.
     - Maturity-gated stiffening schedule (2-epoch warmup followed by progressive
       consolidation) driving final active mean stiffness k_bar into [0.12, 0.35].
  4. Law 4 (Row-Normalized Hebbian Feedback Alignment):
     - Adapts feedback matrices B^(l) locally via Delta B^(l) proportional to a^(l) e^T
       with row-wise L2 normalization to prevent numerical blowup, and destroys B^(l)
       completely upon production_freeze().

Milestone 2.4 Scaling Laws (opt-in via enable_block_sparse=True):
  - Contiguous Block-Sparse Top-K Capsules (block_size=16) with load-balanced selection
    and Active-Slice matrix execution [:d_active, :d_eff].
  - Depth-Scaled Residual Highway (h^(l) = RMSNorm(h^(l-1) + (alpha_res/sqrt(L)) a_block^(l)))
    for stable 12-layer propagation (RMS variance ratio L12/L1 ~ 1.00).
  - Highway-Anchored Deep DFA coupling B^(l) with row-normalized W_head^T + local Hebbian trace.

Milestone 2.5 Post-Training Behavioral Alignment Laws (Reward-Modulated Three-Factor Plasticity):
  - Decision-Horizon & Salience-Filtered Eligibility Trace (tau > T_prefix, gamma_elig^(T-tau) * s(x_tau)).
  - Selective Thermal Stiffness Melting on rejected paths (k_ij *= 1 - eta_melt * |R| * norm(|E^-|))
    and Reward Anchor Locking on preferred paths (w -> w0 consolidation & k_ij up).
  - Supports both scalar rewards R in [-1, +1] with reward baseline and contrastive preference pairs,
    consolidating all behavioral shifts directly into Frozen W_0 (Zero-Overhead Export).

Strictly isolated from v1.0 models; zero autograd / zero backward().
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.sparse_attention import SparseCausalAttention
from engine.dynamic_spring_layer import DynamicSpringLayer
from engine.dfa_router import DFARouter
from engine.spring_layer import SpringLayer
from engine.kernels import (
    SpringScratchBuffer,
    MixedPrecisionCompensator,
    fused_spring_update_inplace,
    indexed_block_activate_highway,
    batch_active_block_union_update,
    FactorizedHighwayDFA,
    SpringInferenceCache,
    normalize_token_sequences,
)


SCALE_PRESETS = {
    "1.5M": {
        "vocab_size": 2048,
        "num_layers": 4,
        "d_init": 64,
        "d_max": 256,
        "delta_d": 32,
        "top_k": 16,
        "attn_top_k": 8,
        "max_seq_len": 64,
    },
    "5M": {
        "vocab_size": 4096,
        "num_layers": 6,
        "d_init": 128,
        "d_max": 512,
        "delta_d": 64,
        "top_k": 32,
        "attn_top_k": 12,
        "max_seq_len": 64,
    },
}


class DualHorizonCausalAttention(SparseCausalAttention):
    """
    Upgraded SparseCausalAttention implementing Law 2: Dual-Horizon Top-K Causal Attention.
    Enforces strict causal masking (M_{t, tau} = -inf for tau > t, Future Leakage = 0.0)
    while partitioning the K_attn slots at each query position t into:
      - (K_attn - 2) Local/Contextual slots (highest raw attention scores)
      - 2 Salient Anchor slots (highest salience-weighted scores across the prefix 0..t)
    """

    def __init__(self, *args, vectorized_topk: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectorized_topk = vectorized_topk
        self._fast_d_eff: int | None = None
        self.register_buffer("token_salience", torch.ones(self.vocab_size))

    @torch.no_grad()
    def set_token_salience(self, salience_vec: torch.Tensor) -> None:
        self.token_salience.copy_(salience_vec[: self.vocab_size])

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        tok_emb = self.token_embedding[input_ids]  # [B, T, D]
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0)  # [1, T, D]
        x = tok_emb + pos_emb

        x_flat = x.reshape(bsz * seq_len, self.d_model)
        q = self.q_proj(x_flat).reshape(bsz, seq_len, self.d_model)
        k = self.k_proj(x_flat).reshape(bsz, seq_len, self.d_model)
        v = self.v_proj(x_flat).reshape(bsz, seq_len, self.d_model)

        scale = 1.0 / math.sqrt(self.d_model)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, T, T]

        # Key token salience s(x_tau) in [0.70, 1.80]: [B, 1, T]
        key_salience = self.token_salience[input_ids].unsqueeze(1)
        # Boost salient entity keys in attention scores without violating causality
        scores = scores + 1.35 * (key_salience - 1.0)

        # Strict Causal Mask: M_{t, tau} = -inf for all tau > t
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

        # Dual-Horizon Top-K Selection per query step t
        if self.attn_top_k is not None and self.attn_top_k < seq_len:
            salient_rank_scores = (4.0 * key_salience + 0.15 * scores).masked_fill(
                ~causal_mask.unsqueeze(0), float("-inf")
            )
            if self.vectorized_topk:
                k_anchor = min(2, seq_len)
                k_local = min(max(1, self.attn_top_k - k_anchor), seq_len)
                sparse_scores = torch.full_like(scores, float("-inf"))
                _, anchor_idx = torch.topk(salient_rank_scores, k=k_anchor, dim=-1)
                sparse_scores.scatter_(-1, anchor_idx, scores.gather(-1, anchor_idx))
                _, local_idx = torch.topk(scores, k=k_local, dim=-1)
                sparse_scores.scatter_(-1, local_idx, scores.gather(-1, local_idx))
                scores = sparse_scores
            else:
                sparse_scores = torch.full_like(scores, float("-inf"))
                k_local = max(1, self.attn_top_k - 2)
                k_anchor = 2

                for t in range(seq_len):
                    if t + 1 <= self.attn_top_k:
                        sparse_scores[:, t, : t + 1] = scores[:, t, : t + 1]
                    else:
                        valid_scores = scores[:, t, : t + 1]
                        valid_sal = salient_rank_scores[:, t, : t + 1]

                        # 1. Pick 2 Salient Anchor slots across prefix 0..t
                        _, anchor_idx = torch.topk(valid_sal, k=k_anchor, dim=-1)
                        sparse_scores[:, t, : t + 1].scatter_(
                            -1, anchor_idx, valid_scores.gather(-1, anchor_idx)
                        )

                        # 2. Pick (K - 2) Local/Contextual slots
                        _, local_idx = torch.topk(valid_scores, k=k_local, dim=-1)
                        sparse_scores[:, t, : t + 1].scatter_(
                            -1, local_idx, valid_scores.gather(-1, local_idx)
                        )
                scores = sparse_scores

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights * causal_mask.unsqueeze(0).float()
        self._last_attn_weights = attn_weights.detach()

        attended = torch.matmul(attn_weights, v) + x
        return attended


class ScaledSpringNetwork(nn.Module):
    """
    Vectorized Mini-Batch SS-ENN v2.0 Model supporting 1.5M and 5M parameter scales,
    equipped with the 4 Permanent v2.0 Physical Laws.
    """

    def __init__(
        self,
        preset: str = "1.5M",
        vocab_size: int | None = None,
        num_layers: int | None = None,
        d_init: int | None = None,
        d_max: int | None = None,
        delta_d: int | None = None,
        top_k: int | None = None,
        attn_top_k: int | None = None,
        max_seq_len: int = 64,
        eta_w: float = 0.08,
        eta_0: float = 0.025,
        gamma: float = 0.08,
        beta_k: float = 0.02,
        eta_b: float = 0.015,
        enable_neurogenesis: bool = True,
        saturation_threshold: float = 0.85,
        stiffness_threshold: float = 0.08,
        error_threshold: float = 0.02,
        cooldown_epochs: int = 2,
        enable_hierarchical_compositional: bool = True,
        enable_block_sparse: bool = False,
        block_size: int = 16,
        use_depth_scaled_highway: bool | None = None,
        alpha_res: float = 1.35,
        use_highway_anchored_dfa: bool | None = None,
        dfa_anchor_blend: float = 0.65,
        block_balance_strength: float = 0.68,
        use_role_bridge: bool | None = None,
        use_fused_kernels: bool = False,
        use_factorized_dfa: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        if preset not in SCALE_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from {list(SCALE_PRESETS.keys())}")

        cfg = SCALE_PRESETS[preset]
        self.preset = preset
        self.vocab_size = vocab_size or cfg["vocab_size"]
        self.num_layers = num_layers or cfg["num_layers"]
        self.d_initial = d_init or cfg["d_init"]
        self.d_max = d_max or cfg["d_max"]
        self.delta_d = delta_d or cfg["delta_d"]
        self.top_k = top_k or cfg["top_k"]
        self.attn_top_k = attn_top_k or cfg["attn_top_k"]
        self.max_seq_len = max_seq_len or cfg["max_seq_len"]
        self.eta_b = eta_b
        self.enable_neurogenesis = enable_neurogenesis
        self.enable_hierarchical_compositional = enable_hierarchical_compositional
        self.enable_block_sparse = enable_block_sparse
        self.block_size = max(1, int(block_size))
        self.use_depth_scaled_highway = (
            enable_block_sparse if use_depth_scaled_highway is None else use_depth_scaled_highway
        )
        self.alpha_res = float(alpha_res)
        self.use_highway_anchored_dfa = (
            enable_block_sparse if use_highway_anchored_dfa is None else use_highway_anchored_dfa
        )
        self.dfa_anchor_blend = float(dfa_anchor_blend)
        self.block_balance_strength = float(block_balance_strength)
        self.use_role_bridge = (
            (enable_hierarchical_compositional and (not enable_block_sparse))
            if use_role_bridge is None
            else use_role_bridge
        )
        self.use_fused_kernels = bool(use_fused_kernels)
        self.use_factorized_dfa = bool(use_factorized_dfa)
        self._kernel_scratch = SpringScratchBuffer()
        self._mp_compensator = MixedPrecisionCompensator()
        self.frozen = False
        self.current_epoch = 1
        self._fast_d_eff: int | None = None
        self._in_fast_generate: bool = False

        if self.enable_block_sparse:
            # Align dimensions strictly to integer multiples of capsule block_size
            self.d_max = max(
                self.block_size,
                ((self.d_max + self.block_size - 1) // self.block_size) * self.block_size,
            )
            self.d_initial = max(
                self.block_size,
                min(self.d_max, (self.d_initial // self.block_size) * self.block_size),
            )
            self.delta_d = max(
                self.block_size,
                ((self.delta_d + self.block_size - 1) // self.block_size) * self.block_size,
            )

        # Law 1: Clipped Log-Salience Vector s(v) in [0.70, 1.80]
        self.register_buffer("token_salience", torch.ones(self.vocab_size))
        # Charging gate alpha(v) for the Salience-Gated EMA Context Capacitor
        self.register_buffer("capacitor_alpha", torch.full((self.vocab_size,), 0.08))
        # Milestone 2.3: Unsupervised Recurring Content-Entity Phrase Vocabulary Mask
        self.register_buffer("is_entity_vocab", torch.zeros(self.vocab_size, dtype=torch.bool))
        self.has_multiword_entities: bool = False

        # 1. Law 2: Dual-Horizon Sparse Causal Self-Attention (Future Leakage = 0.0)
        self.sparse_attention = DualHorizonCausalAttention(
            vocab_size=self.vocab_size,
            d_model=self.d_max,
            attn_top_k=self.attn_top_k,
            max_seq_len=self.max_seq_len,
            eta_w=eta_w,
            eta_0=eta_0,
            gamma=gamma,
            beta_k=beta_k,
            vectorized_topk=self.enable_block_sparse,
            seed=seed,
        )

        gen = torch.Generator()
        gen.manual_seed(seed + 333)

        pos_enc = torch.zeros(self.max_seq_len, self.d_max)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_max, 2, dtype=torch.float32) * (-math.log(1000.0) / self.d_max)
        )
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = F.normalize(pos_enc, p=2, dim=-1) * math.sqrt(self.d_max / 5.0)
        self.register_buffer("alibi_pos_encoding", pos_enc)

        causal_tril = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        idx = torch.arange(self.max_seq_len)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0).float()
        alibi_weights = (0.68 ** dist) * causal_tril
        alibi_weights[:, 0] += 0.30 * causal_tril[:, 0]
        alibi_weights[:, 1] += 0.20 * causal_tril[:, 1]
        alibi_weights = alibi_weights / alibi_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("alibi_causal_weights", alibi_weights)

        # Milestone 2.3: Translation-invariant sliding ALiBi causal window (without pos-0/pos-1 bias)
        alibi_sliding = (0.60 ** dist) * causal_tril
        alibi_sliding = alibi_sliding / alibi_sliding.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("alibi_sliding_weights", alibi_sliding)

        # Milestone 2.3 & 2.4: Hierarchical Top-K (element/block) & Stiffness Spectrum across depth
        half_layers = max(1, self.num_layers // 2)
        wide_k = min(self.d_max, int(math.floor(1.5 * self.top_k)))
        n_max_blocks = max(1, self.d_max // self.block_size)
        self.hierarchical_top_ks: list[int] = []
        self.hierarchical_top_k_blocks: list[int] = []
        self.layer_stiff_multipliers: list[float] = []
        self.layer_stiff_ceilings: list[float] = []
        for l_idx in range(self.num_layers):
            if l_idx < half_layers:
                k_l = wide_k
            else:
                denom = max(self.num_layers - half_layers, 1)
                t_pos = float(l_idx - half_layers + 1) / float(denom)
                k_l = int(round(wide_k - t_pos * float(wide_k - self.top_k)))
            k_clamped = max(4, min(self.d_max, k_l))
            self.hierarchical_top_ks.append(k_clamped)
            self.hierarchical_top_k_blocks.append(
                max(1, min(n_max_blocks, int(round(k_clamped / float(self.block_size)))))
            )
            rho_l = float(l_idx) / float(max(self.num_layers - 1, 1))
            self.layer_stiff_multipliers.append(0.45 + 1.15 * rho_l)
            self.layer_stiff_ceilings.append(0.28 + 0.57 * rho_l)

        lag1_proj = torch.randn(self.d_max, self.d_max, generator=gen) / math.sqrt(self.d_max)
        lag2_proj = torch.randn(self.d_max, self.d_max, generator=gen) / math.sqrt(self.d_max)
        cap_proj = torch.randn(self.d_max, self.d_max, generator=gen) / math.sqrt(self.d_max)
        self.register_buffer("lag1_proj", lag1_proj)
        self.register_buffer("lag2_proj", lag2_proj)
        self.register_buffer("cap_proj", cap_proj)

        # 2. Dynamic Spring Layers with Dormant Reserve (D_init -> D_max)
        self.spring_layers = nn.ModuleList([
            DynamicSpringLayer(
                d_max=self.d_max,
                d_initial=self.d_initial,
                delta_d=self.delta_d,
                top_k=self.top_k,
                eta_w=eta_w,
                eta_0=eta_0,
                gamma=gamma,
                beta_k=beta_k,
                use_residual=True,
                saturation_threshold=saturation_threshold,
                stiffness_threshold=stiffness_threshold,
                error_threshold=error_threshold,
                cooldown_epochs=cooldown_epochs,
                layer_id=l_idx + 1,
                seed=seed + 10 * (l_idx + 1),
            )
            for l_idx in range(self.num_layers)
        ])
        self.hidden_layers = self.spring_layers
        if self.enable_block_sparse:
            for l_idx, layer in enumerate(self.spring_layers):
                layer.top_k = min(layer.d_active, self.hierarchical_top_k_blocks[l_idx] * self.block_size)
        elif self.enable_hierarchical_compositional:
            for l_idx, layer in enumerate(self.spring_layers):
                layer.top_k = min(layer.d_active, self.hierarchical_top_ks[l_idx])

        # Milestone 2.3: Role-Factored Compositional Springs (Scene vs Entity Start / End)
        self.role_bridge_head = SpringLayer(
            in_features=4 * self.d_max + 1,
            out_features=3,
            top_k=None,
            eta_w=0.25,
            eta_0=0.08,
            gamma=0.02,
            beta_k=0.02,
            activation="identity",
            use_residual=False,
            seed=seed + 888,
        )
        with torch.no_grad():
            self.role_bridge_head.w.zero_()
            self.role_bridge_head.w0.zero_()
            # Default prior strongly favors SCENE (~91%) over Entity Recall (~9%) until bridge activates
            scene_prior = torch.tensor([1.5, -1.5, -1.5])
            self.role_bridge_head.w[:, -1].copy_(scene_prior)
            self.role_bridge_head.w0[:, -1].copy_(scene_prior)
            self.role_bridge_head.k.zero_()

        self.entity_proj = SpringLayer(
            in_features=self.d_max,
            out_features=self.d_max,
            top_k=None,
            eta_w=0.05,
            eta_0=0.02,
            gamma=0.05,
            beta_k=0.02,
            activation="identity",
            use_residual=False,
            seed=seed + 889,
        )
        with torch.no_grad():
            eye_d = torch.eye(self.d_max)
            self.entity_proj.w.copy_(eye_d)
            self.entity_proj.w0.copy_(eye_d)
            self.entity_proj.k.zero_()

        self._last_layer_raw_rms: list[float] = [0.0 for _ in range(self.num_layers)]
        self._last_layer_normed_rms: list[float] = [0.0 for _ in range(self.num_layers)]
        self.block_win_counts: list[torch.Tensor] = [
            torch.zeros(n_max_blocks, dtype=torch.float32) for _ in range(self.num_layers)
        ]
        self.epoch_block_win_counts: list[torch.Tensor] = [
            torch.zeros(n_max_blocks, dtype=torch.float32) for _ in range(self.num_layers)
        ]
        self._last_epoch_block_wins: list[torch.Tensor] = [
            torch.zeros(n_max_blocks, dtype=torch.float32) for _ in range(self.num_layers)
        ]
        self.block_usage_ema: list[torch.Tensor] = [
            torch.ones(n_max_blocks, dtype=torch.float32) for _ in range(self.num_layers)
        ]
        self._last_block_entropy: list[float] = [0.0 for _ in range(self.num_layers)]
        self._frozen_block_wins: list[torch.Tensor] | None = None
        self._is_training_step: bool = False
        self._last_role_probs: torch.Tensor | None = None
        self._last_ent_logits: torch.Tensor | None = None
        self._last_ent_slots: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        # 3. Law 4 / Phase 3.1: Adaptive Hebbian Feedback Router OR Factorized Highway DFA
        if self.use_factorized_dfa:
            self.dfa_router = None
            self.factorized_dfa = FactorizedHighwayDFA(
                num_layers=self.num_layers,
                d_max=self.d_max,
                vocab_size=self.vocab_size,
                eta_b=eta_b,
                dfa_anchor_blend=self.dfa_anchor_blend,
                seed=seed + 500,
            )
        else:
            self.factorized_dfa = None
            self.dfa_router = DFARouter(
                num_layers=self.num_layers,
                d_model=self.d_max,
                vocab_size=self.vocab_size,
                seed=seed + 500,
            )

        # Output Linear Spring Head (D_max -> V)
        self.output_head = SpringLayer(
            in_features=self.d_max,
            out_features=self.vocab_size,
            top_k=None,
            eta_w=eta_w * 3.5,
            eta_0=eta_0 * 2.5,
            gamma=gamma * 0.25,
            beta_k=beta_k,
            activation="identity",
            use_residual=False,
            seed=seed + 999,
        )
        with torch.no_grad():
            self._head_latent_cols = self.output_head.w.clone()
            self.output_head.w[:, self.d_initial:].zero_()
            self.output_head.w0[:, self.d_initial:].zero_()
            self.output_head.k[:, self.d_initial:].zero_()

        self.growth_log: list[dict] = []
        self.sleep_logs: list[dict] = []
        self.alignment_logs: list[dict] = []
        self.reward_baseline: float = 0.0

        # Milestone 2.2: Dual-Stage Sleep Consolidation, Synaptic Recycling & Armor Locking State
        self.use_cubic_shield: bool = True
        self.continual_mode: bool = False
        self.continual_stream_dim: int = 128
        self.rejuvenated_slots: list[list[int]] = [[] for _ in range(self.num_layers)]
        self.total_recycled_cells_per_layer: list[int] = [0 for _ in range(self.num_layers)]
        self.armor_locked_masks: list[torch.Tensor] = [
            torch.zeros(self.d_max, dtype=torch.bool) for _ in range(self.num_layers)
        ]
        self.head_armor_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.locked_excl_vocab_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.phase_active_vocab_mask = torch.ones(self.vocab_size, dtype=torch.bool)
        self.phase_seen_targets = torch.zeros(self.vocab_size, dtype=torch.float32)
        self.vocab_phase_counts = torch.zeros(self.vocab_size, dtype=torch.float32)
        self.head_w0_bedrock = torch.zeros(self.vocab_size, self.d_max, dtype=torch.float32)
        self.layer_w0_bedrock: list[torch.Tensor] = [
            torch.zeros(self.d_max, self.d_max, dtype=torch.float32) for _ in range(self.num_layers)
        ]
        self.layer_cell_phase_counts: list[torch.Tensor] = [
            torch.zeros(self.d_max, dtype=torch.float32) for _ in range(self.num_layers)
        ]
        self.attn_armor_locked: bool = False
        self._ref_excl_norm: float | None = None
        self._ref_layer_norms: list[float | None] = [None for _ in range(self.num_layers)]
        self._last_phase_rejuvenated: int = 0
        self.current_phase_slots: list[list[int]] = [[] for _ in range(self.num_layers)]
        self.phase_locked_slots: dict[int, list[torch.Tensor]] = {}
        self.phase_compacted_slots: dict[int, list[torch.Tensor]] = {}
        self.phase_resonance_profiles: dict[int, torch.Tensor] = {}
        self.phase_shared_bedrock: dict[int, torch.Tensor] = {}
        self.phase_vocab_masks: dict[int, torch.Tensor] = {}
        self.shared_vocab_tensor: torch.Tensor | None = None
        self._rejuv_gen = torch.Generator()
        self._rejuv_gen.manual_seed(seed + 777)

        if self.use_highway_anchored_dfa:
            self._sync_highway_anchored_dfa(init_only=True)

        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def calibrate_salience_from_tokens(
        self,
        token_sequences,
        pad_id: int = 0,
        bos_id: int | None = 1,
        eos_id: int | None = 2,
    ) -> dict[str, float]:
        """
        Law 1: Clipped Log-Salience Weighting s(v) in [0.70, 1.80].
        Combines Inverse Log Token/Document Frequency with Intra-Document Recurrence
        (Burstiness = P(count(v in doc) >= 2 | v in doc)):
          - High-frequency function/punctuation/hub tokens ('the', '.', 'and', 'then', 'afterward')
            have high document frequency and zero burstiness -> map near 0.70 - 0.85.
          - Recurring thematic/entity tokens ('rabbit', 'bird', 'squirrel', 'kitten', etc.)
            have high IDF and high intra-document burstiness -> map near 1.55 - 1.80.
        Also calibrates the Salience-Gated EMA Capacitor charging rate alpha(v):
          - High-salience thematic entities charge the capacitor strongly (alpha ~ 0.45),
            whereas low-salience hub/setting words barely overwrite it (alpha <= 0.005).
        """
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
                    # Exclude ultra-ubiquitous structural tokens (appearing in >30% of docs) from burstiness
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
            # Inverse log document/token frequency
            inv_log_tf = torch.log((total_tokens + 1.0) / (counts + 1.0))
            idf = torch.log((float(n_docs) + 1.0) / (doc_freq + 1.0))

            # Intra-document recurrence ratio for selective content words (DF < 30% of docs)
            selective_mask = (doc_freq < (0.30 * float(n_docs))).float()
            burst_ratio = (burst_freq / (doc_freq + 1.0)) * selective_mask

            raw_info = (
                0.50 * inv_log_tf
                + 0.40 * idf
                + 1.50 * burst_ratio
                + 2.20 * phrase_burst_score
            )
            min_val = float(raw_info[active_vocab].min().item())
            max_val = float(raw_info[active_vocab].max().item())
            span = max(max_val - min_val, 1e-6)

            norm_sal = (raw_info - min_val) / span  # in [0, 1] for seen tokens
            salience = (0.70 + 1.10 * norm_sal).clamp(0.70, 1.80)
            salience = torch.where(active_vocab, salience, torch.ones_like(salience))
        else:
            salience = torch.ones(self.vocab_size)

        if 0 <= pad_tok < self.vocab_size:
            salience[pad_tok] = 0.0
        if bos_tok is not None and 0 <= bos_tok < self.vocab_size:
            salience[bos_tok] = 0.70
        if eos_tok is not None and 0 <= eos_tok < self.vocab_size:
            salience[eos_tok] = 0.80

        self.token_salience.copy_(salience)
        self.sparse_attention.set_token_salience(salience)

        phrase_mask = phrase_burst_score >= 0.45
        uni_cross_ratio = unigram_cross_burst / (doc_freq + 1e-6)
        single_mask = (uni_cross_ratio >= 0.65) & (doc_freq < (0.25 * float(n_docs))) & (salience >= 1.45)
        self.has_multiword_entities = bool(phrase_mask.any().item())
        entity_mask = phrase_mask | single_mask
        if 0 <= pad_tok < self.vocab_size:
            entity_mask[pad_tok] = False
        if bos_tok is not None and 0 <= bos_tok < self.vocab_size:
            entity_mask[bos_tok] = False
        if eos_tok is not None and 0 <= eos_tok < self.vocab_size:
            entity_mask[eos_tok] = False
        self.is_entity_vocab.copy_(entity_mask)

        # Salience-Gated EMA Capacitor charging rate alpha(v):
        # Discourse entities with high salience (s(v) >= 1.45) strongly latch into c_t (alpha ~ 0.46),
        # while local modifiers (s < 1.45) and hub bridges ('. afterward', '. then') have tiny alpha (<= 0.004)
        # so the protagonist entity is preserved across sentence boundaries.
        rel_sal = ((salience - 0.70) / 1.10).clamp(0.0, 1.0)
        alpha = torch.where(
            salience >= 1.45,
            0.38 + 0.12 * rel_sal,
            0.004 * (rel_sal ** 2),
        )
        if 0 <= pad_tok < self.vocab_size:
            alpha[pad_tok] = 0.0
        if bos_tok is not None and 0 <= bos_tok < self.vocab_size:
            alpha[bos_tok] = 0.0
        self.capacitor_alpha.copy_(alpha)

        seen_sal = salience[active_vocab]
        return {
            "min_salience": float(seen_sal.min().item()),
            "max_salience": float(seen_sal.max().item()),
            "mean_salience": float(seen_sal.mean().item()),
        }

    @torch.no_grad()
    def get_active_dimensions(self) -> list[int]:
        dims = []
        for l_idx, layer in enumerate(self.spring_layers):
            if self.frozen and hasattr(layer, "_compacted_indices") and layer._compacted_indices is not None:
                dims.append(int(layer._compacted_indices.numel()))
            else:
                dims.append(layer.d_active - len(self.rejuvenated_slots[l_idx]))
        return dims

    @torch.no_grad()
    def get_allocated_dimensions(self) -> list[int]:
        return [layer.d_active for layer in self.spring_layers]

    @torch.no_grad()
    def _effective_context_dim(self) -> int:
        if self.continual_mode:
            return self.continual_stream_dim
        return max(layer.d_active for layer in self.spring_layers)

    @torch.no_grad()
    def _final_layer_dim(self) -> int:
        if self.continual_mode:
            return self.continual_stream_dim
        return self.spring_layers[-1].d_active

    @torch.no_grad()
    def enable_fused_kernels(self, enabled: bool = True) -> None:
        """Toggles Phase 3.1 Fused In-Place Spring Kernels & Active-Block Execution."""
        self.use_fused_kernels = bool(enabled)
        if not self.use_fused_kernels:
            self._kernel_scratch = SpringScratchBuffer()
            self._mp_compensator = MixedPrecisionCompensator()

    @torch.no_grad()
    def _sync_highway_anchored_dfa(self, init_only: bool = False) -> None:
        """
        Law 3 (Milestone 2.4): Highway-Anchored Deep Hebbian DFA.
        Aligns feedback matrices B^(l) with the Output Head transpose (W_head^T)
        along active residual highway coordinates while blending layer-specific
        Hebbian specialization (1 - blend) and preserving row L2 normalization.
        """
        if not self.use_highway_anchored_dfa or self.frozen or self.dfa_router is None:
            return
        target_norm = math.sqrt(self.vocab_size / float(self.d_max))
        d_eff = self._effective_context_dim()
        w_head_src = torch.empty(self.vocab_size, self.d_max, device=self.output_head.w.device)
        w_head_src[:, :d_eff].copy_(self.output_head.w[:, :d_eff])
        if d_eff < self.d_max:
            if init_only and self._head_latent_cols is not None:
                w_head_src[:, d_eff:].copy_(self._head_latent_cols[:, d_eff:])
            elif self._head_latent_cols is not None:
                w_head_src[:, d_eff:].copy_(self._head_latent_cols[:, d_eff:])
            else:
                w_head_src[:, d_eff:].zero_()

        anchor_full = w_head_src.t()  # [d_max, V]
        anchor_unit = F.normalize(anchor_full, p=2, dim=-1, eps=1e-6)
        blend = self.dfa_anchor_blend

        for l_idx, layer in enumerate(self.spring_layers):
            b_mat = self.dfa_router.get_matrix(l_idx)
            if b_mat is None:
                continue
            d_act = self.d_max if init_only else layer.d_active
            b_unit = F.normalize(b_mat[:d_act, :], p=2, dim=-1, eps=1e-6)
            mixed = F.normalize(
                blend * anchor_unit[:d_act, :] + (1.0 - blend) * b_unit,
                p=2,
                dim=-1,
                eps=1e-6,
            ) * target_norm
            b_mat[:d_act, :].copy_(mixed)

    @torch.no_grad()
    def count_parameters_breakdown(self) -> dict:
        v = self.vocab_size
        d_max = self.d_max
        d_init = self.d_initial
        d_acts = self.get_active_dimensions()
        d_eff = self._effective_context_dim() if not self.frozen else int(self.spring_layers[0].w0.shape[1])
        d_last = self._final_layer_dim() if not self.frozen else int(self.output_head.w0.shape[1])

        fast_d = self._fast_d_eff if (self.frozen and self._fast_d_eff is not None) else d_max
        emb_train_params = v * d_max + self.max_seq_len * d_max + 3 * (d_max * d_max)
        emb_frozen_params = v * d_max + self.max_seq_len * d_max + 3 * (fast_d * d_max)
        attn_train_params = 3 * 3 * (d_max * d_max) + (d_max * v)
        attn_frozen_params = 3 * (d_max * d_max)

        hidden_max_train = self.num_layers * 3 * (d_max * d_max)
        hidden_init_active = self.num_layers * 3 * (d_init * d_init)
        hidden_curr_active = sum(3 * (d * d_eff) for d in d_acts)
        hidden_frozen_w0 = sum(d * d_eff for d in d_acts)

        dfa_params = (
            self.factorized_dfa.parameter_count()
            if self.factorized_dfa is not None
            else self.num_layers * (d_max * v)
        )
        dfa_init_active = (
            self.num_layers * (d_init * d_init)
            if self.factorized_dfa is not None
            else self.num_layers * d_init * v
        )
        dfa_curr_active = (
            sum(d * d_last for d in d_acts)
            if self.factorized_dfa is not None
            else sum(d * v for d in d_acts)
        )

        head_max_train = 3 * (v * d_max)
        head_init_active = 3 * (v * d_init)
        head_curr_active = 3 * (v * d_last)
        head_frozen_w0 = v * d_last

        role_w0 = ((4 * d_max + 1) * 3 + d_max * d_max) if self.use_role_bridge else 0
        role_train = 3 * role_w0

        total_latent_train_params = (
            emb_train_params + attn_train_params + hidden_max_train + dfa_params + head_max_train + role_train
        )
        total_init_active_params = (
            emb_train_params + attn_train_params + hidden_init_active + dfa_init_active + head_init_active + role_train
        )
        total_curr_active_params = (
            emb_train_params + attn_train_params + hidden_curr_active + dfa_curr_active + head_curr_active + role_train
        )
        total_frozen_prod_params = emb_frozen_params + attn_frozen_params + hidden_frozen_w0 + head_frozen_w0 + role_w0

        live_bytes = sum(b.numel() * b.element_size() for b in self.buffers() if b is not None)
        live_mb = live_bytes / (1024.0 * 1024.0)

        train_peak_mb = (total_latent_train_params * 4.0) / (1024.0 * 1024.0)
        frozen_prod_mb = (total_frozen_prod_params * 4.0) / (1024.0 * 1024.0)
        memory_saved_pct = ((train_peak_mb - frozen_prod_mb) / max(train_peak_mb, 1e-6)) * 100.0

        return {
            "preset": self.preset,
            "vocab_size": v,
            "d_init": d_init,
            "d_max": d_max,
            "active_dims": d_acts,
            "allocated_dims": self.get_allocated_dimensions(),
            "latent_max_params": total_latent_train_params,
            "initial_active_params": total_init_active_params,
            "current_active_params": total_curr_active_params,
            "frozen_w0_params": total_frozen_prod_params,
            "live_buffer_mb": live_mb,
            "train_footprint_mb": train_peak_mb,
            "frozen_footprint_mb": frozen_prod_mb,
            "memory_saved_pct": memory_saved_pct,
        }

    @torch.no_grad()
    def _build_vectorized_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Law 2: Vectorized Dual-Horizon Causal Attention + Salience-Gated EMA Context Capacitor.
        In Block-Sparse Active-Slice mode (or frozen fast inference), computes context directly
        on the active slice [:d_eff] to eliminate wasted dormant-column FLOPs.
        """
        bsz, seq_len = input_ids.shape
        d_eff = self._effective_context_dim()
        use_active_slice = (self.enable_block_sparse or (self._fast_d_eff is not None)) and (not self.continual_mode)

        # 1. Dual-Horizon Sparse Causal Self-Attention [B, T, D_eff or D_max]
        attn_out = self.sparse_attention(input_ids)
        tok_emb = self.sparse_attention.token_embedding[input_ids]

        # 3. ALiBi local causal window + capacitor scaling
        if self.enable_hierarchical_compositional and self.use_role_bridge and (not self.continual_mode):
            w_alibi = self.alibi_sliding_weights[:seq_len, :seq_len]
            cap_gain = 0.45
        else:
            w_alibi = self.alibi_causal_weights[:seq_len, :seq_len]
            cap_gain = 1.65

        # 4. Salience-Gated EMA Context Capacitor c_t [B, T, D_max]
        alphas = self.capacitor_alpha[input_ids].unsqueeze(-1)  # [B, T, 1]
        c_states = torch.zeros_like(tok_emb)
        c_prev = torch.zeros(bsz, self.d_max, device=input_ids.device)
        for t in range(seq_len):
            a_t = alphas[:, t, :]
            c_prev = (1.0 - a_t) * c_prev + a_t * tok_emb[:, t, :]
            c_states[:, t, :] = c_prev

        if use_active_slice:
            out_dim = d_eff
            attn_slice = attn_out[:, :, :out_dim] if attn_out.shape[-1] > out_dim else attn_out
            x_pos_sub = tok_emb[:, :, :out_dim] + self.alibi_pos_encoding[:seq_len, :out_dim].unsqueeze(0)
            alibi_ctx = torch.matmul(w_alibi.unsqueeze(0), x_pos_sub)
            cap_ctx = F.linear(c_states, self.cap_proj[:out_dim, :])
            lag1 = torch.zeros(bsz, seq_len, out_dim, device=input_ids.device)
            if seq_len > 1:
                lag1[:, 1:, :] = F.linear(tok_emb[:, :-1, :], self.lag1_proj[:out_dim, :])
            lag2 = torch.zeros(bsz, seq_len, out_dim, device=input_ids.device)
            if seq_len > 2:
                lag2[:, 2:, :] = F.linear(tok_emb[:, :-2, :], self.lag2_proj[:out_dim, :])
            return attn_slice + alibi_ctx + cap_gain * cap_ctx + lag1 + lag2

        pos_emb = self.alibi_pos_encoding[:seq_len].unsqueeze(0)
        x_pos = tok_emb + pos_emb
        alibi_ctx = torch.matmul(w_alibi.unsqueeze(0), x_pos)
        cap_ctx = F.linear(c_states, self.cap_proj)

        # 5. Immediate causal lag-1 and lag-2 projections for sharp local syntax
        lag1 = torch.zeros_like(tok_emb)
        if seq_len > 1:
            lag1[:, 1:, :] = F.linear(tok_emb[:, :-1, :], self.lag1_proj)

        lag2 = torch.zeros_like(tok_emb)
        if seq_len > 2:
            lag2[:, 2:, :] = F.linear(tok_emb[:, :-2, :], self.lag2_proj)

        return attn_out + alibi_ctx + cap_gain * cap_ctx + lag1 + lag2

    @staticmethod
    def _apply_branch_rmsnorm(
        a_sparse: torch.Tensor,
        mask: torch.Tensor,
        target_rms: float = 0.65,
    ) -> torch.Tensor:
        """
        Law 2 (Milestone 2.3): Parameter-Free Active Sparse RMSNorm over winning units K_wins.
        """
        k_wins = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        active_rms = torch.sqrt(a_sparse.pow(2).sum(dim=-1, keepdim=True) / k_wins + 1e-6)
        return a_sparse * (target_rms / active_rms)

    @staticmethod
    def _apply_subspace_rmsnorm(
        x_sub: torch.Tensor,
        target_rms: float = 1.20,
        return_stats: bool = True,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Law 2 (Milestone 2.3): Parameter-Free Layer-Subspace RMS Stabilization across depth.
        Returns (normalized_tensor, raw_rms_mean, normed_rms_mean).
        """
        raw_rms = torch.sqrt(x_sub.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        x_normed = x_sub * (target_rms / raw_rms)
        if not return_stats:
            return x_normed, 0.0, target_rms
        normed_rms = torch.sqrt(x_normed.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x_normed, float(raw_rms.mean().item()), float(normed_rms.mean().item())

    @torch.no_grad()
    def _extract_causal_entity_slots(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Causally tracks the most recently completed recurring content-entity phrase
        (start_emb, end_emb, start_id, end_id, has_completed) across time t = 0..T-1
        without peeking at any future tokens.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        tok_emb = self.sparse_attention.token_embedding[input_ids]
        is_ent = self.is_entity_vocab[input_ids]

        last_start_emb = torch.zeros(bsz, seq_len, self.d_max, device=device)
        last_end_emb = torch.zeros(bsz, seq_len, self.d_max, device=device)
        last_start_id = torch.full((bsz, seq_len), -1, dtype=torch.long, device=device)
        last_end_id = torch.full((bsz, seq_len), -1, dtype=torch.long, device=device)
        has_completed = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)

        cur_start_emb = torch.zeros(bsz, self.d_max, device=device)
        cur_end_emb = torch.zeros(bsz, self.d_max, device=device)
        cur_start_id = torch.full((bsz,), -1, dtype=torch.long, device=device)
        cur_end_id = torch.full((bsz,), -1, dtype=torch.long, device=device)
        cur_len = torch.zeros(bsz, dtype=torch.long, device=device)

        comp_start_emb = torch.zeros(bsz, self.d_max, device=device)
        comp_end_emb = torch.zeros(bsz, self.d_max, device=device)
        comp_start_id = torch.full((bsz,), -1, dtype=torch.long, device=device)
        comp_end_id = torch.full((bsz,), -1, dtype=torch.long, device=device)
        comp_ready = torch.zeros(bsz, dtype=torch.bool, device=device)

        for t in range(seq_len):
            ent_t = is_ent[:, t]
            non_ent_t = ~ent_t
            id_t = input_ids[:, t]
            emb_t = tok_emb[:, t, :]

            just_finished = non_ent_t & (cur_len > 0)
            valid_commit = just_finished & (cur_len >= 2) if self.has_multiword_entities else just_finished
            if valid_commit.any():
                comp_start_emb[valid_commit] = cur_start_emb[valid_commit]
                comp_end_emb[valid_commit] = cur_end_emb[valid_commit]
                comp_start_id[valid_commit] = cur_start_id[valid_commit]
                comp_end_id[valid_commit] = cur_end_id[valid_commit]
                comp_ready[valid_commit] = True
            cur_len[non_ent_t] = 0

            new_phrase = ent_t & (cur_len == 0)
            cont_phrase = ent_t & (cur_len >= 1)
            if new_phrase.any():
                cur_start_emb[new_phrase] = emb_t[new_phrase]
                cur_end_emb[new_phrase] = emb_t[new_phrase]
                cur_start_id[new_phrase] = id_t[new_phrase]
                cur_end_id[new_phrase] = id_t[new_phrase]
            if cont_phrase.any():
                cur_end_emb[cont_phrase] = emb_t[cont_phrase]
                cur_end_id[cont_phrase] = id_t[cont_phrase]
            cur_len[ent_t] += 1

            last_start_emb[:, t, :] = comp_start_emb
            last_end_emb[:, t, :] = comp_end_emb
            last_start_id[:, t] = comp_start_id
            last_end_id[:, t] = comp_end_id
            has_completed[:, t] = comp_ready

        return last_start_emb, last_end_emb, last_start_id, last_end_id, has_completed

    @torch.no_grad()
    def _compute_role_factored_entity_logits(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Law 3 (Milestone 2.3): Role-Factored Compositional Subject-Scene Decoupling.
        Reads local [lag2_unit, lag1_unit, tok_unit] through `role_bridge_head` to predict
        (P_SCENE, P_ENT_START, P_ENT_END). When entity retrieval activates after a completed
        entity phrase, projects `ent_mix` via `entity_proj` into unit token embedding space.
        """
        bsz, seq_len = input_ids.shape
        n_total = bsz * seq_len
        tok_emb = self.sparse_attention.token_embedding[input_ids]
        tok_unit = F.normalize(tok_emb, p=2, dim=-1, eps=1e-6)

        lag1_unit = torch.zeros_like(tok_unit)
        if seq_len > 1:
            lag1_unit[:, 1:, :] = tok_unit[:, :-1, :]
        lag2_unit = torch.zeros_like(tok_unit)
        if seq_len > 2:
            lag2_unit[:, 2:, :] = tok_unit[:, :-2, :]
        lag3_unit = torch.zeros_like(tok_unit)
        if seq_len > 3:
            lag3_unit[:, 3:, :] = tok_unit[:, :-3, :]
        bias_col = torch.ones(bsz, seq_len, 1, device=input_ids.device)

        win_feat = torch.cat(
            [lag3_unit, lag2_unit, lag1_unit, tok_unit, bias_col], dim=-1
        ).reshape(n_total, 4 * self.d_max + 1)
        role_logits = self.role_bridge_head(win_feat)  # [N, 3]
        role_probs = F.softmax(role_logits, dim=-1)
        p_ent_start = role_probs[:, 1]
        p_ent_end = role_probs[:, 2]
        p_ent_sum = p_ent_start + p_ent_end

        start_emb, end_emb, start_id, end_id, has_comp = self._extract_causal_entity_slots(input_ids)
        start_flat = start_emb.reshape(n_total, self.d_max)
        end_flat = end_emb.reshape(n_total, self.d_max)
        has_comp_flat = has_comp.reshape(n_total)

        ent_mix = (
            p_ent_start.unsqueeze(-1) * start_flat + p_ent_end.unsqueeze(-1) * end_flat
        ) / p_ent_sum.clamp(min=1e-6).unsqueeze(-1)

        ent_proj_out = self.entity_proj(ent_mix)
        ent_vec = F.normalize(ent_proj_out, p=2, dim=-1, eps=1e-6)
        emb_unit = F.normalize(self.sparse_attention.token_embedding, p=2, dim=-1, eps=1e-6)
        ent_logits = 22.0 * F.linear(ent_vec, emb_unit)  # [N, V]

        ent_gate = has_comp_flat & (p_ent_sum > 0.60)
        slots_tuple = (start_id.reshape(n_total), end_id.reshape(n_total), has_comp_flat)
        return ent_logits, ent_gate, role_probs, p_ent_sum, slots_tuple

    @staticmethod
    def _compute_block_entropy(counts: torch.Tensor) -> float:
        total = float(counts.sum().item())
        n = counts.numel()
        if total <= 0.0 or n <= 1:
            return 0.0
        p = counts.float() / total
        p = p[p > 0]
        ent = -float(torch.sum(p * torch.log2(p)).item())
        return ent / math.log2(n)

    @torch.no_grad()
    def _forward_block_sparse_layer(
        self,
        layer: DynamicSpringLayer,
        a_prev_eff: torch.Tensor,
        l_idx: int = 0,
    ) -> torch.Tensor:
        """
        Laws 1, 2 & 4 (Milestone 2.4):
        Contiguous Block-Sparse Elastic Capsules (B_block = 16) with:
          - Active-Slice contiguous matmul on [:d_act, :d_eff]
          - Load-Balanced Top-K_blocks capsule competition during training
          - Depth-Scaled Residual Highway (alpha_res / sqrt(L)) + Local Active RMSNorm
        """
        in_dim = a_prev_eff.shape[1]
        if self.frozen:
            w_sub = layer.w0[:, :in_dim] if layer.w0.shape[1] != in_dim else layer.w0
            z_act = F.linear(a_prev_eff, w_sub)
            d_sub = z_act.shape[1]
        else:
            d_act = layer.d_active
            w_sub = layer.w[:d_act, :in_dim]
            z_act = F.linear(a_prev_eff, w_sub)
            d_sub = d_act

        n_tok = z_act.shape[0]
        b_size = self.block_size
        n_blocks = max(1, d_sub // b_size)
        if self.enable_hierarchical_compositional:
            k_blk = min(self.hierarchical_top_k_blocks[l_idx], n_blocks)
        else:
            k_blk = max(1, min(n_blocks, int(round(self.top_k / float(b_size)))))

        z_blocks = z_act.view(n_tok, n_blocks, b_size)
        block_energy = torch.sqrt(z_blocks.pow(2).mean(dim=-1) + 1e-6)  # [N, N_blocks]

        if (
            self._is_training_step
            and (not self.frozen)
            and k_blk < n_blocks
            and self.block_balance_strength > 0.0
        ):
            ema = self.block_usage_ema[l_idx][:n_blocks]
            rel_u = ema / ema.mean().clamp(min=1e-6)
            rank_score = block_energy * (rel_u + 0.10).pow(-self.block_balance_strength)
        else:
            rank_score = block_energy

        if (
            self.use_fused_kernels
            and self.frozen
            and k_blk < n_blocks
        ):
            _, top_b_idx = torch.topk(rank_score, k=k_blk, dim=-1)
            in_fast_gen = getattr(self, "_in_fast_generate", False)
            if self._frozen_block_wins is not None and not in_fast_gen:
                self._frozen_block_wins[l_idx][:n_blocks].scatter_add_(
                    0,
                    top_b_idx.reshape(-1),
                    torch.ones(n_tok * k_blk, device=z_act.device, dtype=torch.float32),
                )
            res_scale = (self.alpha_res / math.sqrt(float(self.num_layers))) if self.use_depth_scaled_highway else 1.0
            raw_out = indexed_block_activate_highway(
                z_act=z_act,
                a_prev_eff=a_prev_eff,
                top_b_idx=top_b_idx,
                block_size=b_size,
                neg_slope=float(layer.negative_slope),
                res_scale=res_scale,
                use_highway=self.use_depth_scaled_highway,
            )
            need_stats = (not self._is_training_step) and (not in_fast_gen)
            if self.use_depth_scaled_highway:
                if d_sub == in_dim:
                    out, raw_rms, normed_rms = self._apply_subspace_rmsnorm(
                        raw_out, target_rms=1.20, return_stats=need_stats
                    )
                else:
                    out_sub, raw_rms, normed_rms = self._apply_subspace_rmsnorm(
                        raw_out[:, :d_sub], target_rms=1.20, return_stats=need_stats
                    )
                    out = raw_out
                    out[:, :d_sub] = out_sub
            else:
                out = raw_out
                raw_rms = (
                    0.0
                    if (self._is_training_step or in_fast_gen)
                    else float(torch.sqrt(out[:, :d_sub].pow(2).mean(dim=-1, keepdim=True) + 1e-6).mean().item())
                )
                normed_rms = raw_rms

            self._last_layer_raw_rms[l_idx] = raw_rms
            self._last_layer_normed_rms[l_idx] = normed_rms
            return out

        if k_blk < n_blocks:
            _, top_b_idx = torch.topk(rank_score, k=k_blk, dim=-1)  # [N, k_blk]
            idx_exp = top_b_idx.unsqueeze(-1).expand(-1, -1, b_size)
            z_win = torch.gather(z_blocks, dim=1, index=idx_exp)
            a_win = F.leaky_relu(z_win, negative_slope=layer.negative_slope)
            a_blocks = torch.zeros_like(z_blocks).scatter_(1, idx_exp, a_win)
            a_sparse = a_blocks.reshape(n_tok, d_sub)
            block_mask = torch.zeros(n_tok, n_blocks, device=z_act.device)
            block_mask.scatter_(1, top_b_idx, 1.0)
        else:
            a_sparse = F.leaky_relu(z_act, negative_slope=layer.negative_slope)
            block_mask = torch.ones(n_tok, n_blocks, device=z_act.device)

        batch_b_wins = block_mask.sum(dim=0)
        if not self.frozen:
            self.block_win_counts[l_idx][:n_blocks].add_(batch_b_wins)
            self.epoch_block_win_counts[l_idx][:n_blocks].add_(batch_b_wins)
            if self._is_training_step:
                ema_slice = self.block_usage_ema[l_idx][:n_blocks]
                ema_slice.mul_(0.88).add_(0.12 * (batch_b_wins / max(float(n_tok), 1.0)))

            cell_wins = batch_b_wins.repeat_interleave(b_size)
            layer.win_counts[:d_sub].add_(cell_wins)
            layer.epoch_win_counts[:d_sub].add_(cell_wins)

            if not self._is_training_step:
                winner_blocks = batch_b_wins > 0
                if winner_blocks.any():
                    winner_cells = winner_blocks.repeat_interleave(b_size)
                    layer._last_winner_k = float(layer.k[:d_sub, :in_dim][winner_cells].mean().item())
                self._last_block_entropy[l_idx] = self._compute_block_entropy(
                    self.block_win_counts[l_idx][:n_blocks]
                )

            layer._last_a_prev = a_prev_eff.detach()
            layer._last_z = z_act.detach()
            layer._last_mask = block_mask.detach()
            layer._last_a = a_sparse.detach()
        elif self._frozen_block_wins is not None:
            self._frozen_block_wins[l_idx][:n_blocks].add_(batch_b_wins)

        if self.use_depth_scaled_highway:
            res_scale = self.alpha_res / math.sqrt(float(self.num_layers))
            need_stats = not self._is_training_step
            if d_sub == in_dim:
                raw_out = a_prev_eff + res_scale * a_sparse
                out, raw_rms, normed_rms = self._apply_subspace_rmsnorm(
                    raw_out, target_rms=1.20, return_stats=need_stats
                )
            else:
                raw_out = a_prev_eff.clone()
                raw_out[:, :d_sub].add_(a_sparse, alpha=res_scale)
                out_sub, raw_rms, normed_rms = self._apply_subspace_rmsnorm(
                    raw_out[:, :d_sub], target_rms=1.20, return_stats=need_stats
                )
                out = raw_out
                out[:, :d_sub] = out_sub
        else:
            if d_sub == in_dim:
                out = a_prev_eff + a_sparse
                raw_rms = (
                    0.0
                    if self._is_training_step
                    else float(torch.sqrt(out.pow(2).mean(dim=-1, keepdim=True) + 1e-6).mean().item())
                )
            else:
                out = a_prev_eff.clone()
                out[:, :d_sub].add_(a_sparse)
                raw_rms = (
                    0.0
                    if self._is_training_step
                    else float(torch.sqrt(out[:, :d_sub].pow(2).mean(dim=-1, keepdim=True) + 1e-6).mean().item())
                )
            normed_rms = raw_rms

        self._last_layer_raw_rms[l_idx] = raw_rms
        self._last_layer_normed_rms[l_idx] = normed_rms
        return out

    @torch.no_grad()
    def _update_block_sparse_layer(
        self,
        layer: DynamicSpringLayer,
        projected_error: torch.Tensor,
        n_valid: float,
        maturity_stiff_gain: float,
        l_idx: int = 0,
    ) -> None:
        """
        Active-Slice Winner-Normalized Local Spring Update for Contiguous Block-Sparse Capsules.
        Updates strictly [:d_act, :d_eff] without touching dormant rows or columns.
        """
        if self.use_fused_kernels:
            d_act = layer.d_active
            n_blocks = max(1, d_act // self.block_size)
            if self.enable_hierarchical_compositional:
                k_blk = min(self.hierarchical_top_k_blocks[l_idx], n_blocks)
            else:
                k_blk = max(1, min(n_blocks, int(round(self.top_k / float(self.block_size)))))
            stiff_mult = self.layer_stiff_multipliers[l_idx] if self.enable_hierarchical_compositional else 1.0
            stiff_ceil = self.layer_stiff_ceilings[l_idx] if self.enable_hierarchical_compositional else 0.85
            batch_active_block_union_update(
                layer=layer,
                projected_error=projected_error,
                n_valid=n_valid,
                maturity_stiff_gain=maturity_stiff_gain,
                block_size=self.block_size,
                k_blk=k_blk,
                stiff_mult=stiff_mult,
                stiff_ceil=stiff_ceil,
                scratch=self._kernel_scratch,
                compensator=self._mp_compensator,
            )
            return

        d_act = layer.d_active
        a_prev_eff = layer._last_a_prev      # [N, d_eff]
        d_eff = a_prev_eff.shape[1]
        z_act = layer._last_z                # [N, d_act]
        block_mask = layer._last_mask        # [N, n_blocks]

        n_tok = z_act.shape[0]
        b_size = self.block_size
        n_blocks = max(1, d_act // b_size)

        g_prime = layer._activate_derivative(z_act)  # [N, d_act]
        delta = (
            (projected_error[:, :d_act] * g_prime).view(n_tok, n_blocks, b_size)
            * block_mask.unsqueeze(-1)
        ).reshape(n_tok, d_act)

        wins_per_block = block_mask.sum(dim=0).clamp(min=1.0)
        wins_per_cell = wins_per_block.repeat_interleave(b_size).unsqueeze(1)  # [d_act, 1]
        if self.enable_hierarchical_compositional:
            k_blk = min(self.hierarchical_top_k_blocks[l_idx], n_blocks)
        else:
            k_blk = max(1, min(n_blocks, int(round(self.top_k / float(b_size)))))
        sparsity_ratio = float(n_blocks) / float(max(k_blk, 1))
        balanced_denom = (0.40 * max(n_valid, 1.0)) + (0.60 * wins_per_cell * sparsity_ratio)
        driving_signal = torch.matmul(delta.t(), a_prev_eff) / balanced_denom  # [d_act, d_eff]

        w_act = layer.w[:d_act, :d_eff]
        w0_act = layer.w0[:d_act, :d_eff]
        k_act = layer.k[:d_act, :d_eff]

        stretch_diff = w_act - w0_act
        delta_w = (layer.eta_w * driving_signal) - (layer.gamma * k_act * stretch_diff)
        w_act.add_(delta_w)

        stretch_diff_new = w_act - w0_act
        delta_w0 = layer.eta_0 * (1.0 - k_act) * stretch_diff_new
        w0_act.add_(delta_w0)

        active_row_gate = (wins_per_block > 0).repeat_interleave(b_size).float().unsqueeze(1)
        win_norm_factor = (max(n_valid, 1.0) / (wins_per_cell * sparsity_ratio)).clamp(1.0, 8.0)
        stiff_mult = self.layer_stiff_multipliers[l_idx] if self.enable_hierarchical_compositional else 1.0
        stiff_ceil = self.layer_stiff_ceilings[l_idx] if self.enable_hierarchical_compositional else 0.85
        effective_beta_k = layer.beta_k * maturity_stiff_gain * win_norm_factor * stiff_mult
        delta_k = effective_beta_k * stretch_diff_new.abs() * (1.0 - k_act) * active_row_gate
        k_act.add_(delta_k).clamp_(0.0, stiff_ceil)

    @torch.no_grad()
    def reset_frozen_block_stats(self) -> None:
        n_max_blocks = max(1, self.d_max // self.block_size)
        self._frozen_block_wins = [
            torch.zeros(n_max_blocks, dtype=torch.float32) for _ in range(self.num_layers)
        ]

    @torch.no_grad()
    def _forward_dynamic_layer(
        self,
        layer: DynamicSpringLayer,
        a_prev: torch.Tensor,
        l_idx: int = 0,
        phase_id: int | None = None,
    ) -> torch.Tensor:
        """
        Executes a DynamicSpringLayer, supporting:
          - Compacted frozen W0 matrices & Resonant Specialist Subspace Routing (continual mode)
          - Hierarchical Top-K Spectrum & Parameter-Free Local Active RMSNorm (Milestone 2.3)
        """
        k_base = self.hierarchical_top_ks[l_idx] if self.enable_hierarchical_compositional else layer.top_k

        if self.frozen:
            has_comp = hasattr(layer, "_compacted_indices") and layer._compacted_indices is not None
            if not has_comp and not self.enable_hierarchical_compositional:
                out = layer(a_prev)
                d_out = layer.w0.shape[0]
                rms = float(torch.sqrt(out[:, :d_out].pow(2).mean(dim=-1, keepdim=True) + 1e-6).mean().item())
                self._last_layer_raw_rms[l_idx] = rms
                self._last_layer_normed_rms[l_idx] = rms
                return out

            in_dim = layer.w0.shape[1]
            if has_comp:
                idx = layer._compacted_indices
                if phase_id is not None and phase_id in self.phase_compacted_slots:
                    c_pos = self.phase_compacted_slots[phase_id][l_idx]
                    w_sub = layer.w0[c_pos, :]
                    orig_idx = idx[c_pos]
                else:
                    w_sub = layer.w0
                    orig_idx = idx
            else:
                w_sub = layer.w0
                orig_idx = None

            z_kept = F.linear(a_prev[:, :in_dim], w_sub)  # [N, D_sub]
            d_kept = z_kept.shape[1]
            if self.continual_mode:
                k_eff = min(k_base or d_kept, max(8, d_kept // 2)) if d_kept > 8 else d_kept
            else:
                k_eff = min(k_base, d_kept) if k_base is not None else d_kept

            if k_eff is not None and 0 < k_eff < d_kept:
                _, top_idx = torch.topk(z_kept.abs(), k=k_eff, dim=-1)
                mask_kept = torch.zeros_like(z_kept)
                mask_kept.scatter_(-1, top_idx, 1.0)
            else:
                mask_kept = torch.ones_like(z_kept)
            act_kept = F.leaky_relu(z_kept, negative_slope=0.1) * mask_kept

            a_out = a_prev.clone()
            if orig_idx is not None:
                a_out[:, orig_idx] = a_out[:, orig_idx] + act_kept
                sub_view = a_out[:, orig_idx]
            else:
                a_out[:, :d_kept] = a_out[:, :d_kept] + act_kept
                sub_view = a_out[:, :d_kept]

            if self.enable_hierarchical_compositional:
                sub_normed, raw_rms, normed_rms = self._apply_subspace_rmsnorm(sub_view, target_rms=1.20)
                if orig_idx is not None:
                    a_out[:, orig_idx] = sub_normed
                else:
                    a_out[:, :d_kept] = sub_normed
            else:
                raw_rms = float(torch.sqrt(sub_view.pow(2).mean(dim=-1, keepdim=True) + 1e-6).mean().item())
                normed_rms = raw_rms

            self._last_layer_raw_rms[l_idx] = raw_rms
            self._last_layer_normed_rms[l_idx] = normed_rms
            return a_out

        if not self.continual_mode and not self.enable_hierarchical_compositional:
            out = layer(a_prev)
            d_act = layer.d_active
            rms = float(torch.sqrt(out[:, :d_act].pow(2).mean(dim=-1, keepdim=True) + 1e-6).mean().item())
            self._last_layer_raw_rms[l_idx] = rms
            self._last_layer_normed_rms[l_idx] = rms
            return out

        d_act = layer.d_active
        w_active = layer.w[:d_act, :]
        z_active = F.linear(a_prev, w_active)

        if self.continual_mode and self.current_phase_slots[l_idx]:
            plastic_mask = torch.zeros(d_act, dtype=torch.bool, device=a_prev.device)
            for s_idx in self.current_phase_slots[l_idx]:
                if s_idx < d_act:
                    plastic_mask[s_idx] = True
            z_for_topk = z_active.abs().masked_fill(~plastic_mask.unsqueeze(0), -1.0)
            pool_size = int(plastic_mask.sum().item())
            k_eff = min(k_base or d_act, max(8, pool_size // 2)) if pool_size > 8 else pool_size
        else:
            z_for_topk = z_active.abs()
            k_eff = min(k_base, d_act) if k_base is not None else d_act

        if k_eff is not None and 0 < k_eff < d_act:
            _, top_indices = torch.topk(z_for_topk, k=k_eff, dim=-1)
            mask_active = torch.zeros_like(z_active)
            mask_active.scatter_(-1, top_indices, 1.0)
        else:
            mask_active = torch.ones_like(z_active)
        a_active = layer._activate(z_active) * mask_active

        bsz = a_prev.shape[0]
        z_full = torch.zeros(bsz, self.d_max, device=a_prev.device)
        mask_full = torch.zeros(bsz, self.d_max, device=a_prev.device)
        a_full = torch.zeros(bsz, self.d_max, device=a_prev.device)
        z_full[:, :d_act] = z_active
        mask_full[:, :d_act] = mask_active

        batch_wins = mask_full.sum(dim=0)
        layer.win_counts.add_(batch_wins)
        layer.epoch_win_counts.add_(batch_wins)

        winner_rows = mask_active.sum(dim=0) > 0
        if winner_rows.any():
            layer._last_winner_k = float(layer.k[:d_act, :][winner_rows].mean().item())

        a_full[:, :d_act] = a_active

        layer._last_a_prev = a_prev.detach()
        layer._last_z = z_full.detach()
        layer._last_mask = mask_full.detach()
        layer._last_a = a_full.detach()

        out = a_prev.clone()
        out[:, :d_act] = a_prev[:, :d_act] + a_active

        if self.enable_hierarchical_compositional:
            out_sub, raw_rms, normed_rms = self._apply_subspace_rmsnorm(out[:, :d_act], target_rms=1.20)
            out[:, :d_act] = out_sub
        else:
            raw_rms = float(torch.sqrt(out[:, :d_act].pow(2).mean(dim=-1, keepdim=True) + 1e-6).mean().item())
            normed_rms = raw_rms

        self._last_layer_raw_rms[l_idx] = raw_rms
        self._last_layer_normed_rms[l_idx] = normed_rms
        return out

    @torch.no_grad()
    def _detect_resonant_phase(self, seq_ids: torch.Tensor) -> int:
        """
        Determines which consolidated continual phase p* in {1..P} resonates strongest
        with the causal token sequence `seq_ids` (1D tensor of length T).
        """
        best_pid = 1
        best_score = -1e9
        for pid, prof in self.phase_resonance_profiles.items():
            score = float(prof[seq_ids].sum().item())
            if score > best_score:
                best_score = score
                best_pid = pid
        return best_pid

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor | list[int]) -> torch.Tensor:
        single_seq = False
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            single_seq = True
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            single_seq = True

        bsz, seq_len = input_ids.shape
        context = self._build_vectorized_context(input_ids)  # [B, T, D_eff or D_max]

        use_block_path = self.enable_block_sparse and (not self.continual_mode)
        if not use_block_path and (self._fast_d_eff is None):
            d_in = self._effective_context_dim()
            if d_in < self.d_max:
                context = context.clone()
                context[:, :, d_in:] = 0.0

        if self.frozen and self.continual_mode and self.phase_resonance_profiles:
            in_dim = self.output_head.w0.shape[1]
            logits = torch.full((bsz, seq_len, self.vocab_size), -1e4, device=input_ids.device)
            seq_pids = [self._detect_resonant_phase(input_ids[b]) for b in range(bsz)]

            for pid in sorted(set(seq_pids)):
                b_indices = [b for b, p in enumerate(seq_pids) if p == pid]
                b_tensor = torch.tensor(b_indices, dtype=torch.long, device=input_ids.device)
                a_sub = context[b_tensor].reshape(len(b_indices) * seq_len, self.d_max)
                for l_idx, layer in enumerate(self.spring_layers):
                    a_sub = self._forward_dynamic_layer(layer, a_sub, l_idx=l_idx, phase_id=pid)

                w0_use = self.output_head.w0.clone()
                if self.shared_vocab_tensor is not None and pid in self.phase_shared_bedrock:
                    w0_use[self.shared_vocab_tensor, :in_dim] = self.phase_shared_bedrock[pid][:, :in_dim]

                sub_logits = F.linear(a_sub[:, :in_dim], w0_use)
                if pid in self.phase_vocab_masks:
                    inactive_v = ~self.phase_vocab_masks[pid]
                    sub_logits[:, inactive_v] = -1e4
                logits[b_tensor] = sub_logits.reshape(len(b_indices), seq_len, self.vocab_size)

            if single_seq:
                return logits[0, -1]
            return logits

        if use_block_path:
            a = context.reshape(bsz * seq_len, context.shape[-1])
            if self.use_depth_scaled_highway:
                a, _, _ = self._apply_subspace_rmsnorm(a, target_rms=1.20, return_stats=False)
            for l_idx, layer in enumerate(self.spring_layers):
                a = self._forward_block_sparse_layer(layer, a, l_idx=l_idx)
        else:
            a = context.reshape(bsz * seq_len, context.shape[-1])
            for l_idx, layer in enumerate(self.spring_layers):
                a = self._forward_dynamic_layer(layer, a, l_idx=l_idx)

        if self.frozen:
            in_dim = self.output_head.w0.shape[1]
            scene_logits = F.linear(a[:, :in_dim] if a.shape[1] != in_dim else a, self.output_head.w0)
        elif use_block_path:
            d_last = self._final_layer_dim()
            a_head = a[:, :d_last] if a.shape[1] != d_last else a
            scene_logits = F.linear(a_head, self.output_head.w[:, :d_last])
            self.output_head._last_a_prev = a_head.detach()
        else:
            d_last = self._final_layer_dim()
            a_masked = a.clone()
            if d_last < self.d_max:
                a_masked[:, d_last:] = 0.0
            scene_logits = self.output_head(a_masked)

        if self.enable_hierarchical_compositional and self.use_role_bridge and (not self.continual_mode):
            ent_logits, ent_gate, role_probs, _, slots_tuple = self._compute_role_factored_entity_logits(
                input_ids
            )
            logits_flat = scene_logits.clone()
            if ent_gate.any():
                logits_flat[ent_gate] = 0.15 * scene_logits[ent_gate] + ent_logits[ent_gate]
            self._last_role_probs = role_probs.detach()
            self._last_ent_logits = ent_logits.detach()
            self._last_ent_slots = slots_tuple
        else:
            logits_flat = scene_logits
            self._last_role_probs = None
            self._last_ent_logits = None
            self._last_ent_slots = None

        logits = logits_flat.reshape(bsz, seq_len, self.vocab_size)
        if single_seq:
            return logits[0, -1]
        return logits

    @torch.no_grad()
    def _update_dynamic_layer_winner_normalized(
        self,
        layer: DynamicSpringLayer,
        projected_error: torch.Tensor,
        n_valid: float,
        maturity_stiff_gain: float,
        l_idx: int = 0,
    ) -> None:
        """
        Law 3 & Milestone 2.2 Cubic Anchor Shield:
        Normalizes each cell j's driving signal and stiffness consolidation by its
        active winner participation N_wins(j) in the mini-batch, and applies the
        Cubic Anchor Shield (1 - k)^3 on sleep-armored cells (k >= 0.80) so that
        consolidated memory resists catastrophic forgetting across phases.
        """
        d_act = layer.d_active
        a_prev = layer._last_a_prev            # [N, d_max]
        z_act = layer._last_z[:, :d_act]       # [N, d_active]
        mask_act = layer._last_mask[:, :d_act] # [N, d_active]
        proj_act = projected_error[:, :d_act] # [N, d_active]

        g_prime = layer._activate_derivative(z_act)
        delta = proj_act * g_prime * mask_act  # [N, d_active]

        # Winner-count normalization per cell j
        wins_per_cell = mask_act.sum(dim=0).clamp(min=1.0).unsqueeze(1)  # [d_active, 1]
        sparsity_ratio = float(d_act) / float(max(layer.top_k or d_act, 1))
        balanced_denom = (0.40 * max(n_valid, 1.0)) + (0.60 * wins_per_cell * sparsity_ratio)
        driving_signal = torch.matmul(delta.t(), a_prev) / balanced_denom  # [d_active, d_max]

        w_act = layer.w[:d_act, :]
        w0_act = layer.w0[:d_act, :]
        k_act = layer.k[:d_act, :]

        # Law 1: Fast Elastic Stretch (w) with armor damping on locked cells
        stretch_diff = w_act - w0_act
        if self.use_cubic_shield:
            plastic_gate_w = torch.where(k_act >= 0.80, (1.0 - k_act).pow(2), torch.ones_like(k_act))
        else:
            plastic_gate_w = 1.0
        delta_w = (layer.eta_w * plastic_gate_w * driving_signal) - (layer.gamma * k_act * stretch_diff)
        w_act.add_(delta_w)

        stretch_diff_new = w_act - w0_act

        # Law 2: Anchor Consolidation (w0) with Cubic Anchor Shield (1 - k)^3 on armored cells
        if self.use_cubic_shield:
            shield_w0 = torch.where(k_act >= 0.80, (1.0 - k_act).pow(3), 1.0 - k_act)
        else:
            shield_w0 = 1.0 - k_act
        delta_w0 = layer.eta_0 * shield_w0 * stretch_diff_new
        w0_act.add_(delta_w0)

        # Law 3: Winner-Count Normalized & Maturity-Gated Stiffening (k) with Hierarchical Spectrum
        active_row_gate = (mask_act.sum(dim=0) > 0).float().unsqueeze(1)
        win_norm_factor = (max(n_valid, 1.0) / (wins_per_cell * sparsity_ratio)).clamp(1.0, 8.0)
        stiff_mult = self.layer_stiff_multipliers[l_idx] if self.enable_hierarchical_compositional else 1.0
        stiff_ceil = self.layer_stiff_ceilings[l_idx] if self.enable_hierarchical_compositional else 0.85
        effective_beta_k = layer.beta_k * maturity_stiff_gain * win_norm_factor * stiff_mult
        delta_k = effective_beta_k * stretch_diff_new.abs() * (1.0 - k_act) * active_row_gate
        # Preserve sleep-armored stiffness (> 0.85) while respecting hierarchical stiffness ceilings
        max_cap = torch.where(k_act >= 0.85, k_act, torch.full_like(k_act, stiff_ceil))
        k_act.add_(delta_k).clamp_(0.0, 0.98)
        k_act.copy_(torch.minimum(k_act, max_cap))

    @torch.no_grad()
    def train_minibatch(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        pad_id: int = 0,
    ) -> dict[str, float]:
        """
        Vectorized Mini-Batch Forward + Salience-Weighted Error + Winner-Normalized Spring Update
        + Row-Normalized Hebbian Feedback Alignment + Cubic Armor Shielding.
        """
        if self.frozen:
            raise RuntimeError("Model is frozen; training is disabled.")

        bsz, seq_len = input_ids.shape
        n_total = bsz * seq_len

        # 1. Vectorized Forward Pass: [B, T, V]
        self._is_training_step = True
        try:
            logits = self.forward(input_ids)
        finally:
            self._is_training_step = False
        logits_flat = logits.reshape(n_total, self.vocab_size)
        targets_flat = target_ids.reshape(n_total)
        inputs_flat = input_ids.reshape(n_total)

        valid_mask = (targets_flat != pad_id).float()
        n_valid = float(valid_mask.sum().item())
        if n_valid == 0:
            return {"ce_loss": 0.0, "mse_loss": 0.0, "correct": 0, "valid_tokens": 0}

        # Track seen target tokens in the current continual phase
        valid_targets = targets_flat[targets_flat != pad_id]
        self.phase_seen_targets.scatter_add_(
            0, valid_targets, torch.ones_like(valid_targets, dtype=torch.float32)
        )

        if self.use_fused_kernels and not (self.continual_mode and self.use_cubic_shield):
            probs = F.softmax(logits_flat, dim=-1)
            valid_rows = targets_flat != pad_id
            tgt_probs = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
            ce_loss = float((-torch.log(tgt_probs[valid_rows].clamp(min=1e-30))).mean().item())
            preds_flat = probs.argmax(dim=-1)
            correct = int(((preds_flat == targets_flat) & valid_rows).sum().item())

            target_sal = self.token_salience[targets_flat]
            input_sal = self.token_salience[inputs_flat]
            step_salience = (0.65 * target_sal + 0.35 * input_sal).clamp(0.70, 1.80)

            probs.neg_()
            probs.scatter_add_(1, targets_flat.unsqueeze(1), valid_mask.unsqueeze(1))
            if n_valid < float(n_total):
                probs.mul_(valid_mask.unsqueeze(-1))
            mse_loss = float((probs.norm().pow(2) * 0.5).item() / n_valid)
            probs.mul_(step_salience.unsqueeze(-1))
            salient_err_hidden = probs
            e_scaled = None  # Scaled in-place right after hidden DFA updates
        else:
            if self.continual_mode and self.use_cubic_shield:
                inactive_vocab = ~self.phase_active_vocab_mask
                train_logits = logits_flat.clone()
                train_logits[:, inactive_vocab] = -1e4
                probs = F.softmax(train_logits, dim=-1)
            else:
                probs = F.softmax(logits_flat, dim=-1)
            y_onehot = F.one_hot(targets_flat, num_classes=self.vocab_size).float()

            # Law 1: Weight error signal by target & input token salience s(v) in [0.70, 1.80]
            target_sal = self.token_salience[targets_flat]
            input_sal = self.token_salience[inputs_flat]
            step_salience = (0.65 * target_sal + 0.35 * input_sal).clamp(0.70, 1.80)

            raw_err = (y_onehot - probs) * valid_mask.unsqueeze(-1)
            if self.continual_mode and self.use_cubic_shield:
                raw_err[:, ~self.phase_active_vocab_mask] = 0.0
            salient_err = raw_err * step_salience.unsqueeze(-1)
            salient_err_hidden = salient_err

            # Metrics on valid tokens (evaluated on full unmasked logits)
            ce_loss = float(F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id).item())
            mse_loss = float((0.5 * (raw_err ** 2).sum()).item() / n_valid)
            preds_flat = logits_flat.argmax(dim=-1)
            correct = int(((preds_flat == targets_flat) & (targets_flat != pad_id)).sum().item())

        # Law 3: Maturity-Gated Stiffening schedule (gentle warmup in Epochs 1-2, then ramps up)
        if self.current_epoch <= 2:
            maturity_gain = 2.0
        else:
            maturity_gain = min(280.0, 32.0 * float(self.current_epoch - 1))

        scale_factor = float(n_total) / max(n_valid, 1.0)
        use_inplace_err = self.use_fused_kernels and not (self.continual_mode and self.use_cubic_shield)
        if not use_inplace_err:
            e_scaled = salient_err * scale_factor
            # 2. Update Dual-Horizon Attention locally (shielded once armor-locked after Phase 1)
            if not (self.use_cubic_shield and self.attn_armor_locked):
                self.sparse_attention.local_spring_update(e_scaled)

        # 3. Law 3 & Law 4: Update Hidden DynamicSpringLayers + Adaptive/Anchored Hebbian Feedback B^(l)
        use_block_path = self.enable_block_sparse and (not self.continual_mode)
        depth_dfa_scale = (
            ((4.0 / float(max(self.num_layers, 4))) ** 0.25)
            if (use_block_path and self.use_depth_scaled_highway and self.num_layers > 4)
            else 1.0
        )
        anchor_unit_full = None
        e_highway = None
        d_last = self._final_layer_dim()
        if self.factorized_dfa is not None:
            head_w_active = self.output_head.w[:, :d_last]
            e_highway = self.factorized_dfa.project_vocab_to_highway(salient_err_hidden, head_w_active)
        elif self.use_highway_anchored_dfa:
            max_act_d = max(l.d_active for l in self.spring_layers)
            anchor_unit_full = F.normalize(
                self.output_head.w[:, :max_act_d].t(),
                p=2,
                dim=-1,
                eps=1e-6,
            )
        target_row_norm = math.sqrt(self.vocab_size / float(self.d_max))

        for l_idx, layer in enumerate(self.spring_layers):
            d_act = layer.d_active
            if self.factorized_dfa is not None:
                delta_l_active = self.factorized_dfa.broadcast_layer_error(
                    layer_idx=l_idx,
                    e_highway=e_highway,
                    d_act=d_act,
                    depth_scale=depth_dfa_scale,
                )
            else:
                B_l = self.dfa_router.get_matrix(l_idx)  # [D_max, V]
                B_l_active = B_l[:d_act, :]              # [d_active, V]

                delta_l_active = F.linear(salient_err_hidden, B_l_active)  # [N, d_active]
                if depth_dfa_scale != 1.0:
                    delta_l_active = delta_l_active * depth_dfa_scale

            if use_block_path:
                self._update_block_sparse_layer(
                    layer=layer,
                    projected_error=delta_l_active,
                    n_valid=n_valid,
                    maturity_stiff_gain=maturity_gain,
                    l_idx=l_idx,
                )
            else:
                self._update_dynamic_layer_winner_normalized(
                    layer=layer,
                    projected_error=delta_l_active,
                    n_valid=n_valid,
                    maturity_stiff_gain=maturity_gain,
                    l_idx=l_idx,
                )

            # Keep rejuvenated/dormant slots zeroed until explicitly awakened
            if self.rejuvenated_slots[l_idx]:
                dorm_idx = torch.tensor(self.rejuvenated_slots[l_idx], dtype=torch.long)
                layer.w[dorm_idx, :] = 0.0
                layer.w0[dorm_idx, :] = 0.0
                layer.k[dorm_idx, :] = 0.0
            if self.continual_mode and self.armor_locked_masks[l_idx][:d_act].any():
                locked_idx = self.armor_locked_masks[l_idx][:d_act]
                layer.w0[:d_act][locked_idx, : self._effective_context_dim()].copy_(
                    self.layer_w0_bedrock[l_idx][:d_act][locked_idx, : self._effective_context_dim()]
                )
                layer.w[:d_act][locked_idx, : self._effective_context_dim()].copy_(
                    self.layer_w0_bedrock[l_idx][:d_act][locked_idx, : self._effective_context_dim()]
                )

            # Law 4 / Law 3 (v2.4 / v3.1): Row-Normalized Hebbian + Highway-Anchored or Factorized DFA
            if self.factorized_dfa is not None and layer._last_a is not None:
                a_act = layer._last_a[:, :d_act]
                unlocked_row_gate = (~self.armor_locked_masks[l_idx][:d_act]).float().unsqueeze(1)
                self.factorized_dfa.adapt_layer(
                    layer_idx=l_idx,
                    a_act=a_act,
                    e_highway=e_highway,
                    n_valid=n_valid,
                    unlocked_row_gate=unlocked_row_gate,
                )
            elif self.use_highway_anchored_dfa and anchor_unit_full is not None and layer._last_a is not None:
                a_act = layer._last_a[:, :d_act]
                hebb_corr = torch.matmul(a_act.t(), salient_err_hidden) / max(n_valid, 1.0)
                unlocked_row_gate = (~self.armor_locked_masks[l_idx][:d_act]).float().unsqueeze(1)
                b_hebb = F.normalize(
                    B_l_active + (self.eta_b * unlocked_row_gate) * hebb_corr,
                    p=2,
                    dim=-1,
                    eps=1e-6,
                )
                B_blended = F.normalize(
                    (1.0 - self.dfa_anchor_blend) * b_hebb + self.dfa_anchor_blend * anchor_unit_full[:d_act, :],
                    p=2,
                    dim=-1,
                    eps=1e-6,
                ) * target_row_norm
                B_l[:d_act, :].copy_(B_blended)
            elif self.eta_b > 0.0 and layer._last_a is not None:
                a_act = layer._last_a[:, :d_act]  # [N, d_active]
                hebb_corr = torch.matmul(a_act.t(), salient_err_hidden) / max(n_valid, 1.0)  # [d_active, V]
                unlocked_row_gate = (~self.armor_locked_masks[l_idx][:d_act]).float().unsqueeze(1)
                B_updated = B_l_active + (self.eta_b * unlocked_row_gate) * hebb_corr
                B_l[:d_act, :].copy_(F.normalize(B_updated, p=2, dim=-1) * target_row_norm)

        if use_inplace_err:
            if scale_factor != 1.0:
                probs.mul_(scale_factor)
            e_scaled = probs
            if not (self.use_cubic_shield and self.attn_armor_locked):
                self.sparse_attention.local_spring_update(e_scaled)

        # 4. Update Output Head locally (Active-Slice execution or Cubic Anchor Shielded)
        d_last = self._final_layer_dim()
        if use_block_path:
            a_head = self.output_head._last_a_prev
            if self.use_fused_kernels:
                driving_head = torch.matmul(e_scaled.t(), a_head).div_(float(max(n_total, 1)))
                fused_spring_update_inplace(
                    w=self.output_head.w[:, :d_last],
                    w0=self.output_head.w0[:, :d_last],
                    k=self.output_head.k[:, :d_last],
                    driving_signal=driving_head,
                    eta_w=self.output_head.eta_w,
                    eta_0=self.output_head.eta_0,
                    gamma=self.output_head.gamma,
                    beta_k_eff=self.output_head.beta_k,
                    stiff_ceil=1.0,
                    active_row_gate=None,
                    cubic_shield=False,
                    scratch=self._kernel_scratch,
                    compensator=self._mp_compensator,
                    comp_key=id(self.output_head),
                )
            else:
                driving_head = torch.matmul(e_scaled.t(), a_head) / float(max(n_total, 1))
                w_h = self.output_head.w[:, :d_last]
                w0_h = self.output_head.w0[:, :d_last]
                k_h = self.output_head.k[:, :d_last]
                stretch_h = w_h - w0_h
                w_h.add_((self.output_head.eta_w * driving_head) - (self.output_head.gamma * k_h * stretch_h))
                stretch_h_new = w_h - w0_h
                w0_h.add_(self.output_head.eta_0 * (1.0 - k_h) * stretch_h_new)
                k_h.add_(self.output_head.beta_k * stretch_h_new.abs() * (1.0 - k_h)).clamp_(0.0, 1.0)
        elif self.continual_mode and self.use_cubic_shield:
            if self.output_head._last_a_prev is not None and d_last < self.d_max:
                self.output_head._last_a_prev[:, d_last:] = 0.0
            inactive_vocab = ~self.phase_active_vocab_mask
            e_head = e_scaled.clone()
            e_head[:, inactive_vocab] = 0.0
            self.output_head.local_spring_update(e_head)
            if self.locked_excl_vocab_mask.any():
                inactive_locked = self.locked_excl_vocab_mask & inactive_vocab
                self.output_head.w0[inactive_locked, :d_last].copy_(
                    self.head_w0_bedrock[inactive_locked, :d_last]
                )
                self.output_head.w[inactive_locked, :d_last].copy_(
                    self.head_w0_bedrock[inactive_locked, :d_last]
                )
        else:
            if self.output_head._last_a_prev is not None and d_last < self.d_max:
                self.output_head._last_a_prev[:, d_last:] = 0.0
            self.output_head.local_spring_update(e_scaled)

        # 5. Milestone 2.3: Unsupervised Local Spring Update of Role-Factored Bridges
        if (
            self.enable_hierarchical_compositional
            and self.use_role_bridge
            and (not self.continual_mode)
            and self._last_ent_slots is not None
            and self._last_role_probs is not None
        ):
            last_start_id, last_end_id, has_comp = self._last_ent_slots
            match_start = has_comp & (targets_flat == last_start_id) & (targets_flat != pad_id)
            match_end = (
                has_comp
                & (~match_start)
                & (targets_flat == last_end_id)
                & (inputs_flat == last_start_id)
                & (targets_flat != pad_id)
            )
            role_target = torch.where(
                match_start,
                torch.ones_like(targets_flat),
                torch.where(match_end, torch.full_like(targets_flat, 2), torch.zeros_like(targets_flat)),
            )

            role_train_mask = valid_mask * has_comp.float()
            n_role_valid = float(role_train_mask.sum().item())
            if n_role_valid > 0:
                role_onehot = F.one_hot(role_target, num_classes=3).float()
                role_class_w = torch.where(role_target > 0, 4.5, 1.0)
                role_err = (role_onehot - self._last_role_probs) * (role_train_mask * role_class_w).unsqueeze(-1)
                role_err = role_err * (float(n_total) / max(n_role_valid, 1.0))
                self.role_bridge_head.local_spring_update(role_err)

            ent_step_mask = (valid_mask > 0) & has_comp & (role_target > 0)
            n_ent = float(ent_step_mask.sum().item())
            if n_ent > 0 and self._last_ent_logits is not None:
                ent_probs = F.softmax(self._last_ent_logits, dim=-1)
                ent_mask_f = ent_step_mask.float().unsqueeze(-1)
                ent_probs.neg_()
                ent_probs.scatter_add_(1, targets_flat.unsqueeze(1), ent_mask_f)
                ent_probs.mul_(ent_mask_f * (float(n_total) / max(n_ent, 1.0)))
                emb_unit = F.normalize(self.sparse_attention.token_embedding, p=2, dim=-1, eps=1e-6)
                proj_err = torch.matmul(ent_probs, emb_unit) * ent_mask_f
                self.entity_proj.local_spring_update(proj_err)

        return {
            "ce_loss": ce_loss,
            "mse_loss": mse_loss,
            "correct": correct,
            "valid_tokens": int(n_valid),
        }

    @torch.no_grad()
    def _init_auxiliary_exploratory_slots(self, layer: DynamicSpringLayer, slot_indices: list[int], d_eff: int) -> None:
        """
        Initializes exploratory candidate slots with low-amplitude correlated pairs so that
        primary specialist cells carry the full Top-K representation while low-utility
        exploratory twins are cleanly fused (Stage 1) and pruned (Stage 2) during Sleep.
        """
        aux_scale = 0.02 / math.sqrt(self.d_max)
        base_vec = None
        for pos, idx in enumerate(slot_indices):
            noise = torch.randn(self.d_max, generator=self._rejuv_gen) * aux_scale
            if pos % 2 == 0 or base_vec is None:
                base_vec = noise
                row = base_vec.clone()
            else:
                row = 0.92 * base_vec + 0.39 * noise
            row[d_eff:] = 0.0
            layer.w[idx, :].copy_(row)
            layer.w0[idx, :].copy_(row)
            layer.k[idx, :].zero_()
            layer.win_counts[idx] = 0
            layer.epoch_win_counts[idx] = 0

    @torch.no_grad()
    def _rejuvenate_layer_slots_in_place(self, l_idx: int, max_to_rejuvenate: int) -> list[int]:
        """
        Milestone 2.2: In-Place Dormant Rejuvenation.
        Re-awakens recycled dormant slots in `self.rejuvenated_slots[l_idx]` in-place
        BEFORE increasing d_active.
        """
        slots = self.rejuvenated_slots[l_idx]
        if not slots or max_to_rejuvenate <= 0:
            return []

        take = min(len(slots), max_to_rejuvenate)
        rejuv_indices = [slots.pop(0) for _ in range(take)]
        layer = self.spring_layers[l_idx]
        d_eff = self._effective_context_dim()
        self._init_auxiliary_exploratory_slots(layer, rejuv_indices, d_eff)

        for idx in rejuv_indices:
            self.armor_locked_masks[l_idx][idx] = False
            self.layer_cell_phase_counts[l_idx][idx] = 0.0

        return rejuv_indices

    @torch.no_grad()
    def prepare_continual_phase(
        self,
        phase_token_sequences: list[list[int]],
        pad_id: int = 0,
    ) -> dict[str, int]:
        """
        Prepares the network for a new continual learning phase:
        1. Activates continual stream mode (`continual_stream_dim = 128`).
        2. Identifies the active vocabulary of the new phase (`phase_active_vocab_mask`).
        3. Rejuvenates recycled dormant slots in-place first, then recruits a bounded
           plastic sub-pool (+28 primary cells/layer) while isolating previously locked
           phase specialist cells in `layer_w0_bedrock`.
        """
        self.continual_mode = True
        self.current_epoch = 1
        self.phase_seen_targets.zero_()
        active_vocab = torch.zeros(self.vocab_size, dtype=torch.bool)
        for seq in phase_token_sequences:
            for tid in seq:
                if 0 <= tid < self.vocab_size and tid != pad_id:
                    active_vocab[tid] = True
        self.phase_active_vocab_mask.copy_(active_vocab)

        d_eff = self._effective_context_dim()
        total_rejuvenated = 0
        is_first_phase = not bool(self.head_armor_mask.any().item())

        for l_idx, layer in enumerate(self.spring_layers):
            layer.win_counts.zero_()
            layer.epoch_win_counts.zero_()
            layer._cooldown_counter = 0

            if is_first_phase:
                layer.d_active = 44
                layer.w[:32, :d_eff].copy_(layer._latent_init_pool[:32, :d_eff])
                layer.w[:32, d_eff:] = 0.0
                layer.w0[:32, :].copy_(layer.w[:32, :])
                self._init_auxiliary_exploratory_slots(layer, list(range(32, 44)), d_eff)
                layer.w[44:, :].zero_()
                layer.w0[44:, :].zero_()
                layer.k.zero_()
                self.current_phase_slots[l_idx] = list(range(44))
            else:
                rejuv_indices = self._rejuvenate_layer_slots_in_place(l_idx, max_to_rejuvenate=32)
                total_rejuvenated += len(rejuv_indices)
                old_d = layer.d_active
                new_d = min(self.d_max, old_d + 28)
                if new_d > old_d:
                    fresh_slice = layer._latent_init_pool[old_d:new_d, :].clone() * 1.15
                    fresh_slice[:, d_eff:] = 0.0
                    layer.w[old_d:new_d, :].copy_(fresh_slice)
                    layer.w0[old_d:new_d, :].copy_(fresh_slice)
                    layer.k[old_d:new_d, :].zero_()
                    layer.active_mask[old_d:new_d] = 1.0
                    layer.d_active = new_d
                self.current_phase_slots[l_idx] = rejuv_indices + list(range(old_d, new_d))

        if self._head_latent_cols is not None:
            if is_first_phase:
                init_cols = self._head_latent_cols[:, :d_eff]
                self.output_head.w[:, :d_eff] = init_cols
                self.output_head.w0[:, :d_eff] = init_cols
                self.output_head.w[:, d_eff:] = 0.0
                self.output_head.w0[:, d_eff:] = 0.0
                self.output_head.k.zero_()
            else:
                init_cols_phase = self._head_latent_cols[active_vocab, :d_eff]
                self.output_head.w[active_vocab, :d_eff] = init_cols_phase
                self.output_head.w0[active_vocab, :d_eff] = init_cols_phase
                self.output_head.w[active_vocab, d_eff:] = 0.0
                self.output_head.w0[active_vocab, d_eff:] = 0.0
                self.output_head.k[active_vocab, :].zero_()

        self._last_phase_rejuvenated = total_rejuvenated
        return {
            "active_phase_vocab": int(active_vocab.sum().item()),
            "total_rejuvenated": total_rejuvenated,
        }

    @torch.no_grad()
    def step_neurogenesis(self, epoch: int, mean_epoch_error: float, token_error_rate: float) -> list[dict]:
        """
        Triggers Dynamic Neurogenesis on saturation/frustration, prioritizing In-Place
        Rejuvenation of recycled dormant slots before expanding d_active.
        """
        self.current_epoch = epoch + 1
        if not self.enable_neurogenesis or self.frozen:
            return []
        if self.continual_mode:
            # In continual mode, per-phase rejuvenation + bounded recruitment is managed
            # deterministically by prepare_continual_phase.
            for l_idx, layer in enumerate(self.spring_layers):
                layer.epoch_win_counts.zero_()
                if self.enable_block_sparse:
                    self._last_epoch_block_wins[l_idx].copy_(self.epoch_block_win_counts[l_idx])
                    self.epoch_block_win_counts[l_idx].zero_()
            return []

        old_d_last = self._final_layer_dim()
        events = []
        for l_idx, layer in enumerate(self.spring_layers):
            ev = layer.check_and_trigger_neurogenesis(
                epoch=epoch,
                mean_epoch_error=mean_epoch_error,
                token_error_rate=token_error_rate,
            )
            if ev is not None:
                if self.enable_block_sparse:
                    layer.top_k = min(layer.d_active, self.hierarchical_top_k_blocks[l_idx] * self.block_size)
                elif self.enable_hierarchical_compositional:
                    layer.top_k = min(layer.d_active, self.hierarchical_top_ks[l_idx])
                ev["rejuvenated_in_place"] = 0
                events.append(ev)
                self.growth_log.append(ev)

        new_d_last = self._final_layer_dim()
        if new_d_last > old_d_last and self._head_latent_cols is not None:
            cols = self._head_latent_cols[:, old_d_last:new_d_last]
            unlocked_vocab = ~self.locked_excl_vocab_mask
            self.output_head.w[unlocked_vocab, old_d_last:new_d_last] = cols[unlocked_vocab]
            self.output_head.w0[unlocked_vocab, old_d_last:new_d_last] = cols[unlocked_vocab]
            self.output_head.k[:, old_d_last:new_d_last].zero_()
            if self.use_highway_anchored_dfa:
                self._sync_highway_anchored_dfa(init_only=False)

        if self.enable_block_sparse:
            for l_idx in range(self.num_layers):
                self._last_epoch_block_wins[l_idx].copy_(self.epoch_block_win_counts[l_idx])
                self.epoch_block_win_counts[l_idx].zero_()

        return events

    @torch.no_grad()
    def sleep_consolidate_and_recycle(
        self,
        phase_id: int,
        merge_cosine_threshold: float = 0.72,
        prune_fraction: float = 0.16,
        armor_stiffness: float = 0.94,
        shared_vocab_ids: set[int] | None = None,
        is_final_phase: bool = False,
    ) -> dict:
        """
        Milestone 2.2: Dual-Stage Offline Sleep Consolidation, Synaptic Recycling & Armor Locking.
        Executes between continual training phases:
          1. Stage 1 (Signal-Preserving High-Cosine Fusion): Fuses redundant twin cells (u, v)
             with high cosine similarity and recycles cell v into `rejuvenated_slots`.
          2. Stage 2 (Low-Utility Awakening Pruning): Prunes low-utility awakened cells
             (||w0_v||_2 * (k_v + 0.05) * log(2 + wins_v)) and returns them to `rejuvenated_slots`.
          3. Stage 3 (Sleep Armor Locking & Bedrock Unification): Snaps w <- w0, unites all
             consolidated phase specialist cells from `layer_w0_bedrock`, and locks k in [0.88, 0.96].
          4. Stage 4 (Orthogonal Subspace Bedrock Consolidation on Output Head): Consolidates
             both phase-exclusive and shared vocabulary anchors with 100% Zero-Replay retention.
        """
        d_eff = self._effective_context_dim()
        merged_per_layer: list[int] = []
        pruned_per_layer: list[int] = []
        locked_per_layer: list[int] = []
        self.phase_locked_slots[phase_id] = []

        for l_idx, layer in enumerate(self.spring_layers):
            d_act = layer.d_active
            wins = layer.win_counts[:d_act].float()
            unlocked_mask = ~self.armor_locked_masks[l_idx][:d_act]

            if self.current_phase_slots[l_idx]:
                candidate_pool = [
                    j for j in self.current_phase_slots[l_idx]
                    if j < d_act and unlocked_mask[j] and (j not in self.rejuvenated_slots[l_idx])
                ]
            else:
                candidate_pool = [
                    j for j in range(d_act)
                    if unlocked_mask[j] and (j not in self.rejuvenated_slots[l_idx])
                ]

            # Snap newly trained plastic cells of Phase p directly to their wakeful equilibrium w
            for j in candidate_pool:
                layer.w0[j, :d_eff].copy_(layer.w[j, :d_eff])
                layer.w0[j, d_eff:] = 0.0

            w0_act = layer.w0[:d_act, :d_eff]
            row_norms = w0_act.norm(p=2, dim=-1)
            plastic_indices = [j for j in candidate_pool if row_norms[j] > 1e-7]

            # Rank plastic cells by synaptic utility U_j = ||w0_j||_2 * (k_j + 0.05) * log(2 + wins_j)
            util_ranked = []
            for j in plastic_indices:
                r_norm = float(layer.w0[j, :d_eff].norm(p=2).item())
                k_mean = float(layer.k[j, :d_eff].mean().item())
                w_cnt = float(wins[j].item())
                util = r_norm * (k_mean + 0.05) * math.log(2.0 + w_cnt)
                util_ranked.append((util, j))
            util_ranked.sort(key=lambda x: x[0])

            # -----------------------------------------------------------------
            # Stage 1: Signal-Preserving High-Cosine Synaptic Fusion
            # -----------------------------------------------------------------
            merged_in_layer = 0
            merge_pool = [j for _, j in util_ranked[:12]] if len(util_ranked) >= 12 else plastic_indices
            if len(merge_pool) >= 6:
                p_tensor = torch.tensor(merge_pool, dtype=torch.long)
                w0_sub = F.normalize(w0_act[p_tensor], p=2, dim=-1)
                cos_mat = torch.matmul(w0_sub, w0_sub.t())
                cos_mat.fill_diagonal_(-1.0)

                used_in_merge: set[int] = set()
                max_merges = 3

                triu_u, triu_v = torch.triu_indices(len(merge_pool), len(merge_pool), offset=1)
                pair_scores = cos_mat[triu_u, triu_v]
                sorted_order = torch.argsort(pair_scores, descending=True)

                for pair_idx in sorted_order.tolist():
                    if merged_in_layer >= max_merges:
                        break
                    sim_val = float(pair_scores[pair_idx].item())
                    if sim_val < min(merge_cosine_threshold, 0.15):
                        break
                    u_local = int(triu_u[pair_idx].item())
                    v_local = int(triu_v[pair_idx].item())
                    u = merge_pool[u_local]
                    v = merge_pool[v_local]
                    if u in used_in_merge or v in used_in_merge:
                        continue

                    u_norm = float(layer.w0[u, :d_eff].norm(p=2).item())
                    fused_vec = 0.70 * layer.w0[u, :d_eff] + 0.30 * layer.w0[v, :d_eff]
                    fused_norm = float(fused_vec.norm(p=2).item())
                    if fused_norm > 1e-7:
                        layer.w0[u, :d_eff].copy_(fused_vec * (u_norm / fused_norm))
                        layer.w[u, :d_eff].copy_(layer.w0[u, :d_eff])

                    layer.w[v, :].zero_()
                    layer.w0[v, :].zero_()
                    layer.k[v, :].zero_()
                    layer.win_counts[v] = 0
                    if v not in self.rejuvenated_slots[l_idx]:
                        self.rejuvenated_slots[l_idx].append(v)

                    used_in_merge.add(u)
                    used_in_merge.add(v)
                    merged_in_layer += 1

            # -----------------------------------------------------------------
            # Stage 2: Low-Utility Awakening Pruning
            # -----------------------------------------------------------------
            remaining_plastic = [
                j for j in plastic_indices
                if j not in self.rejuvenated_slots[l_idx]
            ]
            pruned_in_layer = 0
            if len(remaining_plastic) >= 12:
                n_to_prune = 9
                utilities = []
                for j in remaining_plastic:
                    r_norm = float(layer.w0[j, :d_eff].norm(p=2).item())
                    k_mean = float(layer.k[j, :d_eff].mean().item())
                    w_cnt = float(wins[j].item())
                    util = r_norm * (k_mean + 0.05) * math.log(2.0 + w_cnt)
                    utilities.append((util, j))
                utilities.sort(key=lambda x: x[0])

                for _, j_prune in utilities[:n_to_prune]:
                    layer.w[j_prune, :].zero_()
                    layer.w0[j_prune, :].zero_()
                    layer.k[j_prune, :].zero_()
                    layer.win_counts[j_prune] = 0
                    if j_prune not in self.rejuvenated_slots[l_idx]:
                        self.rejuvenated_slots[l_idx].append(j_prune)
                    pruned_in_layer += 1

            # -----------------------------------------------------------------
            # Stage 3: Sleep Armor Locking & Bedrock Consolidation
            # -----------------------------------------------------------------
            surviving_new = sorted([
                j for j in remaining_plastic
                if j not in self.rejuvenated_slots[l_idx]
            ])
            for j in surviving_new:
                self.layer_w0_bedrock[l_idx][j, :d_eff].copy_(layer.w0[j, :d_eff])
                self.armor_locked_masks[l_idx][j] = True
                self.layer_cell_phase_counts[l_idx][j] = 1.0

            self.phase_locked_slots[phase_id].append(torch.tensor(surviving_new, dtype=torch.long))

            # Lock all consolidated specialist cells from layer_w0_bedrock into layer.w0
            locked_in_layer = 0
            for j in range(d_act):
                if self.armor_locked_masks[l_idx][j] and (j not in self.rejuvenated_slots[l_idx]):
                    layer.w0[j, :d_eff].copy_(self.layer_w0_bedrock[l_idx][j, :d_eff])
                    layer.w0[j, d_eff:] = 0.0
                    layer.w[j, :].copy_(layer.w0[j, :])
                    layer.k[j, :d_eff].fill_(armor_stiffness)
                    locked_in_layer += 1

            self.total_recycled_cells_per_layer[l_idx] += (merged_in_layer + pruned_in_layer)
            merged_per_layer.append(merged_in_layer)
            pruned_per_layer.append(pruned_in_layer)
            locked_per_layer.append(locked_in_layer)

        # Lock Dual-Horizon Attention projections after Phase 1 so the context space stays invariant
        for proj in (self.sparse_attention.q_proj, self.sparse_attention.k_proj, self.sparse_attention.v_proj):
            if not self.attn_armor_locked:
                proj.w0.copy_(proj.w)
            proj.w.copy_(proj.w0)
            proj.k.clamp_(min=0.92, max=0.96)
        self.attn_armor_locked = True

        # ---------------------------------------------------------------------
        # Stage 4: Output Head Orthogonal Subspace Bedrock Consolidation (OSBC)
        # ---------------------------------------------------------------------
        d_last = self._final_layer_dim()
        seen_mask = self.phase_active_vocab_mask | (self.phase_seen_targets > 0)
        if shared_vocab_ids is not None:
            shared_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            for sid in shared_vocab_ids:
                if 0 <= sid < self.vocab_size:
                    shared_mask[sid] = True
        else:
            shared_mask = self.token_salience < 1.18

        if self.shared_vocab_tensor is None:
            self.shared_vocab_tensor = torch.nonzero(shared_mask, as_tuple=False).squeeze(-1)

        new_excl_mask = seen_mask & (~shared_mask) & (~self.locked_excl_vocab_mask)
        new_excl_indices = torch.nonzero(new_excl_mask, as_tuple=False).squeeze(-1)

        # Snap newly trained rows in output_head directly to their wakeful equilibrium w
        for v_idx in torch.nonzero(seen_mask, as_tuple=False).squeeze(-1).tolist():
            self.output_head.w0[v_idx, :d_last].copy_(self.output_head.w[v_idx, :d_last])
            self.output_head.w0[v_idx, d_last:] = 0.0

        # Lock phase-exclusive rows into head_w0_bedrock
        if new_excl_indices.numel() > 0:
            self.head_w0_bedrock[new_excl_indices, :d_last].copy_(
                self.output_head.w0[new_excl_indices, :d_last]
            )
            self.output_head.w[new_excl_indices, :d_last].copy_(
                self.output_head.w0[new_excl_indices, :d_last]
            )
            self.output_head.k[new_excl_indices, :d_last].fill_(0.96)
            self.locked_excl_vocab_mask[new_excl_indices] = True
            self.vocab_phase_counts[new_excl_indices] = 1.0

        # Snapshot Phase p's shared vocabulary equilibrium for Resonant Subspace Routing
        if self.shared_vocab_tensor.numel() > 0:
            self.phase_shared_bedrock[phase_id] = (
                self.output_head.w0[self.shared_vocab_tensor, :d_last].clone()
            )

        # Orthogonal Subspace Bedrock Consolidation (OSBC) for shared vocabulary rows
        seen_shared_indices = torch.nonzero(seen_mask & shared_mask, as_tuple=False).squeeze(-1).tolist()
        for v_idx in seen_shared_indices:
            c_v = float(self.vocab_phase_counts[v_idx].item())
            curr_row = self.output_head.w0[v_idx, :d_last]
            if c_v == 0.0:
                self.head_w0_bedrock[v_idx, :d_last].copy_(curr_row)
                self.vocab_phase_counts[v_idx] = 1.0
            else:
                bed_row = self.head_w0_bedrock[v_idx, :d_last]
                bed_norm_sq = float(torch.dot(bed_row, bed_row).item()) + 1e-8
                proj_coeff = float(torch.dot(curr_row, bed_row).item()) / bed_norm_sq
                orth_vec = curr_row - (proj_coeff * bed_row)
                keep_coeff = max(1.0, min(1.15, proj_coeff))
                consolidated_row = (keep_coeff * bed_row) + (0.90 * orth_vec)
                self.head_w0_bedrock[v_idx, :d_last].copy_(consolidated_row)
                self.output_head.w0[v_idx, :d_last].copy_(consolidated_row)
                self.vocab_phase_counts[v_idx] = c_v + 1.0
            self.output_head.w[v_idx, :d_last].copy_(self.output_head.w0[v_idx, :d_last])
            self.output_head.k[v_idx, :d_last].fill_(0.90)

        # Restore all locked exclusive vocabulary rows directly from bedrock
        all_locked_excl = torch.nonzero(self.locked_excl_vocab_mask, as_tuple=False).squeeze(-1)
        if all_locked_excl.numel() > 0:
            self.output_head.w0[all_locked_excl, :d_last].copy_(
                self.head_w0_bedrock[all_locked_excl, :d_last]
            )
            self.output_head.w[all_locked_excl, :d_last].copy_(
                self.head_w0_bedrock[all_locked_excl, :d_last]
            )

        # Build Phase p's vocabulary mask and salience-weighted resonance profile
        self.phase_vocab_masks[phase_id] = seen_mask.clone()
        prof = torch.full((self.vocab_size,), -2.0)
        prof[:4] = 0.0  # Neutral weight on special tokens (<pad>, <bos>, <eos>, <unk>)
        excl_p_mask = seen_mask & (~shared_mask)
        shared_p_mask = seen_mask & shared_mask
        prof[excl_p_mask] = 5.0 * self.token_salience[excl_p_mask]
        prof[shared_p_mask] = 0.05 * self.token_salience[shared_p_mask]
        self.phase_resonance_profiles[phase_id] = prof

        self.head_armor_mask[seen_mask] = True

        sleep_record = {
            "phase_id": phase_id,
            "allocated_dims": self.get_allocated_dimensions(),
            "effective_active_dims": self.get_active_dimensions(),
            "merged_per_layer": merged_per_layer,
            "pruned_per_layer": pruned_per_layer,
            "recycled_this_sleep": sum(merged_per_layer) + sum(pruned_per_layer),
            "locked_per_layer": locked_per_layer,
            "rejuvenated_pool_sizes": [len(s) for s in self.rejuvenated_slots],
            "locked_excl_vocab_count": int(self.locked_excl_vocab_mask.sum().item()),
        }
        self.sleep_logs.append(sleep_record)
        return sleep_record

    @staticmethod
    def _normalize_active_eligibility(tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalizes non-zero magnitudes |E_ij| into [0, 1] so selective thermal k-melting
        and reward locking act sharply on offending/endorsed synapses while zero-eligibility
        general knowledge synapses stay exactly at 0.0.
        """
        abs_t = tensor.abs()
        nz = abs_t[abs_t > 1e-8]
        if nz.numel() == 0:
            return torch.zeros_like(abs_t)
        max_val = float(nz.max().item())
        mean_val = float(nz.mean().item())
        ref = max(0.20 * max_val + 0.80 * (1.5 * mean_val), 1e-6)
        return (abs_t / ref).clamp(0.0, 1.0)

    def _build_decision_horizon_weights(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        prefix_lens: torch.Tensor | list[int] | int,
        pad_id: int = 0,
        gamma_elig: float = 0.96,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Law 1 (Milestone 2.5): Decision-Horizon & Salience-Filtered Eligibility Weighting.
        Masks out all prefix steps (tau <= T_prefix) so context grammar is never penalized,
        and weights continuation decisions by gamma_elig^(T - tau) * salience(x_tau).
        """
        bsz, seq_len = target_ids.shape
        device = target_ids.device
        if isinstance(prefix_lens, int):
            p_list = [prefix_lens] * bsz
        elif isinstance(prefix_lens, torch.Tensor):
            p_list = [int(x) for x in prefix_lens.tolist()]
        else:
            p_list = [int(x) for x in prefix_lens]

        horizon_w = torch.zeros(bsz, seq_len, dtype=torch.float32, device=device)
        for b in range(bsz):
            valid_pos = torch.nonzero(
                (target_ids[b] != pad_id) & (target_ids[b] != 1) & (target_ids[b] != 2),
                as_tuple=False,
            ).squeeze(-1)
            if valid_pos.numel() == 0:
                continue
            last_t = int(valid_pos[-1].item())
            start_t = max(0, p_list[b] - 1)
            valid_dec = valid_pos[valid_pos >= start_t]
            if valid_dec.numel() == 0:
                continue
            dists = (last_t - valid_dec).clamp(min=0).float()
            horizon_w[b, valid_dec] = float(gamma_elig) ** dists

        inputs_flat = input_ids.reshape(bsz * seq_len)
        targets_flat = target_ids.reshape(bsz * seq_len)
        horizon_flat = horizon_w.reshape(bsz * seq_len)
        dec_mask = horizon_flat > 0.0

        target_sal = self.token_salience[targets_flat]
        input_sal = self.token_salience[inputs_flat]
        step_salience = (0.75 * target_sal + 0.25 * input_sal).clamp(0.70, 1.80)
        elig_weights = horizon_flat * step_salience
        return elig_weights, dec_mask

    @torch.no_grad()
    def _apply_three_factor_pass(
        self,
        seq_ids: torch.Tensor,
        prefix_lens: torch.Tensor | list[int] | int,
        sample_rewards: torch.Tensor,
        token_filter_mask: torch.Tensor | None = None,
        pad_id: int = 0,
        gamma_elig: float = 0.96,
        eta_melt: float = 0.82,
        eta_lock: float = 0.25,
        eta_align_w: float = 1.6,
        eta_align_0: float = 0.55,
        eta_head_w: float = 7.5,
        eta_head_0: float = 0.78,
    ) -> dict[str, float]:
        """
        Executes a single unified Reward-Modulated Three-Factor Spring Plasticity pass
        (Laws 1, 2, 3 & 4 of Milestone 2.5) across Dynamic/Block-Sparse hidden layers
        and Output Head without autograd or any critic network.
        """
        bsz = seq_ids.shape[0]
        if seq_ids.shape[1] < 2:
            return {"valid_decisions": 0, "ce_loss": 0.0}

        input_ids = seq_ids[:, :-1]
        target_ids = seq_ids[:, 1:]
        seq_len = input_ids.shape[1]
        n_total = bsz * seq_len

        self._is_training_step = False
        logits = self.forward(input_ids)
        logits_flat = logits.reshape(n_total, self.vocab_size)
        targets_flat = target_ids.reshape(n_total)

        elig_weights, dec_mask = self._build_decision_horizon_weights(
            input_ids=input_ids,
            target_ids=target_ids,
            prefix_lens=prefix_lens,
            pad_id=pad_id,
            gamma_elig=gamma_elig,
        )
        if token_filter_mask is not None:
            filter_flat = token_filter_mask.reshape(n_total).float()
            eff_elig_weights = elig_weights * filter_flat
            eff_dec_mask = dec_mask & (filter_flat > 0.0)
        else:
            eff_elig_weights = elig_weights
            eff_dec_mask = dec_mask

        n_dec = max(int(eff_dec_mask.sum().item()), 1)
        if int(eff_dec_mask.sum().item()) == 0:
            return {"valid_decisions": 0, "ce_loss": 0.0}

        r_per_tok = sample_rewards.unsqueeze(1).expand(bsz, seq_len).reshape(n_total)
        pos_gate = (r_per_tok > 0.0).float()
        neg_gate = (r_per_tok < 0.0).float()

        probs = F.softmax(logits_flat, dim=-1)
        y_onehot = F.one_hot(targets_flat, num_classes=self.vocab_size).float()

        # Token-level selection residual for R > 0 vs R < 0
        pos_scale = (r_per_tok.clamp(min=0.0) * eff_elig_weights * pos_gate).unsqueeze(-1)
        err_pos = (y_onehot - probs) * pos_scale

        p_chosen = (probs * y_onehot).sum(dim=-1, keepdim=True)
        p_other = probs * (1.0 - y_onehot)
        p_alt = p_other / p_other.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        neg_scale = ((-r_per_tok).clamp(min=0.0) * eff_elig_weights * neg_gate).unsqueeze(-1)
        err_neg = (p_alt - y_onehot) * p_chosen * neg_scale

        err_total = err_pos + err_neg

        has_neg = bool(((neg_gate > 0.0) & eff_dec_mask).any().item())
        has_pos = bool(((pos_gate > 0.0) & eff_dec_mask).any().item())
        mean_r_neg = (
            float((-r_per_tok[(neg_gate > 0.0) & eff_dec_mask]).abs().mean().item()) if has_neg else 0.0
        )
        mean_r_pos = (
            float(r_per_tok[(pos_gate > 0.0) & eff_dec_mask].mean().item()) if has_pos else 0.0
        )

        # Decision-horizon CE & chosen probability telemetry
        dec_logits = logits_flat[eff_dec_mask]
        dec_targets = targets_flat[eff_dec_mask]
        ce_loss = float(F.cross_entropy(dec_logits, dec_targets).item())
        mean_chosen_prob = float(p_chosen[eff_dec_mask].mean().item())

        # Shared hidden-layer Three-Factor Plasticity loop (Block-Sparse & Element-Wise)
        use_block_path = self.enable_block_sparse and (not self.continual_mode)
        depth_dfa_scale = (
            ((4.0 / float(max(self.num_layers, 4))) ** 0.25)
            if (use_block_path and self.use_depth_scaled_highway and self.num_layers > 4)
            else 1.0
        )

        k_hid_rej_before: list[float] = []
        k_hid_rej_after: list[float] = []
        k_hid_pref_before: list[float] = []
        k_hid_pref_after: list[float] = []

        e_hwy_neg = e_hwy_tot = e_hwy_pos = None
        if self.factorized_dfa is not None:
            head_w_act_hwy = self.output_head.w[:, : self._final_layer_dim()]
            e_hwy_neg = self.factorized_dfa.project_vocab_to_highway(err_neg, head_w_act_hwy) if has_neg else None
            e_hwy_tot = self.factorized_dfa.project_vocab_to_highway(err_total, head_w_act_hwy)
            e_hwy_pos = self.factorized_dfa.project_vocab_to_highway(err_pos, head_w_act_hwy) if has_pos else None

        for l_idx, layer in enumerate(self.spring_layers):
            d_act = layer.d_active
            a_prev = layer._last_a_prev
            if a_prev is None:
                continue
            in_dim = a_prev.shape[1]
            z_act = layer._last_z[:, :d_act]
            raw_mask = layer._last_mask
            if use_block_path:
                cell_mask = raw_mask.repeat_interleave(self.block_size, dim=1)
                n_blocks = max(1, d_act // self.block_size)
                k_eff = (
                    min(self.hierarchical_top_k_blocks[l_idx], n_blocks) * self.block_size
                    if self.enable_hierarchical_compositional
                    else (layer.top_k or d_act)
                )
            else:
                cell_mask = raw_mask[:, :d_act]
                k_eff = layer.top_k or d_act

            g_prime = layer._activate_derivative(z_act)
            active_cell = g_prime * cell_mask  # [N, d_act]
            B_l_act = None if self.factorized_dfa is not None else self.dfa_router.get_matrix(l_idx)[:d_act, :]

            wins_per_cell = cell_mask.sum(dim=0).clamp(min=1.0).unsqueeze(1)
            avg_dec_per_seq = float(n_dec) / float(max(bsz, 1))
            denom = (0.35 * avg_dec_per_seq) + (0.65 * wins_per_cell)

            w_act = layer.w[:d_act, :in_dim]
            w0_act = layer.w0[:d_act, :in_dim]
            k_act = layer.k[:d_act, :in_dim]

            # 1. Negative Pass (R < 0): Selective Stiffness Melting on rejected synapses
            elig_neg_l = None
            if has_neg:
                if self.factorized_dfa is not None:
                    proj_neg = self.factorized_dfa.broadcast_layer_error(
                        l_idx, e_hwy_neg, d_act, depth_dfa_scale
                    )
                else:
                    proj_neg = F.linear(err_neg, B_l_act) * depth_dfa_scale
                delta_neg = proj_neg * active_cell
                elig_neg_l = torch.matmul(delta_neg.t(), a_prev) / denom
                norm_e_neg = self._normalize_active_eligibility(elig_neg_l)
                melt_mask = norm_e_neg > 0.05
                if melt_mask.any():
                    k_hid_rej_before.append(float(k_act[melt_mask].mean().item()))
                melt_frac = (eta_melt * mean_r_neg * norm_e_neg).clamp(0.0, 0.92)
                k_act.mul_(1.0 - melt_mask.float() * melt_frac)
                if melt_mask.any():
                    k_hid_rej_after.append(float(k_act[melt_mask].mean().item()))

            # 2. Combined Elastic Push & Anchor Deposition into W_0 (shielded by surviving k)
            # Normalize by mean squared activation norm so step scale is invariant to d_act / d_last.
            mean_a2_hid = max(float(a_prev.pow(2).sum(dim=-1).mean().item()), 1.0)
            if self.factorized_dfa is not None:
                proj_tot = self.factorized_dfa.broadcast_layer_error(
                    l_idx, e_hwy_tot, d_act, depth_dfa_scale
                )
            else:
                proj_tot = F.linear(err_total, B_l_act) * depth_dfa_scale
            delta_tot = proj_tot * active_cell
            elig_tot_l = torch.matmul(delta_tot.t(), a_prev) / denom
            plastic_gate_hid = (1.0 - k_act).clamp(min=0.05).pow(2.5)
            stretch = w_act - w0_act
            delta_w = ((eta_align_w / mean_a2_hid) * plastic_gate_hid * elig_tot_l) - (layer.gamma * k_act * stretch)
            w_act.add_(delta_w)
            stretch_new = w_act - w0_act
            w0_act.add_(eta_align_0 * (1.0 - k_act) * stretch_new)

            # 3. Positive Pass (R > 0): Reward Locking on aligned synapses
            if has_pos:
                if self.factorized_dfa is not None:
                    proj_pos = self.factorized_dfa.broadcast_layer_error(
                        l_idx, e_hwy_pos, d_act, depth_dfa_scale
                    )
                else:
                    proj_pos = F.linear(err_pos, B_l_act) * depth_dfa_scale
                delta_pos = proj_pos * active_cell
                elig_pos_l = torch.matmul(delta_pos.t(), a_prev) / denom
                norm_e_pos = self._normalize_active_eligibility(elig_pos_l)
                lock_mask = norm_e_pos > 0.05
                if lock_mask.any():
                    k_hid_pref_before.append(float(k_act[lock_mask].mean().item()))
                stiff_ceil = (
                    self.layer_stiff_ceilings[l_idx]
                    if self.enable_hierarchical_compositional
                    else 0.85
                )
                delta_k = eta_lock * mean_r_pos * norm_e_pos * (1.0 - k_act)
                k_act.add_(delta_k).clamp_(0.0, stiff_ceil)
                if lock_mask.any():
                    k_hid_pref_after.append(float(k_act[lock_mask].mean().item()))

        # Output Head Three-Factor Plasticity (Melt rejected vocab links -> Consolidate & Lock preferred)
        d_last = self._final_layer_dim()
        a_head_raw = self.output_head._last_a_prev
        a_head = a_head_raw[:, :d_last] if a_head_raw.shape[1] != d_last else a_head_raw
        w_h = self.output_head.w[:, :d_last]
        w0_h = self.output_head.w0[:, :d_last]
        k_h = self.output_head.k[:, :d_last]

        # Batch-size-invariant per-token occurrence normalization for output_head rows
        avg_dec_per_seq = float(n_dec) / float(max(bsz, 1))
        target_hits_v = torch.zeros(self.vocab_size, dtype=torch.float32, device=seq_ids.device)
        target_hits_v.scatter_add_(
            0,
            dec_targets,
            torch.ones(dec_targets.numel(), dtype=torch.float32, device=seq_ids.device),
        )
        denom_h = (0.20 * avg_dec_per_seq) + (0.80 * target_hits_v.clamp(min=1.0).unsqueeze(1))

        # Track untouched vocab rows (zero negative & zero positive token presence)
        dec_vocab_active = torch.zeros(self.vocab_size, dtype=torch.bool, device=seq_ids.device)
        dec_vocab_active[dec_targets] = True
        untouched_vocab = ~dec_vocab_active
        untouched_k_before = float(k_h[untouched_vocab].mean().item()) if untouched_vocab.any() else 0.0

        k_head_rej_before = 0.0
        k_head_rej_after = 0.0
        k_head_pref_before = 0.0
        k_head_pref_after = 0.0

        if has_neg:
            # Melt ONLY synapses connected to actually executed rejected actions (zero spillover to general vocab)
            elig_head_neg = torch.matmul((y_onehot * neg_scale * p_chosen.clamp(min=0.10)).t(), a_head) / denom_h
            norm_eh_neg = self._normalize_active_eligibility(elig_head_neg)
            melt_h_mask = norm_eh_neg > 0.05
            if melt_h_mask.any():
                k_head_rej_before = float(k_h[melt_h_mask].mean().item())
            melt_h = (eta_melt * mean_r_neg * norm_eh_neg).clamp(0.0, 0.92)
            k_h.mul_(1.0 - melt_h_mask.float() * melt_h)
            if melt_h_mask.any():
                k_head_rej_after = float(k_h[melt_h_mask].mean().item())

        mean_a2_head = max(float(a_head.pow(2).sum(dim=-1).mean().item()), 1.0)
        elig_head_tot = torch.matmul(err_total.t(), a_head) / denom_h
        plastic_gate_head = (1.0 - k_h).clamp(min=0.04).pow(2.0)
        stretch_h = w_h - w0_h
        w_h.add_(((eta_head_w / mean_a2_head) * plastic_gate_head * elig_head_tot) - (self.output_head.gamma * k_h * stretch_h))
        stretch_h_new = w_h - w0_h
        w0_h.add_(eta_head_0 * (1.0 - k_h) * stretch_h_new)

        if has_pos:
            # Lock ONLY synapses connected to actually executed preferred actions (never lock rejected/general rows)
            elig_head_pos = torch.matmul((y_onehot * pos_scale).t(), a_head) / denom_h
            norm_eh_pos = self._normalize_active_eligibility(elig_head_pos)
            lock_h_mask = norm_eh_pos > 0.05
            if lock_h_mask.any():
                k_head_pref_before = float(k_h[lock_h_mask].mean().item())
            k_h.add_(eta_lock * mean_r_pos * norm_eh_pos * (1.0 - k_h)).clamp_(0.0, 0.92)
            if lock_h_mask.any():
                k_head_pref_after = float(k_h[lock_h_mask].mean().item())

            if self.use_highway_anchored_dfa:
                self._sync_highway_anchored_dfa(init_only=False)

        untouched_k_after = float(k_h[untouched_vocab].mean().item()) if untouched_vocab.any() else 0.0

        return {
            "valid_decisions": n_dec,
            "ce_loss": ce_loss,
            "mean_chosen_prob": mean_chosen_prob,
            "k_head_rej_before": k_head_rej_before,
            "k_head_rej_melted": k_head_rej_after,
            "k_head_pref_before": k_head_pref_before,
            "k_head_pref_locked": k_head_pref_after,
            "k_hid_rej_before": sum(k_hid_rej_before) / max(len(k_hid_rej_before), 1),
            "k_hid_rej_melted": sum(k_hid_rej_after) / max(len(k_hid_rej_after), 1),
            "k_hid_pref_before": sum(k_hid_pref_before) / max(len(k_hid_pref_before), 1),
            "k_hid_pref_locked": sum(k_hid_pref_after) / max(len(k_hid_pref_after), 1),
            "k_untouched_head_before": untouched_k_before,
            "k_untouched_head_after": untouched_k_after,
            "layer_normed_rms": list(self._last_layer_normed_rms),
        }

    @torch.no_grad()
    def align_with_reward_modulation(
        self,
        seq_ids: torch.Tensor | None = None,
        prefix_lens: torch.Tensor | list[int] | int = 1,
        rewards: torch.Tensor | list[float] | float = 0.0,
        preferred_seqs: torch.Tensor | None = None,
        rejected_seqs: torch.Tensor | None = None,
        pad_id: int = 0,
        gamma_elig: float = 0.96,
        eta_melt: float = 0.82,
        eta_lock: float = 0.25,
        eta_align_w: float = 1.6,
        eta_align_0: float = 0.55,
        eta_head_w: float = 7.5,
        eta_head_0: float = 0.78,
        use_reward_baseline: bool = True,
        mask_shared_tokens_in_pairs: bool = True,
    ) -> dict[str, float]:
        """
        Milestone 2.5 Post-Training Behavioral Alignment (Critic-Free, Zero-Autograd):
        Accepts either:
          (a) Contrastive behavioral preference pairs (`preferred_seqs` with R=+1 vs
              `rejected_seqs` with R=-1 for the same context prefixes), or
          (b) Individual rollout sequences (`seq_ids` with scalar rewards `rewards` in [-1, +1]
              centered against the balanced EMA `reward_baseline`).
        All aligned behavior consolidates directly into equilibrium anchors `W_0`.
        """
        if self.frozen:
            raise RuntimeError("Model is already frozen; run alignment before production_freeze().")

        if preferred_seqs is not None and rejected_seqs is not None:
            bsz = preferred_seqs.shape[0]
            if isinstance(prefix_lens, int):
                p_list = [prefix_lens] * bsz
            elif isinstance(prefix_lens, torch.Tensor):
                p_list = [int(x) for x in prefix_lens.tolist()]
            else:
                p_list = [int(x) for x in prefix_lens]

            rej_filter = None
            if mask_shared_tokens_in_pairs:
                t_rej = rejected_seqs.shape[1] - 1
                rej_filter = torch.ones(bsz, t_rej, dtype=torch.float32, device=rejected_seqs.device)
                rej_targets = rejected_seqs[:, 1:]
                for b in range(bsz):
                    pref_cont = set(
                        int(t)
                        for t in preferred_seqs[b, p_list[b]:].tolist()
                        if t not in (pad_id, 1, 2)
                    )
                    start_t = max(0, p_list[b] - 1)
                    for t in range(start_t, t_rej):
                        tid = int(rej_targets[b, t].item())
                        if tid == pad_id:
                            continue
                        if tid in pref_cont:
                            rej_filter[b, t] = 0.0

            neg_rewards = torch.full((bsz,), -1.0, dtype=torch.float32, device=rejected_seqs.device)
            rej_stats = self._apply_three_factor_pass(
                seq_ids=rejected_seqs,
                prefix_lens=p_list,
                sample_rewards=neg_rewards,
                token_filter_mask=rej_filter,
                pad_id=pad_id,
                gamma_elig=gamma_elig,
                eta_melt=eta_melt,
                eta_lock=eta_lock,
                eta_align_w=eta_align_w,
                eta_align_0=eta_align_0,
                eta_head_w=eta_head_w,
                eta_head_0=eta_head_0,
            )

            pos_rewards = torch.full((bsz,), 1.0, dtype=torch.float32, device=preferred_seqs.device)
            pref_stats = self._apply_three_factor_pass(
                seq_ids=preferred_seqs,
                prefix_lens=p_list,
                sample_rewards=pos_rewards,
                token_filter_mask=None,
                pad_id=pad_id,
                gamma_elig=gamma_elig,
                eta_melt=eta_melt,
                eta_lock=eta_lock,
                eta_align_w=eta_align_w,
                eta_align_0=eta_align_0,
                eta_head_w=eta_head_w,
                eta_head_0=eta_head_0,
            )

            stats = {
                "mode": "contrastive_pair",
                "valid_decisions": rej_stats["valid_decisions"] + pref_stats["valid_decisions"],
                "pref_ce_loss": pref_stats["ce_loss"],
                "mean_pref_prob": pref_stats["mean_chosen_prob"],
                "mean_rej_prob": rej_stats["mean_chosen_prob"],
                "k_head_rej_before": rej_stats["k_head_rej_before"],
                "k_head_rej_melted": rej_stats["k_head_rej_melted"],
                "k_head_pref_before": pref_stats["k_head_pref_before"],
                "k_head_pref_locked": pref_stats["k_head_pref_locked"],
                "k_hid_rej_before": rej_stats["k_hid_rej_before"],
                "k_hid_rej_melted": rej_stats["k_hid_rej_melted"],
                "k_hid_pref_before": pref_stats["k_hid_pref_before"],
                "k_hid_pref_locked": pref_stats["k_hid_pref_locked"],
                "k_untouched_head_before": rej_stats["k_untouched_head_before"],
                "k_untouched_head_after": pref_stats["k_untouched_head_after"],
                "layer_normed_rms": pref_stats["layer_normed_rms"],
            }
            self.alignment_logs.append(stats)
            return stats

        if seq_ids is None:
            raise ValueError("Provide either `seq_ids` + `rewards` or (`preferred_seqs`, `rejected_seqs`).")

        bsz = seq_ids.shape[0]
        if isinstance(rewards, (int, float)):
            r_tensor = torch.full((bsz,), float(rewards), dtype=torch.float32, device=seq_ids.device)
        elif isinstance(rewards, torch.Tensor):
            r_tensor = rewards.to(dtype=torch.float32, device=seq_ids.device)
        else:
            r_tensor = torch.tensor(rewards, dtype=torch.float32, device=seq_ids.device)

        if use_reward_baseline:
            r_adv = (r_tensor - self.reward_baseline).clamp(-1.5, 1.5)
            batch_mean_r = float(r_tensor.mean().item())
            self.reward_baseline = max(-0.30, min(0.30, 0.90 * self.reward_baseline + 0.10 * batch_mean_r))
        else:
            r_adv = r_tensor.clamp(-1.5, 1.5)

        stats = self._apply_three_factor_pass(
            seq_ids=seq_ids,
            prefix_lens=prefix_lens,
            sample_rewards=r_adv,
            token_filter_mask=None,
            pad_id=pad_id,
            gamma_elig=gamma_elig,
            eta_melt=eta_melt,
            eta_lock=eta_lock,
            eta_align_w=eta_align_w,
            eta_align_0=eta_align_0,
            eta_head_w=eta_head_w,
            eta_head_0=eta_head_0,
        )
        stats["mode"] = "scalar_reward"
        self.alignment_logs.append(stats)
        return stats

    @torch.no_grad()
    def production_freeze(self, prune_dormant: bool = True) -> dict:
        """
        Final consolidation, synchronous compaction & production export:
        1. Snaps plastic anchors toward converged equilibrium w where stiffness permits.
        2. Physically compacts out any recycled/pruned dormant rows in each layer's W0
           while preserving exact residual coordinate alignment via `_compacted_indices`.
        3. Destroys w, k, and adaptive feedback matrices B^(l) completely so only W_final = W_0 remains.
        """
        self.frozen = True

        if not self.continual_mode:
            for l_idx, layer in enumerate(self.spring_layers):
                d_act = layer.d_active
                unlocked = ~self.armor_locked_masks[l_idx][:d_act]
                for j in range(d_act):
                    if unlocked[j] and j not in self.rejuvenated_slots[l_idx]:
                        layer.w0[j, :].lerp_(layer.w[j, :], 0.25)

            unlocked_vocab = ~self.locked_excl_vocab_mask
            self.output_head.w0[unlocked_vocab].lerp_(self.output_head.w[unlocked_vocab], 0.30)

        self.role_bridge_head.w0.lerp_(self.role_bridge_head.w, 0.45)
        self.role_bridge_head.production_freeze()
        self.entity_proj.w0.lerp_(self.entity_proj.w, 0.30)
        self.entity_proj.production_freeze()

        self.sparse_attention.production_freeze()
        self.sparse_attention.B_attn = None

        d_eff = self._effective_context_dim()
        d_last = self._final_layer_dim()

        layer_prune_stats = []
        for l_idx, layer in enumerate(self.spring_layers):
            d_act = layer.d_active
            recycled_set = set(self.rejuvenated_slots[l_idx])
            if self.enable_block_sparse and (not self.continual_mode):
                kept_indices = list(range(d_act))
            else:
                kept_indices = [
                    j for j in range(d_act)
                    if (j not in recycled_set) and (layer.w0[j, :d_eff].norm(p=2).item() > 1e-7)
                ]
                if not kept_indices:
                    kept_indices = list(range(d_act))

            orig_to_compacted = {orig_j: c_pos for c_pos, orig_j in enumerate(kept_indices)}
            for pid, p_slots in self.phase_locked_slots.items():
                if l_idx < len(p_slots):
                    mapped = [
                        orig_to_compacted[int(s.item())]
                        for s in p_slots[l_idx]
                        if int(s.item()) in orig_to_compacted
                    ]
                    self.phase_compacted_slots.setdefault(pid, {})[l_idx] = torch.tensor(
                        mapped, dtype=torch.long, device=layer.w0.device
                    )

            stats = layer.production_freeze(prune_dormant=prune_dormant, prev_active_dim=d_eff)
            if prune_dormant and len(kept_indices) < d_act:
                kept_tensor = torch.tensor(kept_indices, dtype=torch.long, device=layer.w0.device)
                compacted_w0 = layer.w0[kept_tensor, :d_eff].contiguous()
                layer.register_buffer("w0", compacted_w0)
                layer.register_buffer("_compacted_indices", kept_tensor)
                stats["active_cells"] = len(kept_indices)
                stats["dormant_pruned"] = self.d_max - len(kept_indices)
                stats["pruned_ratio_pct"] = (stats["dormant_pruned"] / float(self.d_max)) * 100.0
            else:
                layer._compacted_indices = None

            layer_prune_stats.append(stats)

        if self.factorized_dfa is not None:
            self.factorized_dfa.freeze()
        elif self.dfa_router is not None:
            for l_idx in range(self.num_layers):
                setattr(self.dfa_router, f"B_{l_idx}", None)

        if prune_dormant and self.enable_block_sparse and (not self.continual_mode):
            self._fast_d_eff = d_eff
            self.register_buffer("cap_proj", self.cap_proj[:d_eff, :].contiguous())
            self.register_buffer("lag1_proj", self.lag1_proj[:d_eff, :].contiguous())
            self.register_buffer("lag2_proj", self.lag2_proj[:d_eff, :].contiguous())
            self.reset_frozen_block_stats()

        self.output_head.frozen = True
        pruned_head_w0 = self.output_head.w0[:, :d_last].contiguous().clone()
        self.output_head.register_buffer("w0", pruned_head_w0)
        self.output_head.w = None
        self.output_head.k = None
        self._head_latent_cols = None

        total_active_cells = sum(s["active_cells"] for s in layer_prune_stats)
        total_max_cells = self.num_layers * self.d_max
        saved_cells = total_max_cells - total_active_cells
        saved_pct = (saved_cells / float(total_max_cells)) * 100.0

        return {
            "active_dims": [s["active_cells"] for s in layer_prune_stats],
            "allocated_dims": [layer.d_active for layer in self.spring_layers],
            "total_active_cells": total_active_cells,
            "total_max_cells": total_max_cells,
            "dormant_cells_saved": saved_cells,
            "dormant_saved_pct": saved_pct,
            "total_recycled_across_sleeps": sum(self.total_recycled_cells_per_layer),
        }

    @torch.no_grad()
    def init_inference_cache(self, batch_size: int = 1) -> SpringInferenceCache:
        """
        Allocates a constant-time O(1) Stateful Inference Cache holding rolling K/V/Salience
        attention state, Salience-Gated EMA Capacitor, Exponential ALiBi accumulator,
        lag-1/2 embeddings, Role-Bridge Entity Slots, and O(1) N-gram Refractory Gate tables.
        """
        use_active_slice = (
            self.enable_block_sparse or (self._fast_d_eff is not None)
        ) and (not self.continual_mode)
        context_dim = self._effective_context_dim() if use_active_slice else self.d_max
        cache = SpringInferenceCache(
            batch_size=batch_size,
            max_seq_len=self.max_seq_len,
            d_max=self.d_max,
            context_dim=context_dim,
            device=self.sparse_attention.token_embedding.device,
            dtype=self.sparse_attention.token_embedding.dtype,
        )
        if self.frozen and self.enable_hierarchical_compositional and self.use_role_bridge:
            cache._emb_unit = F.normalize(
                self.sparse_attention.token_embedding, p=2, dim=-1, eps=1e-6
            )
        else:
            cache._emb_unit = None
        return cache

    @torch.no_grad()
    def forward_step_cached(
        self,
        token_id: int,
        cache: SpringInferenceCache,
        compute_logits: bool = True,
    ) -> torch.Tensor | None:
        """
        Executes a single causal token step using `SpringInferenceCache` in O(1) state updates
        with mathematical parity to full-prefix recomputation (`self.forward(window)`).
        When `compute_logits=False` (prompt prefix warmup), skips Q projection, softmax,
        hidden layers, and output head while faithfully advancing all causal states.
        """
        t = cache.step
        if t >= self.max_seq_len:
            raise IndexError(
                f"SpringInferenceCache step {t} exceeds max_seq_len {self.max_seq_len}."
            )

        tok_emb = self.sparse_attention.token_embedding[token_id].unsqueeze(0)  # [1, D_max]
        pos_t = self.sparse_attention.pos_embedding[t].unsqueeze(0)              # [1, D_max]
        x_t = tok_emb + pos_t                                                    # [1, D_max]

        cache.k_cache[:, t, :].copy_(self.sparse_attention.k_proj(x_t))
        cache.v_cache[:, t, :].copy_(self.sparse_attention.v_proj(x_t))
        sal_val = self.sparse_attention.token_salience[token_id]
        cache.sal_cache[:, t].fill_(sal_val)

        # Salience-gated EMA Capacitor update
        alpha_t = float(self.capacitor_alpha[token_id].item())
        cache.c_state.mul_(1.0 - alpha_t).add_(tok_emb, alpha=alpha_t)

        # Exponential O(1) ALiBi accumulator (matches _build_vectorized_context bit-for-bit)
        use_sliding = self.enable_hierarchical_compositional and self.use_role_bridge and (not self.continual_mode)
        use_active_slice = (self.enable_block_sparse or (self._fast_d_eff is not None)) and (not self.continual_mode)
        d_eff = self._effective_context_dim()
        out_dim = d_eff if use_active_slice else self.d_max

        x_pos_t = tok_emb[:, :out_dim] + self.alibi_pos_encoding[t, :out_dim].unsqueeze(0)
        alibi_ctx = None
        if use_sliding:
            cache.alibi_u.mul_(0.60).add_(x_pos_t)
            cache.alibi_denom_base = cache.alibi_denom_base * 0.60 + 1.0
            if compute_logits:
                alibi_ctx = cache.alibi_u / max(cache.alibi_denom_base, 1e-6)
        else:
            cache.alibi_u.mul_(0.68).add_(x_pos_t)
            cache.alibi_denom_base = cache.alibi_denom_base * 0.68 + 1.0
            if t == 0:
                cache.x_pos_0.copy_(x_pos_t)
            elif t == 1:
                cache.x_pos_1.copy_(x_pos_t)
            if compute_logits:
                if t == 0:
                    alibi_num = cache.alibi_u + 0.30 * cache.x_pos_0
                    denom = cache.alibi_denom_base + 0.30
                else:
                    alibi_num = cache.alibi_u + 0.30 * cache.x_pos_0 + 0.20 * cache.x_pos_1
                    denom = cache.alibi_denom_base + 0.50
                alibi_ctx = alibi_num / max(denom, 1e-6)

        # Causal Role-Bridge Entity Slot tracking (updates even during compute_logits=False prefill)
        ent_logits_t = None
        ent_gate_t = False
        tok_unit = None
        use_role = self.enable_hierarchical_compositional and self.use_role_bridge and (not self.continual_mode)
        if use_role:
            tok_unit = F.normalize(tok_emb, p=2, dim=-1, eps=1e-6)
            ent_t = bool(self.is_entity_vocab[token_id].item())
            cur_l = int(cache.cur_len[0].item())
            if not ent_t:
                if cur_l > 0:
                    valid_commit = (cur_l >= 2) if self.has_multiword_entities else True
                    if valid_commit:
                        cache.comp_start_emb.copy_(cache.cur_start_emb)
                        cache.comp_end_emb.copy_(cache.cur_end_emb)
                        cache.comp_start_id.copy_(cache.cur_start_id)
                        cache.comp_end_id.copy_(cache.cur_end_id)
                        cache.comp_ready.fill_(True)
                cache.cur_len.zero_()
            else:
                if cur_l == 0:
                    cache.cur_start_emb.copy_(tok_emb)
                    cache.cur_end_emb.copy_(tok_emb)
                    cache.cur_start_id.fill_(token_id)
                    cache.cur_end_id.fill_(token_id)
                else:
                    cache.cur_end_emb.copy_(tok_emb)
                    cache.cur_end_id.fill_(token_id)
                cache.cur_len.add_(1)

            if compute_logits:
                bias_col = torch.ones(1, 1, device=tok_emb.device, dtype=tok_emb.dtype)
                win_feat = torch.cat(
                    [cache.lag3_unit, cache.lag2_unit, cache.lag1_unit, tok_unit, bias_col],
                    dim=-1,
                )
                role_logits = self.role_bridge_head(win_feat)
                role_probs = F.softmax(role_logits, dim=-1)
                p_ent_start = role_probs[:, 1]
                p_ent_end = role_probs[:, 2]
                p_ent_sum = p_ent_start + p_ent_end
                has_comp_flag = bool(cache.comp_ready[0].item())
                if has_comp_flag and float(p_ent_sum[0].item()) > 0.60:
                    ent_gate_t = True
                    ent_mix = (
                        p_ent_start.unsqueeze(-1) * cache.comp_start_emb
                        + p_ent_end.unsqueeze(-1) * cache.comp_end_emb
                    ) / p_ent_sum.clamp(min=1e-6).unsqueeze(-1)
                    ent_vec = F.normalize(self.entity_proj(ent_mix), p=2, dim=-1, eps=1e-6)
                    emb_unit = (
                        cache._emb_unit
                        if getattr(cache, "_emb_unit", None) is not None
                        else F.normalize(self.sparse_attention.token_embedding, p=2, dim=-1, eps=1e-6)
                    )
                    ent_logits_t = 22.0 * F.linear(ent_vec, emb_unit)[0]

        logits_out = None
        if compute_logits:
            k_len = t + 1
            q_t = self.sparse_attention.q_proj(x_t)  # [1, D_max]
            k_prefix = cache.k_cache[:, :k_len, :]
            v_prefix = cache.v_cache[:, :k_len, :]
            sal_prefix = cache.sal_cache[:, :k_len]

            scale = 1.0 / math.sqrt(self.d_max)
            scores = torch.bmm(q_t.unsqueeze(1), k_prefix.transpose(1, 2)).squeeze(1) * scale
            scores = scores + 1.35 * (sal_prefix - 1.0)

            attn_top_k = self.sparse_attention.attn_top_k
            if attn_top_k is not None and attn_top_k < k_len:
                salient_rank = 4.0 * sal_prefix + 0.15 * scores
                k_anchor = min(2, k_len)
                k_local = min(max(1, attn_top_k - k_anchor), k_len)
                sparse_scores = torch.full_like(scores, float("-inf"))
                _, anchor_idx = torch.topk(salient_rank, k=k_anchor, dim=-1)
                sparse_scores.scatter_(-1, anchor_idx, scores.gather(-1, anchor_idx))
                _, local_idx = torch.topk(scores, k=k_local, dim=-1)
                sparse_scores.scatter_(-1, local_idx, scores.gather(-1, local_idx))
                scores = sparse_scores

            attn_weights = F.softmax(scores, dim=-1)
            attended = torch.bmm(attn_weights.unsqueeze(1), v_prefix).squeeze(1) + x_t

            cap_gain = 0.45 if use_sliding else 1.65
            attn_slice = attended[:, :out_dim] if attended.shape[1] > out_dim else attended
            cap_ctx = F.linear(cache.c_state, self.cap_proj[:out_dim, :])
            ctx_t = attn_slice + alibi_ctx + cap_gain * cap_ctx
            if t >= 1:
                ctx_t = ctx_t + F.linear(cache.prev_emb_1, self.lag1_proj[:out_dim, :])
            if t >= 2:
                ctx_t = ctx_t + F.linear(cache.prev_emb_2, self.lag2_proj[:out_dim, :])

            use_block_path = self.enable_block_sparse and (not self.continual_mode)
            if (not use_block_path) and (self._fast_d_eff is None) and d_eff < self.d_max:
                ctx_t = ctx_t.clone()
                ctx_t[:, d_eff:] = 0.0

            if use_block_path:
                a = ctx_t
                if self.use_depth_scaled_highway:
                    a, _, _ = self._apply_subspace_rmsnorm(a, target_rms=1.20, return_stats=False)
                for l_idx, layer in enumerate(self.spring_layers):
                    a = self._forward_block_sparse_layer(layer, a, l_idx=l_idx)
            else:
                a = ctx_t
                for l_idx, layer in enumerate(self.spring_layers):
                    a = self._forward_dynamic_layer(layer, a, l_idx=l_idx)

            if self.frozen:
                in_dim = self.output_head.w0.shape[1]
                scene_logits = F.linear(
                    a[:, :in_dim] if a.shape[1] != in_dim else a,
                    self.output_head.w0,
                )
            elif use_block_path:
                d_last = self._final_layer_dim()
                a_head = a[:, :d_last] if a.shape[1] != d_last else a
                scene_logits = F.linear(a_head, self.output_head.w[:, :d_last])
            else:
                d_last = self._final_layer_dim()
                a_masked = a.clone()
                if d_last < self.d_max:
                    a_masked[:, d_last:] = 0.0
                scene_logits = self.output_head(a_masked)

            if use_role and ent_gate_t and ent_logits_t is not None:
                logits_out = 0.15 * scene_logits[0] + ent_logits_t
            else:
                logits_out = scene_logits[0].clone()

        # Advance causal lags, N-gram tables, and step index strictly in causal order
        cache.prev_emb_2.copy_(cache.prev_emb_1)
        cache.prev_emb_1.copy_(tok_emb)
        if use_role and tok_unit is not None:
            cache.lag3_unit.copy_(cache.lag2_unit)
            cache.lag2_unit.copy_(cache.lag1_unit)
            cache.lag1_unit.copy_(tok_unit)
        cache.record_token_ngrams(token_id)
        cache.step += 1
        return logits_out

    def _apply_refractory_gate_logits(
        self,
        logits: torch.Tensor,
        seq: list[int],
        cache: SpringInferenceCache | None = None,
    ) -> torch.Tensor:
        if len(seq) < 1:
            return logits
        prev1 = seq[-1]
        if 0 <= prev1 < self.vocab_size:
            logits[prev1] = float("-inf")

        if len(seq) >= 3:
            prev2 = seq[-2]
            prev3 = seq[-3]
            if cache is not None:
                bigram_counts = cache.bigram_next.get(prev1, {})
                trigram_counts = cache.trigram_next.get((prev2, prev1), {})
                fourgram_counts = cache.fourgram_next.get((prev3, prev2, prev1), {})
            else:
                bigram_counts: dict[int, int] = {}
                trigram_counts: dict[int, int] = {}
                fourgram_counts: dict[int, int] = {}
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
                    logits[cand_id] -= 15.0 * float(fg_cnt)
                if sal >= 0.95 and (t_cnt >= 2 or b_cnt >= 2):
                    logits[cand_id] -= 12.0 * float(max(t_cnt, b_cnt))
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 28,
        eos_id: int | None = None,
        enable_refractory_gate: bool = True,
        use_fast_cache: bool = False,
    ) -> list[int]:
        """
        Autoregressive rollout equipped with the Milestone 2.2 Two-Tier Transition Refractory Gate
        and Phase 3.1 O(1) Stateful Inference Cache (`use_fast_cache=True`).
        """
        seq = list(prompt_ids)
        if not seq:
            return seq

        if use_fast_cache and not (self.frozen and self.continual_mode and self.phase_resonance_profiles):
            prev_fast_gen = getattr(self, "_in_fast_generate", False)
            self._in_fast_generate = True
            try:
                cache = self.init_inference_cache(batch_size=1)
                init_window = seq[-self.max_seq_len:]
                # Keep O(1) N-gram refractory state synchronized over any clipped prefix before window
                prefix_cut = len(seq) - len(init_window)
                for tid in seq[:prefix_cut]:
                    cache.record_token_ngrams(tid)

                # Fast prefill over prompt_ids[:-1] (skips Q/softmax/hidden/head), compute logits on last prompt token
                logits = None
                for idx, tid in enumerate(init_window):
                    is_last = idx == (len(init_window) - 1)
                    logits = self.forward_step_cached(tid, cache, compute_logits=is_last)

                for _ in range(max_new_tokens):
                    if logits is None:
                        break
                    if enable_refractory_gate:
                        self._apply_refractory_gate_logits(logits, seq, cache=cache)
                    next_id = int(torch.argmax(logits).item())
                    seq.append(next_id)
                    if eos_id is not None and next_id == eos_id:
                        break

                    if cache.step >= self.max_seq_len:
                        # Sliding window shifted past max_seq_len: resync rolling positional window
                        saved_hist = cache.seq_history
                        saved_bi = cache.bigram_next
                        saved_tri = cache.trigram_next
                        saved_four = cache.fourgram_next
                        cache = self.init_inference_cache(batch_size=1)
                        win = seq[-self.max_seq_len:]
                        for w_idx, w_tid in enumerate(win):
                            is_last = w_idx == (len(win) - 1)
                            logits = self.forward_step_cached(w_tid, cache, compute_logits=is_last)
                        cache.seq_history = saved_hist
                        cache.bigram_next = saved_bi
                        cache.trigram_next = saved_tri
                        cache.fourgram_next = saved_four
                        cache.record_token_ngrams(next_id)
                    else:
                        logits = self.forward_step_cached(next_id, cache, compute_logits=True)
                return seq
            finally:
                self._in_fast_generate = prev_fast_gen

        for _ in range(max_new_tokens):
            window = seq[-self.max_seq_len:]
            logits = self.forward(window).clone()
            if enable_refractory_gate:
                self._apply_refractory_gate_logits(logits, seq, cache=None)
            next_id = int(torch.argmax(logits).item())
            seq.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return seq

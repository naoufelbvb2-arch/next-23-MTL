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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            sparse_scores = torch.full_like(scores, float("-inf"))
            # Salience-dominated anchor ranking (strictly causal: tau <= t)
            salient_rank_scores = (4.0 * key_salience + 0.15 * scores).masked_fill(
                ~causal_mask.unsqueeze(0), float("-inf")
            )
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
        self.frozen = False
        self.current_epoch = 1

        # Law 1: Clipped Log-Salience Vector s(v) in [0.70, 1.80]
        self.register_buffer("token_salience", torch.ones(self.vocab_size))
        # Charging gate alpha(v) for the Salience-Gated EMA Context Capacitor
        self.register_buffer("capacitor_alpha", torch.full((self.vocab_size,), 0.08))

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

        # 3. Law 4: Adaptive Hebbian Feedback Router B^(l) in R^{D_max x V}
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

        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def calibrate_salience_from_tokens(
        self,
        token_sequences: list[list[int]],
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
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
        counts = torch.zeros(self.vocab_size, dtype=torch.float32)
        doc_freq = torch.zeros(self.vocab_size, dtype=torch.float32)
        burst_freq = torch.zeros(self.vocab_size, dtype=torch.float32)
        n_docs = max(len(token_sequences), 1)

        for seq in token_sequences:
            seq_counts: dict[int, int] = {}
            for tid in seq:
                if 0 <= tid < self.vocab_size and tid not in (pad_id, bos_id, eos_id):
                    counts[tid] += 1.0
                    seq_counts[tid] = seq_counts.get(tid, 0) + 1
            for tid, c in seq_counts.items():
                doc_freq[tid] += 1.0
                if c >= 2:
                    # Exclude ultra-ubiquitous structural tokens (appearing in >35% of docs) from burstiness
                    burst_freq[tid] += 1.0

        total_tokens = float(counts.sum().item())
        active_vocab = counts > 0
        if total_tokens > 0 and active_vocab.any():
            # Inverse log document/token frequency
            inv_log_tf = torch.log((total_tokens + 1.0) / (counts + 1.0))
            idf = torch.log((float(n_docs) + 1.0) / (doc_freq + 1.0))

            # Intra-document recurrence ratio for selective content words (DF < 30% of docs)
            selective_mask = (doc_freq < (0.30 * float(n_docs))).float()
            burst_ratio = (burst_freq / (doc_freq + 1.0)) * selective_mask

            raw_info = 0.55 * inv_log_tf + 0.45 * idf + 1.85 * burst_ratio
            min_val = float(raw_info[active_vocab].min().item())
            max_val = float(raw_info[active_vocab].max().item())
            span = max(max_val - min_val, 1e-6)

            norm_sal = (raw_info - min_val) / span  # in [0, 1] for seen tokens
            salience = (0.70 + 1.10 * norm_sal).clamp(0.70, 1.80)
            salience = torch.where(active_vocab, salience, torch.ones_like(salience))
        else:
            salience = torch.ones(self.vocab_size)

        salience[pad_id] = 0.0
        salience[bos_id] = 0.70
        salience[eos_id] = 0.80

        self.token_salience.copy_(salience)
        self.sparse_attention.set_token_salience(salience)

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
        alpha[pad_id] = 0.0
        alpha[bos_id] = 0.0
        self.capacitor_alpha.copy_(alpha)

        seen_sal = salience[active_vocab]
        return {
            "min_salience": float(seen_sal.min().item()),
            "max_salience": float(seen_sal.max().item()),
            "mean_salience": float(seen_sal.mean().item()),
        }

    @torch.no_grad()
    def get_active_dimensions(self) -> list[int]:
        return [layer.d_active for layer in self.spring_layers]

    @torch.no_grad()
    def _effective_context_dim(self) -> int:
        return max(layer.d_active for layer in self.spring_layers)

    @torch.no_grad()
    def _final_layer_dim(self) -> int:
        return self.spring_layers[-1].d_active

    @torch.no_grad()
    def count_parameters_breakdown(self) -> dict:
        v = self.vocab_size
        d_max = self.d_max
        d_init = self.d_initial
        d_acts = self.get_active_dimensions()
        d_last = d_acts[-1]
        d_eff = max(d_acts)

        emb_params = v * d_max + self.max_seq_len * d_max + 3 * (d_max * d_max)
        attn_train_params = 3 * 3 * (d_max * d_max) + (d_max * v)
        attn_frozen_params = 3 * (d_max * d_max)

        hidden_max_train = self.num_layers * 3 * (d_max * d_max)
        hidden_init_active = self.num_layers * 3 * (d_init * d_init)
        hidden_curr_active = sum(3 * (d * d_eff) for d in d_acts)
        hidden_frozen_w0 = sum(d * d_eff for d in d_acts)

        dfa_params = self.num_layers * (d_max * v)

        head_max_train = 3 * (v * d_max)
        head_init_active = 3 * (v * d_init)
        head_curr_active = 3 * (v * d_last)
        head_frozen_w0 = v * d_last

        total_latent_train_params = (
            emb_params + attn_train_params + hidden_max_train + dfa_params + head_max_train
        )
        total_init_active_params = (
            emb_params + attn_train_params + hidden_init_active + (self.num_layers * d_init * v) + head_init_active
        )
        total_curr_active_params = (
            emb_params + attn_train_params + hidden_curr_active + sum(d * v for d in d_acts) + head_curr_active
        )
        total_frozen_prod_params = emb_params + attn_frozen_params + hidden_frozen_w0 + head_frozen_w0

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
        c_t = (1 - alpha_t) * c_{t-1} + alpha_t * Embed(x_t)
        Preserves subject/entity memory across sentence boundaries and high-frequency hub words.
        """
        bsz, seq_len = input_ids.shape

        # 1. Dual-Horizon Sparse Causal Self-Attention [B, T, D_max]
        attn_out = self.sparse_attention(input_ids)

        # 2. Token embeddings + Positional Phase [B, T, D_max]
        tok_emb = self.sparse_attention.token_embedding[input_ids]
        pos_emb = self.alibi_pos_encoding[:seq_len].unsqueeze(0)
        x_pos = tok_emb + pos_emb

        # 3. ALiBi local causal window [B, T, D_max]
        w_alibi = self.alibi_causal_weights[:seq_len, :seq_len]
        alibi_ctx = torch.matmul(w_alibi.unsqueeze(0), x_pos)

        # 4. Salience-Gated EMA Context Capacitor c_t [B, T, D_max]
        # Computed causally along the sequence length T (T=36 is tiny, vectorized across B and D)
        alphas = self.capacitor_alpha[input_ids].unsqueeze(-1)  # [B, T, 1]
        c_states = torch.zeros_like(tok_emb)
        c_prev = torch.zeros(bsz, self.d_max, device=input_ids.device)
        for t in range(seq_len):
            a_t = alphas[:, t, :]
            c_prev = (1.0 - a_t) * c_prev + a_t * tok_emb[:, t, :]
            c_states[:, t, :] = c_prev
        cap_ctx = F.linear(c_states, self.cap_proj)

        # 5. Immediate causal lag-1 and lag-2 projections for sharp local syntax
        lag1 = torch.zeros_like(tok_emb)
        if seq_len > 1:
            lag1[:, 1:, :] = F.linear(tok_emb[:, :-1, :], self.lag1_proj)

        lag2 = torch.zeros_like(tok_emb)
        if seq_len > 2:
            lag2[:, 2:, :] = F.linear(tok_emb[:, :-2, :], self.lag2_proj)

        return attn_out + alibi_ctx + 1.65 * cap_ctx + lag1 + lag2

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
        context = self._build_vectorized_context(input_ids)  # [B, T, D_max]

        d_in = self.spring_layers[0].d_active
        if d_in < self.d_max:
            context = context.clone()
            context[:, :, d_in:] = 0.0

        a = context.reshape(bsz * seq_len, self.d_max)
        for layer in self.spring_layers:
            a = layer(a)

        if self.frozen:
            in_dim = self.output_head.w0.shape[1]
            logits_flat = F.linear(a[:, :in_dim], self.output_head.w0)
        else:
            d_last = self._final_layer_dim()
            a_masked = a.clone()
            if d_last < self.d_max:
                a_masked[:, d_last:] = 0.0
            logits_flat = self.output_head(a_masked)

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
    ) -> None:
        """
        Law 3: Winner-Count Normalized Spring & Maturity-Gated Stiffening Update.
        Normalizes each cell j's driving signal and stiffness consolidation by its
        active winner participation N_wins(j) in the mini-batch rather than raw B * T,
        and applies maturity-gated stiffening so k_bar converges to [0.12, 0.35].
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

        # Law 1: Fast Elastic Stretch (w)
        stretch_diff = w_act - w0_act
        delta_w = (layer.eta_w * driving_signal) - (layer.gamma * k_act * stretch_diff)
        w_act.add_(delta_w)

        stretch_diff_new = w_act - w0_act

        # Law 2: Anchor Consolidation (w0)
        delta_w0 = layer.eta_0 * (1.0 - k_act) * stretch_diff_new
        w0_act.add_(delta_w0)

        # Law 3: Winner-Count Normalized & Maturity-Gated Stiffening (k)
        active_row_gate = (mask_act.sum(dim=0) > 0).float().unsqueeze(1)
        win_norm_factor = (max(n_valid, 1.0) / (wins_per_cell * sparsity_ratio)).clamp(1.0, 8.0)
        effective_beta_k = layer.beta_k * maturity_stiff_gain * win_norm_factor
        delta_k = effective_beta_k * stretch_diff_new.abs() * (1.0 - k_act) * active_row_gate
        k_act.add_(delta_k).clamp_(0.0, 0.85)

    @torch.no_grad()
    def train_minibatch(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        pad_id: int = 0,
    ) -> dict[str, float]:
        """
        Vectorized Mini-Batch Forward + Salience-Weighted Error + Winner-Normalized Spring Update
        + Row-Normalized Hebbian Feedback Alignment.
        """
        if self.frozen:
            raise RuntimeError("Model is frozen; training is disabled.")

        bsz, seq_len = input_ids.shape
        n_total = bsz * seq_len

        # 1. Vectorized Forward Pass: [B, T, V]
        logits = self.forward(input_ids)
        logits_flat = logits.reshape(n_total, self.vocab_size)
        targets_flat = target_ids.reshape(n_total)
        inputs_flat = input_ids.reshape(n_total)

        valid_mask = (targets_flat != pad_id).float()
        n_valid = float(valid_mask.sum().item())
        if n_valid == 0:
            return {"ce_loss": 0.0, "mse_loss": 0.0, "correct": 0, "valid_tokens": 0}

        probs = F.softmax(logits_flat, dim=-1)
        y_onehot = F.one_hot(targets_flat, num_classes=self.vocab_size).float()

        # Law 1: Weight error signal by target & input token salience s(v) in [0.70, 1.80]
        target_sal = self.token_salience[targets_flat]
        input_sal = self.token_salience[inputs_flat]
        step_salience = (0.65 * target_sal + 0.35 * input_sal).clamp(0.70, 1.80)

        raw_err = (y_onehot - probs) * valid_mask.unsqueeze(-1)
        salient_err = raw_err * step_salience.unsqueeze(-1)

        # Metrics on valid tokens
        ce_loss = float(F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id).item())
        mse_loss = float((0.5 * (raw_err ** 2).sum()).item() / n_valid)
        preds_flat = logits_flat.argmax(dim=-1)
        correct = int(((preds_flat == targets_flat) & (targets_flat != pad_id)).sum().item())

        # Law 3: Maturity-Gated Stiffening schedule (gentle warmup in Epochs 1-2, then ramps up)
        if self.current_epoch <= 2:
            maturity_gain = 2.0
        else:
            maturity_gain = min(280.0, 32.0 * float(self.current_epoch - 1))

        # 2. Update Dual-Horizon Attention locally
        scale_factor = float(n_total) / max(n_valid, 1.0)
        e_scaled = salient_err * scale_factor
        self.sparse_attention.local_spring_update(e_scaled)

        # 3. Law 3 & Law 4: Update Hidden DynamicSpringLayers + Adaptive Hebbian Feedback B^(l)
        for l_idx, layer in enumerate(self.spring_layers):
            d_act = layer.d_active
            B_l = self.dfa_router.get_matrix(l_idx)  # [D_max, V]
            B_l_active = B_l[:d_act, :]              # [d_active, V]

            delta_l_active = F.linear(salient_err, B_l_active)  # [N, d_active]
            self._update_dynamic_layer_winner_normalized(
                layer=layer,
                projected_error=delta_l_active,
                n_valid=n_valid,
                maturity_stiff_gain=maturity_gain,
            )

            # Law 4: Row-Normalized Hebbian Feedback Alignment on B^(l)
            if self.eta_b > 0.0 and layer._last_a is not None:
                a_act = layer._last_a[:, :d_act]  # [N, d_active]
                hebb_corr = torch.matmul(a_act.t(), salient_err) / max(n_valid, 1.0)  # [d_active, V]
                B_updated = B_l_active + self.eta_b * hebb_corr
                # Row-normalize to match initial scale 1 / sqrt(D_max) and prevent blowup
                target_row_norm = math.sqrt(self.vocab_size / float(self.d_max))
                B_l[:d_act, :].copy_(F.normalize(B_updated, p=2, dim=-1) * target_row_norm)

        # 4. Update Output Head locally
        d_last = self._final_layer_dim()
        if self.output_head._last_a_prev is not None and d_last < self.d_max:
            self.output_head._last_a_prev[:, d_last:] = 0.0
        self.output_head.local_spring_update(e_scaled)

        return {
            "ce_loss": ce_loss,
            "mse_loss": mse_loss,
            "correct": correct,
            "valid_tokens": int(n_valid),
        }

    @torch.no_grad()
    def step_neurogenesis(self, epoch: int, mean_epoch_error: float, token_error_rate: float) -> list[dict]:
        self.current_epoch = epoch + 1
        if not self.enable_neurogenesis or self.frozen:
            return []

        old_d_last = self._final_layer_dim()
        events = []
        for layer in self.spring_layers:
            ev = layer.check_and_trigger_neurogenesis(
                epoch=epoch,
                mean_epoch_error=mean_epoch_error,
                token_error_rate=token_error_rate,
            )
            if ev is not None:
                events.append(ev)
                self.growth_log.append(ev)

        new_d_last = self._final_layer_dim()
        if new_d_last > old_d_last:
            cols = self._head_latent_cols[:, old_d_last:new_d_last]
            self.output_head.w[:, old_d_last:new_d_last].copy_(cols)
            self.output_head.w0[:, old_d_last:new_d_last].copy_(cols)
            self.output_head.k[:, old_d_last:new_d_last].zero_()

        return events

    @torch.no_grad()
    def production_freeze(self, prune_dormant: bool = True) -> dict:
        """
        Final consolidation & production export:
        1. Snaps plastic anchors toward converged equilibrium w where stiffness permits.
        2. Destroys w, k, and adaptive feedback matrices B^(l) completely.
        3. Prunes unawakened dormant cells so only W_final = W_0 remains.
        """
        self.frozen = True

        # Gentle final consolidation pass before destroying w and k
        for layer in self.spring_layers:
            d_act = layer.d_active
            layer.w0[:d_act, :].lerp_(layer.w[:d_act, :], 0.25)
        self.output_head.w0.lerp_(self.output_head.w, 0.30)

        self.sparse_attention.production_freeze()
        self.sparse_attention.B_attn = None

        d_eff = self._effective_context_dim()
        d_last = self._final_layer_dim()

        layer_prune_stats = []
        for layer in self.spring_layers:
            stats = layer.production_freeze(prune_dormant=prune_dormant, prev_active_dim=d_eff)
            layer_prune_stats.append(stats)

        for l_idx in range(self.num_layers):
            setattr(self.dfa_router, f"B_{l_idx}", None)

        self.output_head.frozen = True
        pruned_head_w0 = self.output_head.w0[:, :d_last].clone()
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
            "total_active_cells": total_active_cells,
            "total_max_cells": total_max_cells,
            "dormant_cells_saved": saved_cells,
            "dormant_saved_pct": saved_pct,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 28,
        eos_id: int | None = None,
    ) -> list[int]:
        seq = list(prompt_ids)
        for _ in range(max_new_tokens):
            window = seq[-self.max_seq_len:]
            logits = self.forward(window)
            next_id = int(torch.argmax(logits).item())
            seq.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return seq

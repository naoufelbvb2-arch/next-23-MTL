"""
model_continuous.py
Phase 2 of Engineering Maturity Roadmap:
Hybrid Continuous-Stream Generative Network combining:
  1. Position-Sensitive Sparse Causal Self-Attention (ALiBi linear distance bias +
     multi-scale causal context + SparseCausalAttention from engine/sparse_attention.py)
     with strict causal mask (M_ij = -inf for j > i, Future Leakage = 0.0).
  2. 4 DynamicSpringLayers (from engine/dynamic_spring_layer.py) with Dormant Reserve
     (D_initial = 32 -> D_max = 128, waking up slices of delta_d = +8 on frustration/saturation).
  3. Direct Feedback Alignment (DFARouter) + Output SpringLayer Head.

Strictly isolated from spring_layer.py, dynamic_spring_layer.py, sparse_attention.py,
and all previous models/tests. Zero autograd / zero backward().
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.sparse_attention import SparseCausalAttention
from engine.dynamic_spring_layer import DynamicSpringLayer
from engine.dfa_router import DFARouter
from engine.spring_layer import SpringLayer


class ContinuousSpringNetwork(nn.Module):
    """
    Unified Sparse Causal Attention + Dynamic Neurogenesis Spring Network
    for Continuous Token Streams and Free-Running Autoregressive Rollouts.
    """

    def __init__(
        self,
        vocab_size: int,
        d_max: int = 128,
        d_initial: int = 32,
        delta_d: int = 8,
        num_layers: int = 4,
        top_k: int = 4,
        attn_top_k: int = 5,
        max_seq_len: int = 32,
        eta_w: float = 0.05,
        eta_0: float = 0.015,
        gamma: float = 0.10,
        beta_k: float = 0.02,
        enable_neurogenesis: bool = True,
        saturation_threshold: float = 0.85,
        stiffness_threshold: float = 0.0020,
        error_threshold: float = 0.01,
        cooldown_epochs: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_max = d_max
        self.d_initial = d_initial
        self.delta_d = delta_d
        self.num_layers = num_layers
        self.top_k = top_k
        self.attn_top_k = attn_top_k
        self.max_seq_len = max_seq_len
        self.enable_neurogenesis = enable_neurogenesis
        self.frozen = False

        # 1. Sparse Causal Self-Attention Layer (strictly causal M_ij = -inf for j > i)
        self.sparse_attention = SparseCausalAttention(
            vocab_size=vocab_size,
            d_model=d_max,
            attn_top_k=attn_top_k,
            max_seq_len=max_seq_len,
            eta_w=eta_w,
            eta_0=eta_0,
            gamma=gamma,
            beta_k=beta_k,
            seed=seed,
        )

        # Precompute non-differentiable multi-horizon ALiBi-style causal kernels so repeated
        # hub tokens ("في", "من", "على", "ثم", "حيث") at different positions t carry unique
        # causal prefix signatures (local bigram/trigram + long-range paragraph theme anchor).
        gen = torch.Generator()
        gen.manual_seed(seed + 777)

        # Orthogonal sinusoidal + random phase positional encodings [max_seq_len, d_max]
        pos_enc = torch.zeros(max_seq_len, d_max)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_max, 2, dtype=torch.float32) * (-math.log(1000.0) / d_max))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = F.normalize(pos_enc, p=2, dim=-1) * math.sqrt(d_max / 5.0)
        self.register_buffer("alibi_pos_encoding", pos_enc)

        # Strict lower-triangular ALiBi multi-horizon causal mixing matrix [max_seq_len, max_seq_len]
        # Combines immediate predecessor decay (0.65^(t-s)) + paragraph theme anchor (s=0, 1)
        causal_tril = torch.tril(torch.ones(max_seq_len, max_seq_len))
        idx = torch.arange(max_seq_len)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0).float()
        alibi_weights = (0.65 ** dist) * causal_tril
        # Boost initial subject/verb seed positions (s = 0, 1) for long-range theme retention
        alibi_weights[:, 0] += 0.35 * causal_tril[:, 0]
        alibi_weights[:, 1] += 0.25 * causal_tril[:, 1]
        alibi_weights = alibi_weights / alibi_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("alibi_causal_weights", alibi_weights)

        # Fixed causal lag projection matrices to prevent collisions when the exact same hub word
        # ("في", "من", "ثم") appears multiple times in one paragraph
        lag1_proj = torch.randn(d_max, d_max, generator=gen) / math.sqrt(d_max)
        lag2_proj = torch.randn(d_max, d_max, generator=gen) / math.sqrt(d_max)
        theme_proj = torch.randn(d_max, d_max, generator=gen) / math.sqrt(d_max)
        self.register_buffer("lag1_proj", lag1_proj)
        self.register_buffer("lag2_proj", lag2_proj)
        self.register_buffer("theme_proj", theme_proj)

        # 2. L = 4 DynamicSpringLayers with Dormant Reserve (D_init = 32 -> D_max = 128)
        self.spring_layers = nn.ModuleList([
            DynamicSpringLayer(
                d_max=d_max,
                d_initial=d_initial,
                delta_d=delta_d,
                top_k=top_k,
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
            for l_idx in range(num_layers)
        ])
        self.hidden_layers = self.spring_layers

        # 3. Fixed Random Feedback Matrices B^(l) in R^{D_max x V}
        self.dfa_router = DFARouter(
            num_layers=num_layers,
            d_model=d_max,
            vocab_size=vocab_size,
            seed=seed + 500,
        )

        # 4. Output Linear Head (SpringLayer holding w, w0, k)
        self.output_head = SpringLayer(
            in_features=d_max,
            out_features=vocab_size,
            top_k=None,
            eta_w=eta_w * 4.5,
            eta_0=eta_0 * 3.0,
            gamma=gamma * 0.20,
            beta_k=beta_k,
            activation="identity",
            use_residual=False,
            seed=seed + 999,
        )
        with torch.no_grad():
            self._head_latent_cols = self.output_head.w.clone()
            self.output_head.w[:, d_initial:].zero_()
            self.output_head.w0[:, d_initial:].zero_()
            self.output_head.k[:, d_initial:].zero_()

        self.growth_log: list[dict] = []

        for param in self.parameters():
            param.requires_grad_(False)

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
    def _build_causal_stream_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes strictly causal position-aware context representations [B, T, D_max]
        combining SparseCausalAttention (Top-K self-attention) with ALiBi multi-horizon
        causal history so repeated hub words at different positions never collide.
        Strictly causal: token at position t depends ONLY on tokens 0..t.
        """
        bsz, seq_len = input_ids.shape

        # 1. Sparse Causal Self-Attention output [B, T, D_max]
        attn_out = self.sparse_attention(input_ids)

        # 2. Raw token embeddings + Sinusoidal positional phase [B, T, D_max]
        tok_emb = self.sparse_attention.token_embedding[input_ids]
        pos_emb = self.alibi_pos_encoding[:seq_len].unsqueeze(0)
        x_pos = tok_emb + pos_emb

        # 3. ALiBi causal weighted history [B, T, D_max] (strictly lower-triangular)
        w_alibi = self.alibi_causal_weights[:seq_len, :seq_len]
        alibi_ctx = torch.matmul(w_alibi.unsqueeze(0), x_pos)

        # 4. Causal lag-1 (t-1), lag-2 (t-2), and paragraph theme (t=0) rotational signatures
        lag1 = torch.zeros_like(tok_emb)
        if seq_len > 1:
            lag1[:, 1:, :] = F.linear(tok_emb[:, :-1, :], self.lag1_proj)

        lag2 = torch.zeros_like(tok_emb)
        if seq_len > 2:
            lag2[:, 2:, :] = F.linear(tok_emb[:, :-2, :], self.lag2_proj)

        theme = F.linear(tok_emb[:, 0:1, :], self.theme_proj).expand(bsz, seq_len, self.d_max)

        combined = attn_out + alibi_ctx + lag1 + lag2 + 0.75 * theme
        return combined

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor | list[int]) -> torch.Tensor:
        """
        Forward pass on a token stream [B, T] or 1D prefix [T].
        """
        single_seq = False
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            single_seq = True
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            single_seq = True

        bsz, seq_len = input_ids.shape
        context = self._build_causal_stream_context(input_ids)  # [B, T, D_max]

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
    def forward_step_prefix(self, prefix_ids: list[int] | torch.Tensor) -> torch.Tensor:
        """
        Step-wise causal forward pass on prefix (w_1, ..., w_t) returning logits [V]
        for next token w_{t+1}, while caching only step t's activations in each layer
        for purely local step-wise spring updates.
        """
        if isinstance(prefix_ids, list):
            input_ids = torch.tensor([prefix_ids], dtype=torch.long)
        elif prefix_ids.dim() == 1:
            input_ids = prefix_ids.unsqueeze(0)
        else:
            input_ids = prefix_ids

        context = self._build_causal_stream_context(input_ids)  # [1, t, D_max]
        # Keep only final causal position t in attention projections for local update
        for proj in (self.sparse_attention.q_proj, self.sparse_attention.k_proj, self.sparse_attention.v_proj):
            if proj._last_a_prev is not None:
                proj._last_a_prev = proj._last_a_prev[-1:, :]
                proj._last_z = proj._last_z[-1:, :]
                proj._last_mask = proj._last_mask[-1:, :]
                proj._last_a = proj._last_a[-1:, :]

        d_in = self.spring_layers[0].d_active
        a = context[:, -1, :].clone()  # [1, D_max]
        if d_in < self.d_max:
            a[:, d_in:] = 0.0

        for layer in self.spring_layers:
            a = layer(a)

        d_last = self._final_layer_dim()
        if d_last < self.d_max:
            a = a.clone()
            a[:, d_last:] = 0.0

        logits = self.output_head(a).squeeze(0)  # [V]
        return logits

    @torch.no_grad()
    def local_step_update(self, e_output: torch.Tensor) -> None:
        """
        Step-wise local spring update broadcasting e_t = y_{t+1} - y_hat_{t+1}
        directly via DFARouter to active rows [:d_active] of each DynamicSpringLayer,
        SparseCausalAttention, and Output Head.
        """
        e_batch = e_output.unsqueeze(0) if e_output.dim() == 1 else e_output

        # 1. Update SparseCausalAttention locally
        self.sparse_attention.local_spring_update(e_batch)

        # 2. Update 4 DynamicSpringLayers locally on their active rows [:d_active]
        for l_idx, layer in enumerate(self.spring_layers):
            d_act = layer.d_active
            B_l_active = self.dfa_router.get_matrix(l_idx)[:d_act, :]
            delta_l_active = F.linear(e_batch, B_l_active)
            layer.local_spring_update(delta_l_active)

        # 3. Update Output Head locally on active columns [:d_last]
        d_last = self._final_layer_dim()
        if self.output_head._last_a_prev is not None and d_last < self.d_max:
            self.output_head._last_a_prev[:, d_last:] = 0.0
        self.output_head.local_spring_update(e_batch)

    @torch.no_grad()
    def step_neurogenesis(self, epoch: int, mean_epoch_error: float, token_error_rate: float) -> list[dict]:
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
        Destroys w and k across SparseCausalAttention, all 4 DynamicSpringLayers,
        and Output Head; prunes unawakened dormant cells; exports strictly W_final = W_0.
        """
        self.frozen = True
        self.sparse_attention.production_freeze()

        d_eff = self._effective_context_dim()
        d_last = self._final_layer_dim()

        layer_prune_stats = []
        for layer in self.spring_layers:
            stats = layer.production_freeze(prune_dormant=prune_dormant, prev_active_dim=d_eff)
            layer_prune_stats.append(stats)

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
    def autoregressive_rollout(self, seed_prefix_ids: list[int], total_length: int) -> list[int]:
        """
        Free-Running Autoregressive Generation Rollout:
        Given ONLY the initial seed tokens (e.g., first 2 words [w_1, w_2]),
        generates the remaining tokens one by one until `total_length` is reached,
        feeding each predicted token w_hat_{t+1} back as input for step t+1.
        """
        generated = list(seed_prefix_ids)
        while len(generated) < total_length:
            logits = self.forward(generated)  # [V] for next token
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
        return generated

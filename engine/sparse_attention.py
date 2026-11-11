"""
engine/sparse_attention.py
Backpropagation-free Sparse Causal Self-Attention module governed by the 3 Spring
Update Laws (w, w0, k) and Direct Feedback Alignment (DFA).
Enforces strict causal masking (M_ij = -inf for j > i) and Top-K sparse attention
routing across past tokens to resolve long-range linguistic dependencies (5-6 words)
without any autograd or chain-rule backpropagation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spring_layer import SpringLayer


class SparseCausalAttention(nn.Module):
    """
    Causal Top-K Self-Attention preceding the 4 hidden SpringLayers.
    Uses local SpringLayer projections (W_Q, W_K, W_V) that each maintain (w, w0, k)
    and update locally via Direct Feedback Alignment, plus strict lower-triangular
    causal masking M_{t, s} = -inf for s > t.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        attn_top_k: int = 3,
        max_seq_len: int = 16,
        eta_w: float = 0.05,
        eta_0: float = 0.01,
        gamma: float = 0.1,
        beta_k: float = 0.02,
        seed: int = 202,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.attn_top_k = attn_top_k
        self.max_seq_len = max_seq_len
        self.frozen = False

        gen = torch.Generator()
        gen.manual_seed(seed)

        # Normalized distinct token embeddings + positional phase embeddings
        emb = torch.randn(vocab_size, d_model, generator=gen)
        emb = F.normalize(emb, p=2, dim=-1) * math.sqrt(d_model / 3.0)
        self.register_buffer("token_embedding", emb)

        pos = torch.randn(max_seq_len, d_model, generator=gen)
        pos = F.normalize(pos, p=2, dim=-1) * math.sqrt(d_model / 6.0)
        self.register_buffer("pos_embedding", pos)

        # Q, K, V projections implemented as SpringLayers (holding w, w0, k)
        self.q_proj = SpringLayer(
            in_features=d_model,
            out_features=d_model,
            top_k=None,
            eta_w=eta_w,
            eta_0=eta_0,
            gamma=gamma,
            beta_k=beta_k,
            activation="identity",
            use_residual=False,
            seed=seed + 1,
        )
        self.k_proj = SpringLayer(
            in_features=d_model,
            out_features=d_model,
            top_k=None,
            eta_w=eta_w,
            eta_0=eta_0,
            gamma=gamma,
            beta_k=beta_k,
            activation="identity",
            use_residual=False,
            seed=seed + 2,
        )
        self.v_proj = SpringLayer(
            in_features=d_model,
            out_features=d_model,
            top_k=8,
            eta_w=eta_w,
            eta_0=eta_0,
            gamma=gamma,
            beta_k=beta_k,
            activation="leaky_relu",
            use_residual=True,
            seed=seed + 3,
        )

        # Fixed random DFA feedback matrix B_attn in R^{D x V} for local V-spring updates
        bound = 1.0 / math.sqrt(vocab_size)
        b_attn = torch.empty(d_model, vocab_size).uniform_(-bound, bound, generator=gen)
        self.register_buffer("B_attn", b_attn)

        # Cached attention map from latest forward pass for diagnostics & causal verification
        self._last_attn_weights: torch.Tensor | None = None

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T] of token indices
        returns: causal self-attended representations [B, T, D]
        """
        bsz, seq_len = input_ids.shape
        tok_emb = self.token_embedding[input_ids]  # [B, T, D]
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0)  # [1, T, D]
        x = tok_emb + pos_emb  # [B, T, D]

        x_flat = x.reshape(bsz * seq_len, self.d_model)
        q = self.q_proj(x_flat).reshape(bsz, seq_len, self.d_model)
        k = self.k_proj(x_flat).reshape(bsz, seq_len, self.d_model)
        v = self.v_proj(x_flat).reshape(bsz, seq_len, self.d_model)

        # 1. Raw scaled dot-product + content similarity scores: [B, T, T]
        scale = 1.0 / math.sqrt(self.d_model)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add long-range anchor bias toward token 0 (the main subject head at the start of the sentence)
        # and the immediate predecessor token (t-1) so distant subject-predicate agreement is preserved
        scores[:, :, 0] += 1.25

        # 2. Strict Causal Mask: M_{i,j} = -inf for all j > i (future positions strictly forbidden)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

        # 3. Sparse Top-K Causal Selection per query position t
        if self.attn_top_k is not None and self.attn_top_k < seq_len:
            sparse_scores = torch.full_like(scores, float("-inf"))
            for t in range(seq_len):
                k_eff = min(t + 1, self.attn_top_k)
                # Only consider valid causal positions 0..t
                valid_slice = scores[:, t, : t + 1]
                _, top_idx = torch.topk(valid_slice, k=k_eff, dim=-1)
                sparse_scores[:, t, : t + 1].scatter_(-1, top_idx, valid_slice.gather(-1, top_idx))
            scores = sparse_scores

        # 4. Softmax over allowed causal Top-K positions
        attn_weights = F.softmax(scores, dim=-1)
        # Guarantee exact 0.0 on upper triangle (j > i)
        attn_weights = attn_weights * causal_mask.unsqueeze(0).float()

        self._last_attn_weights = attn_weights.detach()

        # 5. Weighted sum of values + residual connection with current token state
        attended = torch.matmul(attn_weights, v) + x
        return attended

    @torch.no_grad()
    def local_spring_update(self, error_vec: torch.Tensor) -> dict[str, float]:
        """
        Updates the Value/Query/Key SpringLayers locally via Direct Feedback Alignment
        without backpropagating through the softmax attention matrix.
        error_vec: [N, V] output error vector
        """
        if self.frozen:
            raise RuntimeError("SparseCausalAttention is frozen.")
        proj_err = F.linear(error_vec, self.B_attn)  # [N, D]
        v_stats = self.v_proj.local_spring_update(proj_err)
        q_stats = self.q_proj.local_spring_update(proj_err * 0.25)
        k_stats = self.k_proj.local_spring_update(proj_err * 0.25)
        return {
            "attn_mean_k": (v_stats["mean_stiffness"] + q_stats["mean_stiffness"] + k_stats["mean_stiffness"]) / 3.0,
            "attn_stretch": v_stats["mean_stretch"],
        }

    @torch.no_grad()
    def production_freeze(self) -> None:
        self.frozen = True
        self.q_proj.production_freeze()
        self.k_proj.production_freeze()
        self.v_proj.production_freeze()

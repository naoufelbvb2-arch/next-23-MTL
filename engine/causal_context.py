"""
engine/causal_context.py
Autoregressive sequence processor, token embedding table, and causal masked context
(supporting Causal Cumulative Mean / Position-Aware Causal Context and Top-K attention).
Strictly forward-only without autograd.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalContext(nn.Module):
    """
    Embedding Layer + Causal Masked Context.
    Given input tokens [x_1, x_2, ..., x_t], produces a causal context representation
    in R^{d_model} at each position t using only tokens <= t.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 16,
        mode: str = "causal_weighted_mean",
        seed: int = 101,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.mode = mode

        gen = torch.Generator()
        gen.manual_seed(seed)

        # Orthogonal / normalized random token embeddings so each token has a distinct signature
        emb = torch.randn(vocab_size, d_model, generator=gen)
        emb = F.normalize(emb, p=2, dim=-1) * math.sqrt(d_model / 4.0)
        self.register_buffer("token_embedding", emb)

        # Fixed positional phase encoding so position 1, 2, 3 in causal context are distinguishable
        pos = torch.randn(max_seq_len, d_model, generator=gen)
        pos = F.normalize(pos, p=2, dim=-1) * math.sqrt(d_model / 8.0)
        self.register_buffer("pos_embedding", pos)

        # Precompute lower-triangular causal weights (exponential recency + cumulative context)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        # Position-specific causal decay weights: w_{t, s} = 0.7^(t - s) for s <= t
        idx = torch.arange(max_seq_len)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0).float()
        weights = (0.75 ** dist) * causal_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("causal_weights", weights)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len] of token indices
        returns: causal context representations [batch_size, seq_len, d_model]
        """
        bsz, seq_len = input_ids.shape
        tok_emb = self.token_embedding[input_ids]  # [B, T, D]
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0)  # [1, T, D]
        x = tok_emb + pos_emb  # [B, T, D]

        if self.mode == "causal_cumulative_mean":
            # Simple causal cumulative mean: (1/t) * sum_{s=1}^t x_s
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            causal_mat = mask / denom
            context = torch.matmul(causal_mat.unsqueeze(0), x)
        else:
            # Causal weighted cumulative context (preserves exact order & prefix identity)
            w = self.causal_weights[:seq_len, :seq_len]
            context = torch.matmul(w.unsqueeze(0), x)

        return context

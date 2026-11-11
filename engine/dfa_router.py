"""
engine/dfa_router.py
Fixed random feedback matrices B^(l) in R^{D x V} for Direct Feedback Alignment (DFA).
Broadcasts the output error vector e = y_target - y_pred directly to each hidden layer
without backpropagation or chain-rule traversal.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFARouter(nn.Module):
    """
    Holds fixed, non-trainable random projection matrices B^(l) in R^{d_model x vocab_size}
    for each hidden layer l = 1..L.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        vocab_size: int,
        seed: int = 42,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size

        gen = torch.Generator()
        gen.manual_seed(seed)

        bound = 1.0 / math.sqrt(vocab_size)
        for l_idx in range(num_layers):
            b_mat = torch.empty(d_model, vocab_size).uniform_(-bound, bound, generator=gen)
            # Fixed random matrix: registered as buffer, never updated
            self.register_buffer(f"B_{l_idx}", b_mat)

    @torch.no_grad()
    def get_matrix(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"B_{layer_idx}")

    @torch.no_grad()
    def project_error(self, layer_idx: int, error_vec: torch.Tensor) -> torch.Tensor:
        """
        Projects output error e in R^{N x V} directly into hidden layer l space R^{N x D}:
            projected = e @ (B^(l))^T  <=>  B^(l) e for each sample.
        """
        b_mat = self.get_matrix(layer_idx)  # [D, V]
        return F.linear(error_vec, b_mat)   # [N, D]

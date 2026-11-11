"""
engine/spring_layer.py
Custom PyTorch module holding (w, w0, k) and performing strictly local manual
tensor updates with torch.no_grad() and zero autograd reliance.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpringLayer(nn.Module):
    """
    A single Spring-Link layer governed by the 3 Spring Update Laws:
    State per link: S_ij = {w_ij, w0_ij, k_ij}
      - w:  Fast elastic stretch weight (momentary weight)
      - w0: Slow mechanical anchor weight (consolidated long-term memory)
      - k:  Adaptive stiffness coefficient in [0, 1]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        top_k: int | None = 4,
        eta_w: float = 0.05,
        eta_0: float = 0.01,
        gamma: float = 0.1,
        beta_k: float = 0.02,
        activation: str = "leaky_relu",
        negative_slope: float = 0.1,
        use_residual: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.top_k = top_k
        self.eta_w = eta_w
        self.eta_0 = eta_0
        self.gamma = gamma
        self.beta_k = beta_k
        self.activation = activation
        self.negative_slope = negative_slope
        self.use_residual = use_residual and (in_features == out_features)
        self.frozen = False

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Initialize weights using Kaiming-like uniform scale without autograd
        bound = 1.0 / math.sqrt(in_features)
        w_init = torch.empty(out_features, in_features).uniform_(-bound, bound, generator=gen)

        # Register as buffers (or non-grad parameters) so autograd is NEVER invoked
        self.register_buffer("w", w_init.clone())
        self.register_buffer("w0", w_init.clone())
        self.register_buffer("k", torch.zeros(out_features, in_features))

        # Cached states from the most recent forward pass for local learning
        self._last_a_prev: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None
        self._last_mask: torch.Tensor | None = None
        self._last_a: torch.Tensor | None = None

    def _activate(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation == "leaky_relu":
            return F.leaky_relu(z, negative_slope=self.negative_slope)
        elif self.activation == "gelu":
            return F.gelu(z)
        elif self.activation == "identity":
            return z
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _activate_derivative(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation == "leaky_relu":
            return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, self.negative_slope))
        elif self.activation == "gelu":
            # Exact derivative of GELU
            sqrt_2 = math.sqrt(2.0)
            cdf = 0.5 * (1.0 + torch.erf(z / sqrt_2))
            pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            return cdf + z * pdf
        elif self.activation == "identity":
            return torch.ones_like(z)
        raise ValueError(f"Unsupported activation: {self.activation}")

    @torch.no_grad()
    def forward(self, a_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using momentary weight w (during training) or consolidated w0 (when frozen).
        Applies Winner-Take-All Causal Top-K competition across the feature dimension.
        a_prev: [N, in_features]
        returns: a [N, out_features]
        """
        active_weight = self.w0 if (self.frozen or self.w is None) else self.w

        # 1. Cumulative potential z_j^(l) = sum_i w_ij^(l) a_i^(l-1)
        z = F.linear(a_prev, active_weight)

        # 2. Winner-Take-All Top-K mask on |z| (or z > 0 / top-|z|)
        if self.top_k is not None and 0 < self.top_k < self.out_features:
            _, top_indices = torch.topk(z.abs(), k=self.top_k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, top_indices, 1.0)
        else:
            mask = torch.ones_like(z)

        # Apply activation and sparse mask
        a_sparse = self._activate(z) * mask

        # Store local state for the subsequent local spring update
        self._last_a_prev = a_prev.detach()
        self._last_z = z.detach()
        self._last_mask = mask.detach()
        self._last_a = a_sparse.detach()

        if self.use_residual:
            return a_prev + a_sparse
        return a_sparse

    @torch.no_grad()
    def local_spring_update(self, projected_error: torch.Tensor) -> dict[str, float]:
        """
        Executes the 3 Spring Update Laws purely locally without any chain rule across layers.
        projected_error: (B^(l) e) of shape [N, out_features] (or direct error e for output layer)
        """
        if self.frozen:
            raise RuntimeError("Cannot update a frozen SpringLayer in production mode.")
        if self._last_a_prev is None or self._last_z is None or self._last_mask is None:
            raise RuntimeError("Must call forward() before local_spring_update().")

        a_prev = self._last_a_prev  # [N, in_features]
        z = self._last_z            # [N, out_features]
        mask = self._last_mask      # [N, out_features] (I(j in Top-K / active))

        # Local error signal:
        # delta_j^(l) = (B^(l) e)_j * g'(z_j^(l)) * I(active)
        g_prime = self._activate_derivative(z)
        delta = projected_error * g_prime * mask  # [N, out_features]

        # Average outer product over batch/sequence items N: [out_features, in_features]
        batch_size = max(1, a_prev.shape[0])
        driving_signal = torch.matmul(delta.t(), a_prev) / float(batch_size)

        # Law 1: Fast Elastic Stretch (w)
        # Delta w_ij = eta_w * delta_j * a_i - gamma * k_ij * (w_ij - w0_ij)
        stretch_diff = self.w - self.w0
        delta_w = (self.eta_w * driving_signal) - (self.gamma * self.k * stretch_diff)
        self.w.add_(delta_w)

        # Recompute stretch after fast move for anchor & stiffness updates
        stretch_diff_new = self.w - self.w0

        # Law 2: Anchor Consolidation (w0)
        # Delta w0_ij = eta_0 * (1 - k_ij) * (w_ij - w0_ij)
        delta_w0 = self.eta_0 * (1.0 - self.k) * stretch_diff_new
        self.w0.add_(delta_w0)

        # Law 3: Local Adaptive Stiffening (k)
        # Delta k_ij = beta_k * |w_ij - w0_ij| * (1 - k_ij)
        delta_k = self.beta_k * stretch_diff_new.abs() * (1.0 - self.k)
        self.k.add_(delta_k).clamp_(0.0, 1.0)

        return {
            "mean_stretch": float(stretch_diff_new.abs().mean().item()),
            "mean_stiffness": float(self.k.mean().item()),
            "max_stiffness": float(self.k.max().item()),
        }

    @torch.no_grad()
    def consolidate_to_w0(self) -> None:
        """
        Soft synchronization check or final snap of w0 toward w when error has converged,
        while preserving the spring equilibrium state.
        """
        # During normal spring dynamics w0 tracks w via Law 2.
        pass

    @torch.no_grad()
    def production_freeze(self) -> None:
        """
        Section 2.4: Production Freeze.
        Discards momentary w and stiffness k matrices completely, exporting the layer
        as a lightweight static network running strictly on W_final = W_0.
        """
        self.frozen = True
        self.w = None
        self.k = None
        self._last_a_prev = None
        self._last_z = None
        self._last_mask = None
        self._last_a = None

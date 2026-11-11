"""
engine/dynamic_spring_layer.py
Dynamic Neurogenesis & Dormant Reserve SpringLayer module.
Strictly isolated from engine/spring_layer.py; zero autograd / zero backward().

Physical Mechanics:
  - Total latent capacity: D_max = 128 cells (out_features = d_max, in_features = d_max)
  - Initial active pool:   D_initial = 32 cells
  - Dormant Reserve:       96 cells initially asleep (w = 0, w0 = 0, k = 0, active_mask = 0)
  - Wake-Up Trigger:       Recruits delta_d = 8 dormant cells when physical frustration
                           (high error + high winner stiffness k or >85% active cell saturation)
                           is detected outside cooldown.
  - Production Freeze:     Destroys w and k, retains only W_0, and physically prunes/slices
                           out any cells that remained dormant through the end of training.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSpringLayer(nn.Module):
    """
    Spring-Link layer equipped with a Dormant Reserve pool and Frustration-Driven
    Dynamic Neurogenesis.
    """

    def __init__(
        self,
        d_max: int = 128,
        d_initial: int = 32,
        delta_d: int = 8,
        top_k: int = 4,
        eta_w: float = 0.05,
        eta_0: float = 0.01,
        gamma: float = 0.1,
        beta_k: float = 0.02,
        activation: str = "leaky_relu",
        negative_slope: float = 0.1,
        use_residual: bool = True,
        saturation_threshold: float = 0.85,
        stiffness_threshold: float = 0.0020,
        error_threshold: float = 0.08,
        cooldown_epochs: int = 4,
        layer_id: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.d_max = d_max
        self.d_initial = d_initial
        self.d_active = d_initial
        self.delta_d = delta_d
        self.top_k = top_k
        self.eta_w = eta_w
        self.eta_0 = eta_0
        self.gamma = gamma
        self.beta_k = beta_k
        self.activation = activation
        self.negative_slope = negative_slope
        self.use_residual = use_residual
        self.saturation_threshold = saturation_threshold
        self.stiffness_threshold = stiffness_threshold
        self.error_threshold = error_threshold
        self.cooldown_epochs = cooldown_epochs
        self.layer_id = layer_id
        self.frozen = False

        self._seed = seed
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

        # Pre-generate latent initial weights for the full [d_max, d_max] space so wake-ups
        # draw deterministic fresh connections, while keeping dormant rows strictly zeroed!
        bound = 1.0 / math.sqrt(d_max)
        latent_w = torch.empty(d_max, d_max).uniform_(-bound, bound, generator=self._gen)
        self.register_buffer("_latent_init_pool", latent_w)

        # Binary Structural Active Mask: 1.0 for active cells [0 .. d_active-1], 0.0 for dormant [d_active .. d_max-1]
        active_mask = torch.zeros(d_max)
        active_mask[:d_initial] = 1.0
        self.register_buffer("active_mask", active_mask)

        # Initialize w, w0, k: dormant rows (> d_initial) are strictly 0.0!
        w_init = torch.zeros(d_max, d_max)
        w_init[:d_initial, :] = latent_w[:d_initial, :]
        self.register_buffer("w", w_init.clone())
        self.register_buffer("w0", w_init.clone())
        self.register_buffer("k", torch.zeros(d_max, d_max))

        # Track cumulative neuron participation (wins in Top-K) across training
        self.register_buffer("win_counts", torch.zeros(d_max))
        self.register_buffer("epoch_win_counts", torch.zeros(d_max))

        # Cached states from forward pass
        self._last_a_prev: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None
        self._last_mask: torch.Tensor | None = None
        self._last_a: torch.Tensor | None = None
        self._last_winner_k: float = 0.0
        self._cooldown_counter: int = 0

        # Pruned production shape metadata
        self.pruned_out_features: int = d_initial
        self.pruned_in_features: int = d_max

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
        Forward pass restricted strictly to the currently active cell pool (0 .. d_active-1).
        Dormant cells (d_active .. d_max-1) do not compute activations or enter Top-K competition.
        """
        if self.frozen:
            in_dim = self.w0.shape[1]
            out_dim = self.w0.shape[0]
            a_in = a_prev[:, :in_dim]
            z_active = F.linear(a_in, self.w0)  # [N, d_active]

            if self.top_k is not None and 0 < self.top_k < out_dim:
                _, top_idx = torch.topk(z_active.abs(), k=self.top_k, dim=-1)
                mask_active = torch.zeros_like(z_active)
                mask_active.scatter_(-1, top_idx, 1.0)
            else:
                mask_active = torch.ones_like(z_active)

            a_sparse = self._activate(z_active) * mask_active

            if a_prev.shape[1] > out_dim:
                out_full = torch.zeros_like(a_prev)
                out_full[:, :out_dim] = a_sparse
                if self.use_residual:
                    res = torch.zeros_like(a_prev)
                    res[:, :out_dim] = a_prev[:, :out_dim] + a_sparse
                    return res
                return out_full
            return (a_prev + a_sparse) if self.use_residual else a_sparse

        # Training mode: only active rows [:d_active] participate
        d_act = self.d_active
        w_active = self.w[:d_act, :]  # [d_active, d_max]
        z_active = F.linear(a_prev, w_active)  # [N, d_active]

        # Winner-Take-All Top-K strictly among the active pool [0 .. d_active-1]
        if self.top_k is not None and 0 < self.top_k < d_act:
            _, top_indices = torch.topk(z_active.abs(), k=self.top_k, dim=-1)
            mask_active = torch.zeros_like(z_active)
            mask_active.scatter_(-1, top_indices, 1.0)
        else:
            mask_active = torch.ones_like(z_active)

        a_active = self._activate(z_active) * mask_active  # [N, d_active]

        # Embed into full d_max vector with zeros in the dormant reserve [d_active .. d_max-1]
        bsz = a_prev.shape[0]
        z_full = torch.zeros(bsz, self.d_max, device=a_prev.device)
        mask_full = torch.zeros(bsz, self.d_max, device=a_prev.device)
        a_full = torch.zeros(bsz, self.d_max, device=a_prev.device)

        z_full[:, :d_act] = z_active
        mask_full[:, :d_act] = mask_active
        a_full[:, :d_act] = a_active

        # Update participation counters for active cells
        batch_wins = mask_full.sum(dim=0)
        self.win_counts.add_(batch_wins)
        self.epoch_win_counts.add_(batch_wins)

        # Compute mean stiffness k of the winning cells in this step
        winner_rows = mask_active.sum(dim=0) > 0
        if winner_rows.any():
            self._last_winner_k = float(self.k[:d_act, :][winner_rows].mean().item())

        self._last_a_prev = a_prev.detach()
        self._last_z = z_full.detach()
        self._last_mask = mask_full.detach()
        self._last_a = a_full.detach()

        if self.use_residual:
            out = torch.zeros_like(a_prev)
            out[:, :d_act] = a_prev[:, :d_act] + a_active
            return out
        return a_full

    @torch.no_grad()
    def local_spring_update(self, projected_error: torch.Tensor) -> dict[str, float]:
        """
        Updates ONLY the active rows [:d_active] using the 3 Spring Update Laws.
        Dormant rows [d_active:] receive zero updates and remain at w = 0, w0 = 0, k = 0.
        projected_error: [N, d_active] or [N, d_max]
        """
        if self.frozen:
            raise RuntimeError("Cannot update a frozen DynamicSpringLayer.")
        if self._last_a_prev is None or self._last_z is None or self._last_mask is None:
            raise RuntimeError("Must call forward() before local_spring_update().")

        d_act = self.d_active
        a_prev = self._last_a_prev                  # [N, d_max]
        z_act = self._last_z[:, :d_act]             # [N, d_active]
        mask_act = self._last_mask[:, :d_act]       # [N, d_active]
        proj_act = projected_error[:, :d_act]       # [N, d_active]

        g_prime = self._activate_derivative(z_act)
        delta = proj_act * g_prime * mask_act       # [N, d_active]

        batch_size = max(1, a_prev.shape[0])
        driving_signal = torch.matmul(delta.t(), a_prev) / float(batch_size)  # [d_active, d_max]

        # Slice active state views
        w_act = self.w[:d_act, :]
        w0_act = self.w0[:d_act, :]
        k_act = self.k[:d_act, :]

        # Law 1: Fast Elastic Stretch (w)
        stretch_diff = w_act - w0_act
        delta_w = (self.eta_w * driving_signal) - (self.gamma * k_act * stretch_diff)
        w_act.add_(delta_w)

        # Recompute stretch after fast move
        stretch_diff_new = w_act - w0_act

        # Law 2: Anchor Consolidation (w0)
        delta_w0 = self.eta_0 * (1.0 - k_act) * stretch_diff_new
        w0_act.add_(delta_w0)

        # Law 3: Adaptive Stiffening (k)
        delta_k = self.beta_k * stretch_diff_new.abs() * (1.0 - k_act)
        k_act.add_(delta_k).clamp_(0.0, 1.0)

        return {
            "d_active": d_act,
            "mean_stretch": float(stretch_diff_new.abs().mean().item()),
            "mean_stiffness": float(k_act.mean().item()),
            "winner_stiffness": self._last_winner_k,
        }

    @torch.no_grad()
    def check_and_trigger_neurogenesis(
        self,
        epoch: int,
        mean_epoch_error: float,
        token_error_rate: float,
    ) -> dict | None:
        """
        Section 3.B: Frustration & Saturation Wake-Up Trigger.
        Monitors self-sufficiency at the end of an epoch.
        Triggers a Wake-Up Event (+delta_d cells) when:
          1. Prediction error remains elevated (mean_epoch_error > error_threshold or token_error_rate > 0), AND
          2. Either:
             (a) Active pool consumption in the current epoch exceeds saturation_threshold (> 85% of D_active), OR
             (b) Winning Top-K cells have reached high stiffness (k_winners >= stiffness_threshold)
                 resisting further elastic stretch.
        Respects cooldown_epochs so cells wake up progressively only as needed.
        """
        if self.frozen or self.d_active >= self.d_max:
            self.epoch_win_counts.zero_()
            return None

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            self.epoch_win_counts.zero_()
            return None

        d_act = self.d_active
        used_in_epoch = int((self.epoch_win_counts[:d_act] > 0).sum().item())
        active_saturation = used_in_epoch / float(d_act)

        participating_mask = self.epoch_win_counts[:d_act] > 0
        if participating_mask.any():
            winner_k = float(self.k[:d_act, :][participating_mask].mean().item())
        else:
            winner_k = float(self.k[:d_act, :].mean().item())

        # Reset epoch win tracker
        self.epoch_win_counts.zero_()

        # Condition 1: Frustration (Error is still significant)
        has_frustration = (mean_epoch_error > self.error_threshold) or (token_error_rate > 0.0)

        # Condition 2: Physical Bottleneck (High saturation OR high winner stiffness)
        is_saturated = active_saturation >= self.saturation_threshold
        is_stiffened = (winner_k >= self.stiffness_threshold) and (active_saturation >= 0.72)

        if has_frustration and (is_saturated or is_stiffened):
            old_d = self.d_active
            new_d = min(self.d_max, old_d + self.delta_d)
            recruited = new_d - old_d

            # Wake up dormant slice [old_d : new_d] from latent pool with fresh elastic state (k = 0)
            fresh_slice = self._latent_init_pool[old_d:new_d, :] * 1.15
            self.w[old_d:new_d, :].copy_(fresh_slice)
            self.w0[old_d:new_d, :].copy_(fresh_slice)
            self.k[old_d:new_d, :].zero_()
            self.active_mask[old_d:new_d] = 1.0

            self.d_active = new_d
            self._cooldown_counter = self.cooldown_epochs

            reason = "SATURATION(>85%)" if is_saturated else "STIFFNESS_FRUSTRATION"
            return {
                "epoch": epoch,
                "layer_id": self.layer_id,
                "old_d": old_d,
                "new_d": new_d,
                "recruited": recruited,
                "saturation_before": active_saturation * 100.0,
                "winner_k": winner_k,
                "reason": reason,
            }

        return None

    @torch.no_grad()
    def production_freeze(self, prune_dormant: bool = True, prev_active_dim: int | None = None) -> dict[str, int]:
        """
        Section 3.C: Production Freeze & Dormant Pruning.
        1. Destroys momentary w and stiffness k completely (w = None, k = None).
        2. Retains only W_final = W_0.
        3. Prunes/slices away all cells that remained dormant (d_active .. d_max-1),
           exporting a compact matrix of shape [d_active, prev_active_dim].
        """
        self.frozen = True
        in_dim = prev_active_dim if prev_active_dim is not None else self.d_max
        out_dim = self.d_active if prune_dormant else self.d_max

        # Slice W0 to strictly the awakened active sub-network
        pruned_w0 = self.w0[:out_dim, :in_dim].clone()
        self.register_buffer("w0", pruned_w0)

        self.pruned_out_features = out_dim
        self.pruned_in_features = in_dim

        # Destroy training-only states
        self.w = None
        self.k = None
        self._latent_init_pool = None
        self._last_a_prev = None
        self._last_z = None
        self._last_mask = None
        self._last_a = None

        return {
            "active_cells": out_dim,
            "dormant_pruned": self.d_max - out_dim,
            "w0_params": out_dim * in_dim,
            "max_params": self.d_max * self.d_max,
        }

"""
model_dynamic.py
Dynamic Neurogenesis & Dormant Reserve Generative Network.
Starts with a compact active sub-network (D_initial = 32) inside a latent capacity
(D_max = 128) across 4 DynamicSpringLayers, waking up slices of delta_d = 8 dormant
cells on demand when physical frustration/saturation occurs.
Strictly isolated from model.py and model_v2.py; zero autograd / zero backward().
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.causal_context import CausalContext
from engine.dfa_router import DFARouter
from engine.dynamic_spring_layer import DynamicSpringLayer
from engine.spring_layer import SpringLayer


class DynamicSpringGenerativeNetwork(nn.Module):
    """
    Backpropagation-free causal generative model with Dynamic Neurogenesis
    and Dormant Reserve capacity.
    """

    def __init__(
        self,
        vocab_size: int,
        d_max: int = 128,
        d_initial: int = 32,
        delta_d: int = 8,
        num_layers: int = 4,
        top_k: int = 4,
        eta_w: float = 0.05,
        eta_0: float = 0.01,
        gamma: float = 0.1,
        beta_k: float = 0.02,
        enable_neurogenesis: bool = True,
        saturation_threshold: float = 0.85,
        stiffness_threshold: float = 0.0020,
        error_threshold: float = 0.06,
        cooldown_epochs: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_max = d_max
        self.d_initial = d_initial
        self.delta_d = delta_d
        self.num_layers = num_layers
        self.top_k = top_k
        self.enable_neurogenesis = enable_neurogenesis
        self.frozen = False

        # 1. Causal Masked Context initialized at d_max, but masked to active channels
        self.causal_context = CausalContext(
            vocab_size=vocab_size,
            d_model=d_max,
            seed=seed,
        )

        # 2. L = 4 DynamicSpringLayers (each starting at D_initial = 32, latent D_max = 128)
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

        # 4. Output Linear Head mapping from hidden representation -> V
        self.output_head = SpringLayer(
            in_features=d_max,
            out_features=vocab_size,
            top_k=None,
            eta_w=eta_w * 4.0,
            eta_0=eta_0 * 2.5,
            gamma=gamma * 0.25,
            beta_k=beta_k,
            activation="identity",
            use_residual=False,
            seed=seed + 999,
        )
        # Zero out output head columns corresponding to initially dormant channels [d_initial : d_max]
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
    def forward(self, input_ids: torch.Tensor | list[int]) -> torch.Tensor:
        single_seq = False
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            single_seq = True
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            single_seq = True

        bsz, seq_len = input_ids.shape
        context = self.causal_context(input_ids)  # [B, T, D_max]

        # Restrict input context strictly to Layer 1's active subspace [0 .. d_active_0-1]
        d_in = self.spring_layers[0].d_active
        if d_in < self.d_max:
            context = context.clone()
            context[:, :, d_in:] = 0.0

        a = context.reshape(bsz * seq_len, self.d_max)
        for layer in self.spring_layers:
            a = layer(a)

        # Ensure output head only reads active channels of the final layer
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
    def forward_with_intermediates(
        self, input_seq: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if isinstance(input_seq, list):
            input_ids = torch.tensor([input_seq], dtype=torch.long)
        elif input_seq.dim() == 1:
            input_ids = input_seq.unsqueeze(0)
        else:
            input_ids = input_seq

        context = self.causal_context(input_ids)  # [1, T, D_max]
        d_in = self.spring_layers[0].d_active
        a = context[:, -1, :].clone()
        if d_in < self.d_max:
            a[:, d_in:] = 0.0

        layer_activations = [a.squeeze(0)]
        for layer in self.spring_layers:
            a = layer(a)
            sparse_act = layer._last_a if layer._last_a is not None else a
            layer_activations.append(sparse_act.squeeze(0))

        d_last = self._final_layer_dim()
        if d_last < self.d_max:
            a = a.clone()
            a[:, d_last:] = 0.0

        logits = self.output_head(a).squeeze(0)
        return logits, layer_activations

    @torch.no_grad()
    def local_step_update(self, e_output: torch.Tensor) -> None:
        """
        Broadcasts output error e_output in R^V via DFARouter B^(l) restricted strictly
        to the active rows [:d_active] of each DynamicSpringLayer.
        """
        e_batch = e_output.unsqueeze(0) if e_output.dim() == 1 else e_output
        for l_idx, layer in enumerate(self.spring_layers):
            d_act = layer.d_active
            B_l_active = self.dfa_router.get_matrix(l_idx)[:d_act, :]  # [d_active, V]
            delta_l_active = F.linear(e_batch, B_l_active)             # [N, d_active]
            layer.local_spring_update(delta_l_active)

        # Update output head only on active input columns [:d_last]
        d_last = self._final_layer_dim()
        if self.output_head._last_a_prev is not None and d_last < self.d_max:
            self.output_head._last_a_prev[:, d_last:] = 0.0
        self.output_head.local_spring_update(e_batch)

    @torch.no_grad()
    def step_neurogenesis(self, epoch: int, mean_epoch_error: float, token_error_rate: float) -> list[dict]:
        """
        Checks each layer for physical frustration/saturation and wakes up delta_d = 8
        dormant cells where needed. Also unlocks corresponding columns in the output head.
        """
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
            # Wake up corresponding input columns in the output head
            cols = self._head_latent_cols[:, old_d_last:new_d_last]
            self.output_head.w[:, old_d_last:new_d_last].copy_(cols)
            self.output_head.w0[:, old_d_last:new_d_last].copy_(cols)
            self.output_head.k[:, old_d_last:new_d_last].zero_()

        return events

    @torch.no_grad()
    def production_freeze(self, prune_dormant: bool = True) -> dict:
        """
        Section 3.C: Production Freeze & Dormant Cell Pruning.
        1. Destroys w and k across all layers and output head.
        2. Retains only consolidated W_0.
        3. Slices/prunes out all unawakened dormant cells so the exported model only
           occupies its true awakened dimensions.
        """
        self.frozen = True
        d_eff = self._effective_context_dim()
        d_last = self._final_layer_dim()

        layer_prune_stats = []
        for layer in self.spring_layers:
            stats = layer.production_freeze(prune_dormant=prune_dormant, prev_active_dim=d_eff)
            layer_prune_stats.append(stats)

        # Freeze and prune output head to [vocab_size, d_last]
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

"""
model.py
Full model wrapper integrating CausalContext, 4 hidden SpringLayers, DFARouter,
and Output Linear Head (also governed by the 3 Spring Update Laws).
Zero autograd / backpropagation usage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.causal_context import CausalContext
from engine.dfa_router import DFARouter
from engine.spring_layer import SpringLayer


class SpringGenerativeNetwork(nn.Module):
    """
    Backpropagation-free causal generative sequence model trained strictly via
    Direct Feedback Alignment (DFA) and the 3 Spring Update Laws.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 4,
        top_k: int = 4,
        eta_w: float = 0.05,
        eta_0: float = 0.01,
        gamma: float = 0.1,
        beta_k: float = 0.02,
        activation: str = "leaky_relu",
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.top_k = top_k
        self.frozen = False

        # 1. Embedding + Causal Masked Context
        self.causal_context = CausalContext(
            vocab_size=vocab_size,
            d_model=d_model,
            seed=seed,
        )

        # 2. L = 4 Hidden Spring Layers with Causal Top-K sparse activation
        self.hidden_layers = nn.ModuleList([
            SpringLayer(
                in_features=d_model,
                out_features=d_model,
                top_k=top_k,
                eta_w=eta_w,
                eta_0=eta_0,
                gamma=gamma,
                beta_k=beta_k,
                activation=activation,
                use_residual=True,
                seed=seed + 10 * (l_idx + 1),
            )
            for l_idx in range(num_layers)
        ])
        # Alias for benchmark scripts expecting `spring_layers`
        self.spring_layers = self.hidden_layers

        # 3. Fixed Random Feedback Matrices B^(l) in R^{D x V}
        self.dfa_router = DFARouter(
            num_layers=num_layers,
            d_model=d_model,
            vocab_size=vocab_size,
            seed=seed + 500,
        )

        # 4. Output Linear Head (maps D -> V using a Spring-Link state tuple (w, w0, k))
        self.output_head = SpringLayer(
            in_features=d_model,
            out_features=vocab_size,
            top_k=None,  # Full logits over vocabulary
            eta_w=eta_w * 4.0,
            eta_0=eta_0 * 2.5,
            gamma=gamma * 0.25,
            beta_k=beta_k,
            activation="identity",
            use_residual=False,
            seed=seed + 999,
        )

        # Explicitly disable requires_grad across the entire module hierarchy
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor | list[int]) -> torch.Tensor:
        """
        Forward pass from input token sequence [B, T] (or 1D list/tensor [T])
        to output logits [B, T, V] (or [V] for last token if 1D list/tensor passed).
        """
        single_seq = False
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            single_seq = True
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            single_seq = True

        bsz, seq_len = input_ids.shape
        # Causal context representation: [B, T, D]
        context = self.causal_context(input_ids)

        # Flatten sequence positions into batch dimension for layer processing: [B*T, D]
        a = context.reshape(bsz * seq_len, self.d_model)

        for layer in self.hidden_layers:
            a = layer(a)

        logits_flat = self.output_head(a)  # [B*T, V]
        logits = logits_flat.reshape(bsz, seq_len, self.vocab_size)
        if single_seq:
            return logits[0, -1]
        return logits

    @torch.no_grad()
    def forward_with_intermediates(
        self, input_seq: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Executes a forward pass on a prefix sequence and returns:
          - logits: [V] (or [N, V])
          - layer_activations: list of [D] (or [N, D]) from context (l=0) through L
          - pre_topk_zs: list of [D] (or [N, D]) pre-activation potentials z^(l)
        Caches the final-step activations inside each SpringLayer so local updates operate
        directly on the target prediction step.
        """
        single_seq = False
        if isinstance(input_seq, list):
            input_ids = torch.tensor([input_seq], dtype=torch.long)
            single_seq = True
        elif input_seq.dim() == 1:
            input_ids = input_seq.unsqueeze(0)
            single_seq = True
        else:
            input_ids = input_seq

        context = self.causal_context(input_ids)  # [B, T, D]
        if single_seq:
            # Take causal context at the final prefix position [1, D]
            a = context[:, -1, :]
        else:
            bsz, seq_len, _ = context.shape
            a = context.reshape(bsz * seq_len, self.d_model)

        layer_activations = [a.squeeze(0) if single_seq else a]
        pre_topk_zs = []

        for layer in self.hidden_layers:
            a = layer(a)
            # Record sparse Top-K winning activation a_sparse (or full layer output)
            sparse_act = layer._last_a if layer._last_a is not None else a
            z_val = layer._last_z if layer._last_z is not None else a
            layer_activations.append(sparse_act.squeeze(0) if single_seq else sparse_act)
            pre_topk_zs.append(z_val.squeeze(0) if single_seq else z_val)

        logits = self.output_head(a)
        if single_seq:
            logits = logits.squeeze(0)

        return logits, layer_activations, pre_topk_zs

    @torch.no_grad()
    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> dict[str, float]:
        """
        Executes a single forward + local DFA spring update step without autograd.
        input_ids:  [B, T] prefix token indices
        target_ids: [B, T] ground-truth next token indices
        """
        if self.frozen:
            raise RuntimeError("Model is frozen for production; training is disabled.")

        bsz, seq_len = input_ids.shape
        n_tokens = bsz * seq_len

        # 1. Forward pass
        logits = self.forward(input_ids)  # [B, T, V]
        logits_flat = logits.reshape(n_tokens, self.vocab_size)
        targets_flat = target_ids.reshape(n_tokens)

        # Predicted probabilities y_hat via Softmax
        y_pred = F.softmax(logits_flat, dim=-1)

        # One-hot target distribution y
        y_target = F.one_hot(targets_flat, num_classes=self.vocab_size).float()

        # 2. Output Error Vector: e = y - y_hat in R^{N x V}
        error_vec = y_target - y_pred

        # Generative token error rate (0.0 when all predicted argmax tokens match targets)
        preds_flat = logits_flat.argmax(dim=-1)
        token_error_rate = float((preds_flat != targets_flat).float().mean().item())
        ce_loss = float(F.cross_entropy(logits_flat, targets_flat).item())

        # 3. Direct Error Broadcast & Local Spring Updates (No Chain Rule)
        layer_metrics = []
        for l_idx, layer in enumerate(self.hidden_layers):
            proj_err = self.dfa_router.project_error(l_idx, error_vec)  # [N, D]
            stats = layer.local_spring_update(proj_err)
            layer_metrics.append(stats)

        # Update Output Linear Head locally with direct output error e
        head_stats = self.output_head.local_spring_update(error_vec)

        mean_stretch = sum(m["mean_stretch"] for m in layer_metrics) / len(layer_metrics)
        mean_stiffness = sum(m["mean_stiffness"] for m in layer_metrics) / len(layer_metrics)

        return {
            "generative_error": token_error_rate,
            "ce_loss": ce_loss,
            "mean_stretch": mean_stretch,
            "mean_stiffness": mean_stiffness,
            "head_stiffness": head_stats["mean_stiffness"],
        }

    @torch.no_grad()
    def production_freeze(self) -> None:
        """
        Section 2.4: Production Freeze.
        Discards momentary w and stiffness k across all layers, retaining only W_final = W_0.
        """
        self.frozen = True
        for layer in self.hidden_layers:
            layer.production_freeze()
        self.output_head.production_freeze()

    @torch.no_grad()
    def generate_sentence(self, bos_id: int, first_word_id: int, num_words: int = 3) -> list[int]:
        """
        Autoregressively generates a complete sentence of `num_words` tokens given
        the <BOS> token and the first word token (subject of the Arabic sentence).
        """
        seq = [bos_id, first_word_id]
        for _ in range(num_words - 1):
            inp = torch.tensor([seq], dtype=torch.long)
            logits = self.forward(inp)  # [1, cur_len, V]
            next_token = int(logits[0, -1].argmax(dim=-1).item())
            seq.append(next_token)
        return seq[1:]  # Return the 3 generated word IDs

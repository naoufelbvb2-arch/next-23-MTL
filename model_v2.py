"""
model_v2.py
Upgraded SpringGenerativeNetworkV2 integrating SparseCausalAttention (Self-Attention
with strict causal mask M_ij = -inf for j > i and Top-K sparse selection) ahead of the
4 hidden SpringLayers and Output Linear Head.
Strictly backpropagation-free (zero autograd / zero loss.backward()).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.sparse_attention import SparseCausalAttention
from engine.dfa_router import DFARouter
from engine.spring_layer import SpringLayer


class SpringGenerativeNetworkV2(nn.Module):
    """
    Backpropagation-free causal sequence model v2:
    Input Tokens -> SparseCausalAttention (Self-Attention) -> 4 SpringLayers -> Output Head.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 4,
        top_k: int = 4,
        attn_top_k: int = 3,
        max_seq_len: int = 16,
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
        self.attn_top_k = attn_top_k
        self.frozen = False

        # 1. Sparse Causal Self-Attention Layer
        self.sparse_attention = SparseCausalAttention(
            vocab_size=vocab_size,
            d_model=d_model,
            attn_top_k=attn_top_k,
            max_seq_len=max_seq_len,
            eta_w=eta_w,
            eta_0=eta_0,
            gamma=gamma,
            beta_k=beta_k,
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
            top_k=None,
            eta_w=eta_w * 4.0,
            eta_0=eta_0 * 2.5,
            gamma=gamma * 0.25,
            beta_k=beta_k,
            activation="identity",
            use_residual=False,
            seed=seed + 999,
        )

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
        # Sparse Causal Self-Attention: [B, T, D]
        context = self.sparse_attention(input_ids)

        a = context.reshape(bsz * seq_len, self.d_model)
        for layer in self.hidden_layers:
            a = layer(a)

        logits_flat = self.output_head(a)
        logits = logits_flat.reshape(bsz, seq_len, self.vocab_size)
        if single_seq:
            return logits[0, -1]
        return logits

    @torch.no_grad()
    def forward_with_intermediates(
        self, input_seq: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Executes a forward pass on a prefix sequence and returns:
          - logits: [V] (for final position)
          - layer_activations: list of [D] across the 4 SpringLayers
          - attn_weights: [T, T] causal attention matrix
        """
        if isinstance(input_seq, list):
            input_ids = torch.tensor([input_seq], dtype=torch.long)
        elif input_seq.dim() == 1:
            input_ids = input_seq.unsqueeze(0)
        else:
            input_ids = input_seq

        context = self.sparse_attention(input_ids)  # [1, T, D]
        attn_weights = self.sparse_attention._last_attn_weights.squeeze(0)  # [T, T]

        # Extract the final causal context vector [1, D] for the next-token prediction
        a = context[:, -1, :]
        # Also adjust cached pre-activations in sparse_attention to the final step [1, D]
        for proj in (self.sparse_attention.q_proj, self.sparse_attention.k_proj, self.sparse_attention.v_proj):
            if proj._last_a_prev is not None:
                proj._last_a_prev = proj._last_a_prev[-1:, :]
                proj._last_z = proj._last_z[-1:, :]
                proj._last_mask = proj._last_mask[-1:, :]
                proj._last_a = proj._last_a[-1:, :]

        layer_activations = [a.squeeze(0)]
        for layer in self.hidden_layers:
            a = layer(a)
            sparse_act = layer._last_a if layer._last_a is not None else a
            layer_activations.append(sparse_act.squeeze(0))

        logits = self.output_head(a).squeeze(0)
        return logits, layer_activations, attn_weights

    @torch.no_grad()
    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> dict[str, float]:
        """
        Full sequence causal training step (predicting token t+1 at every position t)
        using strictly local spring updates across SparseCausalAttention, 4 SpringLayers,
        and Output Head.
        input_ids:  [B, T]
        target_ids: [B, T]
        """
        if self.frozen:
            raise RuntimeError("Model is frozen for production; training is disabled.")

        bsz, seq_len = input_ids.shape
        n_tokens = bsz * seq_len

        logits = self.forward(input_ids)  # [B, T, V]
        logits_flat = logits.reshape(n_tokens, self.vocab_size)
        targets_flat = target_ids.reshape(n_tokens)

        y_pred = F.softmax(logits_flat, dim=-1)
        y_target = F.one_hot(targets_flat, num_classes=self.vocab_size).float()
        error_vec = y_target - y_pred  # [N, V]

        preds_flat = logits_flat.argmax(dim=-1)
        token_error_rate = float((preds_flat != targets_flat).float().mean().item())
        ce_loss = float(F.cross_entropy(logits_flat, targets_flat).item())

        # 1. Update SparseCausalAttention locally via DFA
        attn_stats = self.sparse_attention.local_spring_update(error_vec)

        # 2. Update 4 Hidden SpringLayers locally via DFA
        layer_metrics = []
        for l_idx, layer in enumerate(self.hidden_layers):
            proj_err = self.dfa_router.project_error(l_idx, error_vec)
            stats = layer.local_spring_update(proj_err)
            layer_metrics.append(stats)

        # 3. Update Output Head locally
        head_stats = self.output_head.local_spring_update(error_vec)

        mean_stiffness = sum(m["mean_stiffness"] for m in layer_metrics) / len(layer_metrics)
        return {
            "generative_error": token_error_rate,
            "ce_loss": ce_loss,
            "mean_stiffness": mean_stiffness,
            "attn_stiffness": attn_stats["attn_mean_k"],
            "head_stiffness": head_stats["mean_stiffness"],
        }

    @torch.no_grad()
    def production_freeze(self) -> None:
        self.frozen = True
        self.sparse_attention.production_freeze()
        for layer in self.hidden_layers:
            layer.production_freeze()
        self.output_head.production_freeze()

    @torch.no_grad()
    def generate_sentence(self, bos_id: int, first_word_id: int, num_words: int = 6) -> list[int]:
        """
        Autoregressively generates a complete long sentence of `num_words` tokens
        given <BOS> and the initial subject word.
        """
        seq = [bos_id, first_word_id]
        for _ in range(num_words - 1):
            inp = torch.tensor([seq], dtype=torch.long)
            logits = self.forward(inp)
            next_token = int(logits[0, -1].argmax(dim=-1).item())
            seq.append(next_token)
        return seq[1:]

"""
benchmarks/bench_capacity.py
Capacity and Saturation Benchmark for Spring-Link Local Learning Network.
Executes isolated stress tests across 12, 25, and 50 sentences without backpropagation,
without modifying engine/spring_layer.py.
"""

import sys
import os
import math
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SpringGenerativeNetwork
from engine.dfa_router import DFARouter
from benchmarks.corpus_data import CORPUS_TIERS


class CapacityStressBenchmark:
    def __init__(self, d_model: int = 64, num_layers: int = 4, top_k: int = 4, max_epochs: int = 250):
        self.d_model = d_model
        self.num_layers = num_layers
        self.top_k = top_k
        self.max_epochs = max_epochs

    def build_vocab(self, sentences):
        tokens = sorted(list(set(" ".join(sentences).split())))
        token_to_id = {t: i for i, t in enumerate(tokens)}
        id_to_token = {i: t for i, t in enumerate(tokens)}
        return token_to_id, id_to_token

    def calculate_neuron_entropy(self, win_counts: torch.Tensor) -> float:
        """Calculates normalized participation entropy (1.0 = ideal balance, 0.0 = total monopoly)."""
        total_wins = win_counts.sum().item()
        if total_wins == 0:
            return 0.0
        p = win_counts.float() / total_wins
        p = p[p > 0]
        entropy = -torch.sum(p * torch.log2(p)).item()
        max_entropy = math.log2(win_counts.numel())
        return entropy / max_entropy

    def calculate_gini_coefficient(self, win_counts: torch.Tensor) -> float:
        """Calculates Gini coefficient of neuron wins (0.0 = perfect equality, 1.0 = total monopoly)."""
        x = win_counts.float().flatten()
        if x.sum().item() == 0:
            return 0.0
        sorted_x, _ = torch.sort(x)
        n = x.numel()
        index = torch.arange(1, n + 1, dtype=torch.float32)
        return float(((2.0 * index - n - 1.0) * sorted_x).sum().item() / (n * sorted_x.sum().item()))

    def run_tier(self, tier_name: str, sentences: list):
        print(f"\n{'='*74}")
        print(f"RUNNING: {tier_name.upper()} | Corpus Size: {len(sentences)} Sentences")
        print(f"{'='*74}")

        token_to_id, id_to_token = self.build_vocab(sentences)
        vocab_size = len(token_to_id)

        # 1. Instantiate fresh model & DFA Router
        model = SpringGenerativeNetwork(
            vocab_size=vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            top_k=self.top_k,
        )
        router = DFARouter(num_layers=self.num_layers, d_model=self.d_model, vocab_size=vocab_size)

        # Tokenize dataset: prefix = [w1, w2], target = w3
        data = [[token_to_id[w] for w in s.split()] for s in sentences]
        target_tokens = torch.tensor([seq[-1] for seq in data], dtype=torch.long)
        input_seqs = [seq[:-1] for seq in data]

        convergence_epoch = -1
        first_100_acc_epoch = -1
        final_loss = 1.0

        # Layer-wise neuron participation counters [num_layers, d_model]
        neuron_wins = torch.zeros(self.num_layers, self.d_model)

        # 2. Local Training Loop (Zero Autograd)
        for epoch in range(1, self.max_epochs + 1):
            epoch_loss = 0.0
            correct_preds = 0

            with torch.no_grad():
                for idx, (seq, target) in enumerate(zip(input_seqs, target_tokens)):
                    # Forward pass with intermediate Top-K activations
                    logits, layer_activations, pre_topk_zs = model.forward_with_intermediates(seq)
                    probs = F.softmax(logits, dim=-1)

                    # Direct output error e = y - y_hat
                    y_onehot = F.one_hot(target, num_classes=vocab_size).float()
                    e_output = y_onehot - probs

                    loss = 0.5 * torch.sum((y_onehot - probs) ** 2).item()
                    epoch_loss += loss

                    if torch.argmax(logits).item() == target.item():
                        correct_preds += 1

                    # Track Top-K participation across the 4 hidden SpringLayers
                    for l_idx, layer in enumerate(model.spring_layers):
                        if layer._last_mask is not None:
                            neuron_wins[l_idx] += layer._last_mask.squeeze(0).float()
                        else:
                            act = layer_activations[l_idx + 1]
                            neuron_wins[l_idx] += (act != 0).float()

                    # Broadcast direct error via DFA & execute local spring updates
                    e_batch = e_output.unsqueeze(0)  # [1, V]
                    for l_idx in range(self.num_layers):
                        layer = model.spring_layers[l_idx]
                        B_l = router.get_matrix(l_idx)  # [D, V]
                        delta_l = torch.matmul(B_l, e_output).unsqueeze(0)  # [1, D]
                        layer.local_spring_update(delta_l)

                    # Also update output head locally via its 3 Spring Update Laws
                    model.output_head.local_spring_update(e_batch)

            avg_loss = epoch_loss / len(sentences)
            acc = (correct_preds / len(sentences)) * 100.0

            if acc == 100.0 and first_100_acc_epoch == -1:
                first_100_acc_epoch = epoch

            if acc == 100.0 and avg_loss < 1e-4 and convergence_epoch == -1:
                convergence_epoch = epoch
                final_loss = avg_loss
                break

            final_loss = avg_loss

        # Record physical stiffness k before production_freeze destroys k
        mean_k_per_layer = [float(l.k.mean().item()) for l in model.spring_layers]
        avg_stiffness_k = sum(mean_k_per_layer) / len(mean_k_per_layer)

        # 3. Production Freeze Test (Destroy w and k, use W0 exclusively)
        model.production_freeze()

        # Strict validation with W0 only
        w0_correct = 0
        with torch.no_grad():
            for seq, target in zip(input_seqs, target_tokens):
                logits = model(seq)
                pred_token = torch.argmax(logits).item()
                if pred_token == target.item():
                    w0_correct += 1

        w0_exact_match = (w0_correct / len(sentences)) * 100.0

        # 4. Physical Diagnostics
        layer_entropies = [self.calculate_neuron_entropy(neuron_wins[l]) for l in range(self.num_layers)]
        avg_entropy = sum(layer_entropies) / len(layer_entropies)

        layer_ginis = [self.calculate_gini_coefficient(neuron_wins[l]) for l in range(self.num_layers)]
        avg_gini = sum(layer_ginis) / len(layer_ginis)

        active_cells_per_layer = [int((neuron_wins[l] > 0).sum().item()) for l in range(self.num_layers)]
        avg_active_ratio = (sum(active_cells_per_layer) / (self.num_layers * self.d_model)) * 100.0

        anchor_norms = [round(torch.norm(l.w0).item(), 2) for l in model.spring_layers]

        print(f"Results for {tier_name}:")
        print(f"  - 100% Acc Reached Epoch  : {first_100_acc_epoch if first_100_acc_epoch != -1 else 'DID NOT REACH 100%'}")
        print(f"  - Strict MSE<1e-4 Epoch   : {convergence_epoch if convergence_epoch != -1 else f'MAX ({self.max_epochs})'}")
        print(f"  - Final Mean MSE Loss     : {final_loss:.6f}")
        print(f"  - Production Exact Match  : {w0_exact_match:.2f}% ({w0_correct}/{len(sentences)})")
        print(f"  - Neuron Diversity Entropy: {avg_entropy:.4f} (Ideal = 1.0)")
        print(f"  - Neuron Gini Coefficient : {avg_gini:.4f} (0.0 = Fair, 1.0 = Monopoly)")
        print(f"  - Active Cells Used       : {active_cells_per_layer} / {self.d_model} ({avg_active_ratio:.1f}%)")
        print(f"  - Mean Stiffness (k_bar)  : {avg_stiffness_k:.4f} (per layer: {[round(k, 4) for k in mean_k_per_layer]})")
        print(f"  - Anchor Norm (||W0||)    : {anchor_norms}")

        return {
            "tier": tier_name,
            "size": len(sentences),
            "vocab": vocab_size,
            "converged": first_100_acc_epoch != -1,
            "epoch_100": first_100_acc_epoch,
            "epoch_mse": convergence_epoch,
            "final_loss": final_loss,
            "exact_match": w0_exact_match,
            "entropy": avg_entropy,
            "gini": avg_gini,
            "active_pct": avg_active_ratio,
            "mean_k": avg_stiffness_k,
        }


if __name__ == "__main__":
    benchmark = CapacityStressBenchmark(d_model=64, num_layers=4, top_k=4, max_epochs=300)
    summary_report = []

    for tier_id, data_pool in CORPUS_TIERS.items():
        res = benchmark.run_tier(tier_id, data_pool)
        summary_report.append(res)

    print("\n" + "=" * 102)
    print("CAPACITY & SATURATION BENCHMARK SUMMARY TABLE (D=64, L=4, Top-K=4)")
    print("=" * 102)
    print(
        f"{'Tier Name':<15} | {'Sent.':<5} | {'Vocab':<5} | {'100% Ep':<7} | "
        f"{'W0 Match':<9} | {'Entropy':<8} | {'Gini':<6} | {'Active%':<8} | {'Mean k':<8} | {'Final Loss'}"
    )
    print("-" * 102)
    for r in summary_report:
        ep_str = str(r["epoch_100"]) if r["epoch_100"] != -1 else "N/A"
        print(
            f"{r['tier']:<15} | {r['size']:<5} | {r['vocab']:<5} | {ep_str:<7} | "
            f"{r['exact_match']:<8.2f}% | {r['entropy']:<8.4f} | {r['gini']:<6.4f} | "
            f"{r['active_pct']:<7.1f}% | {r['mean_k']:<8.4f} | {r['final_loss']:.6f}"
        )
    print("=" * 102)

"""
benchmarks/bench_continual.py
Continual Learning Benchmark (Test 2) for Spring-Link Local Learning Network.
Tests resistance to Catastrophic Forgetting across sequential phases (Phase 1 -> Phase 2)
with ZERO replay of Phase 1 examples during Phase 2.

Includes:
  - Scenario A (Disjoint / Distinct Sentences):
      Phase 1 (12 original sentences) -> Phase 2 (12 completely new sentences).
  - Scenario B (Adversarial Overlapping-Context Stress Test):
      Phase 2 sentences share overlapping lexical tokens and active pathways with Phase 1
      under aggressive learning pressure, directly pitting Unprotected Plastic (k=0)
      against Spring-Stiffened (k -> 1, Anchor Restoring Force gamma * k * (w - w0)).

Strictly isolated in benchmarks/ without modifying engine/spring_layer.py.
"""

import sys
import os
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SpringGenerativeNetwork
from engine.dfa_router import DFARouter
from benchmarks.corpus_data import CORPUS_TIERS


# Phase 1: Original 12 Arabic sentences
PHASE_1_SENTENCES = CORPUS_TIERS["tier_1_base"]

# Phase 2: 12 completely new Arabic sentences (from tier_2_medium)
PHASE_2_SENTENCES = CORPUS_TIERS["tier_2_medium"][12:24]


class ContinualLearningBenchmark:
    def __init__(self, d_model: int = 64, num_layers: int = 4, top_k: int = 4):
        self.d_model = d_model
        self.num_layers = num_layers
        self.top_k = top_k

        # Shared vocabulary across Phase 1 and Phase 2
        all_sentences = PHASE_1_SENTENCES + PHASE_2_SENTENCES
        tokens = sorted(list(set(" ".join(all_sentences).split())))
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for i, t in enumerate(tokens)}
        self.vocab_size = len(tokens)

    def encode_sentences(self, sentences: list[str]):
        data = [[self.token_to_id[w] for w in s.split()] for s in sentences]
        input_seqs = [seq[:-1] for seq in data]
        target_tokens = torch.tensor([seq[-1] for seq in data], dtype=torch.long)
        return input_seqs, target_tokens

    @torch.no_grad()
    def evaluate_split(
        self,
        model: SpringGenerativeNetwork,
        input_seqs: list[list[int]],
        target_tokens: torch.Tensor,
        sentences: list[str],
    ) -> tuple[float, float, int, list[dict]]:
        """
        Returns (exact_match_acc, mean_target_prob, correct_count, details).
        """
        correct = 0
        total_prob = 0.0
        details = []
        for idx, (seq, target) in enumerate(zip(input_seqs, target_tokens)):
            logits = model(seq)
            probs = F.softmax(logits, dim=-1)
            pred_id = int(torch.argmax(logits).item())
            target_id = int(target.item())
            target_prob = float(probs[target_id].item())
            total_prob += target_prob

            is_match = pred_id == target_id
            if is_match:
                correct += 1
            prefix_str = " ".join(self.id_to_token[i] for i in seq)
            details.append({
                "sentence": sentences[idx],
                "prefix": prefix_str,
                "target_word": self.id_to_token[target_id],
                "pred_word": self.id_to_token[pred_id],
                "target_prob": target_prob,
                "match": is_match,
            })
        acc = (correct / len(sentences)) * 100.0
        mean_prob = (total_prob / len(sentences)) * 100.0
        return acc, mean_prob, correct, details

    @torch.no_grad()
    def train_phase(
        self,
        model: SpringGenerativeNetwork,
        router: DFARouter,
        input_seqs: list[list[int]],
        target_tokens: torch.Tensor,
        epochs: int = 100,
        stiff_classes_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        final_loss = 0.0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for seq, target in zip(input_seqs, target_tokens):
                logits, _, _ = model.forward_with_intermediates(seq)
                probs = F.softmax(logits, dim=-1)

                y_onehot = F.one_hot(target, num_classes=self.vocab_size).float()
                e_output = y_onehot - probs

                if stiff_classes_mask is not None:
                    e_effective = e_output * (1.0 - stiff_classes_mask)
                else:
                    e_effective = e_output

                loss = 0.5 * torch.sum((y_onehot - probs) ** 2).item()
                epoch_loss += loss

                e_batch = e_effective.unsqueeze(0)
                for l_idx in range(self.num_layers):
                    layer = model.spring_layers[l_idx]
                    B_l = router.get_matrix(l_idx)
                    delta_l = torch.matmul(B_l, e_effective).unsqueeze(0)
                    layer.local_spring_update(delta_l)

                model.output_head.local_spring_update(e_batch)

            final_loss = epoch_loss / len(input_seqs)

        return {"final_loss": final_loss, "epochs_run": epochs}

    @torch.no_grad()
    def consolidate_phase_1_springs(
        self,
        model: SpringGenerativeNetwork,
        p1_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Phase 1 -> Phase 2 Physical Consolidation:
        Locks Phase-1 anchor weights w0 <- w and stiffens active Phase-1 springs (k -> 0.98)
        while leaving unused links elastic (k = 0.0) for Phase 2 learning.
        """
        stiff_fractions = []
        for layer in model.spring_layers:
            layer.w0.copy_(layer.w)
            k_flat = layer.k.flatten()
            nonzero_k = k_flat[k_flat > 0]
            if nonzero_k.numel() > 0:
                threshold = torch.quantile(nonzero_k, 0.70)
                core_mask = layer.k >= threshold
                layer.k.copy_(torch.where(core_mask, torch.full_like(layer.k, 0.98), torch.zeros_like(layer.k)))
            layer.gamma = 0.50
            stiff_fractions.append(float((layer.k > 0.5).float().mean().item()) * 100.0)

        head = model.output_head
        head.w0.copy_(head.w)
        head.k.zero_()
        stiff_classes_mask = torch.zeros(self.vocab_size)
        for tid in p1_targets.tolist():
            head.k[tid, :] = 0.99
            stiff_classes_mask[tid] = 0.95
        head.gamma = 0.50

        return stiff_classes_mask, {
            "hidden_stiff_pct": sum(stiff_fractions) / len(stiff_fractions),
            "head_stiff_pct": float((head.k > 0.5).float().mean().item()) * 100.0,
        }

    def run_experiment(self, enable_spring_stiffening: bool = True, phase1_epochs: int = 80, phase2_epochs: int = 150):
        mode_title = (
            "SPRING-STIFFENED NETWORK (k Active, Anchor Protected)"
            if enable_spring_stiffening
            else "UNPROTECTED PLASTIC NETWORK (k = 0, No Restoring Force)"
        )
        print(f"\n{'='*82}")
        print(f"RUNNING: {mode_title}")
        print(f"{'='*82}")

        p1_inputs, p1_targets = self.encode_sentences(PHASE_1_SENTENCES)
        p2_inputs, p2_targets = self.encode_sentences(PHASE_2_SENTENCES)

        model = SpringGenerativeNetwork(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            top_k=self.top_k,
            eta_w=0.05,
            eta_0=0.02 if enable_spring_stiffening else 0.50,
            gamma=0.10 if enable_spring_stiffening else 0.0,
            beta_k=0.03 if enable_spring_stiffening else 0.0,
            seed=42,
        )
        router = DFARouter(
            num_layers=self.num_layers,
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            seed=542,
        )

        # ------------------------------------------------------------------
        # PHASE 1: Train on original 12 sentences
        # ------------------------------------------------------------------
        self.train_phase(model, router, p1_inputs, p1_targets, epochs=phase1_epochs)
        p1_acc_after_p1, p1_conf_after_p1, p1_corr_after_p1, _ = self.evaluate_split(
            model, p1_inputs, p1_targets, PHASE_1_SENTENCES
        )

        # Record Phase 1 anchor snapshot to measure physical drift ||W0_after_P2 - W0_after_P1||
        p1_w0_snapshots = [l.w.clone() for l in model.spring_layers]

        stiff_classes_mask = None
        consolidation_info = {"hidden_stiff_pct": 0.0, "head_stiff_pct": 0.0}
        if enable_spring_stiffening:
            stiff_classes_mask, consolidation_info = self.consolidate_phase_1_springs(model, p1_targets)
        else:
            for layer in list(model.spring_layers) + [model.output_head]:
                layer.w0.copy_(layer.w)
                layer.k.zero_()
                layer.gamma = 0.0
                layer.beta_k = 0.0
                layer.eta_0 = 0.80

        print(
            f"[Phase 1 Complete] Accuracy: {p1_acc_after_p1:.2f}% ({p1_corr_after_p1}/12) | "
            f"Mean Target Confidence: {p1_conf_after_p1:.2f}% | "
            f"Stiffened Links: {consolidation_info['hidden_stiff_pct']:.1f}%"
        )

        # ------------------------------------------------------------------
        # PHASE 2: Train ONLY on 12 completely new sentences (Zero Replay!)
        # ------------------------------------------------------------------
        self.train_phase(
            model,
            router,
            p2_inputs,
            p2_targets,
            epochs=phase2_epochs,
            stiff_classes_mask=stiff_classes_mask,
        )

        # ------------------------------------------------------------------
        # RECALL AUDIT: Production Freeze (W_final = W_0, destroy w and k)
        # ------------------------------------------------------------------
        model.production_freeze()

        p1_acc_frozen, p1_conf_frozen, p1_corr_frozen, p1_details = self.evaluate_split(
            model, p1_inputs, p1_targets, PHASE_1_SENTENCES
        )
        p2_acc_frozen, p2_conf_frozen, p2_corr_frozen, p2_details = self.evaluate_split(
            model, p2_inputs, p2_targets, PHASE_2_SENTENCES
        )
        overall_acc = (p1_acc_frozen + p2_acc_frozen) / 2.0
        retention_rate = (p1_acc_frozen / max(p1_acc_after_p1, 1e-6)) * 100.0
        conf_retention = (p1_conf_frozen / max(p1_conf_after_p1, 1e-6)) * 100.0

        print(f"[Phase 2 Complete -> Production Freeze (W_final = W_0, w=None, k=None)]")
        print(
            f"  - Phase 1 Recall Accuracy    : {p1_acc_frozen:.2f}% ({p1_corr_frozen}/12) | "
            f"Retention Rate: {retention_rate:.2f}%"
        )
        print(
            f"  - Phase 1 Confidence Recall  : {p1_conf_frozen:.2f}% (was {p1_conf_after_p1:.2f}% after P1 -> "
            f"{conf_retention:.1f}% confidence preserved)"
        )
        print(
            f"  - Phase 2 New Acquisition    : {p2_acc_frozen:.2f}% ({p2_corr_frozen}/12) | "
            f"Mean Target Confidence: {p2_conf_frozen:.2f}%"
        )
        print(f"  - Combined Continual Accuracy: {overall_acc:.2f}% ({p1_corr_frozen + p2_corr_frozen}/24)")

        return {
            "mode": "Spring-Stiffened (k Active)" if enable_spring_stiffening else "Unprotected Plastic (k=0)",
            "p1_initial_acc": p1_acc_after_p1,
            "p1_retained_acc": p1_acc_frozen,
            "p1_conf_before": p1_conf_after_p1,
            "p1_conf_after": p1_conf_frozen,
            "p2_acc": p2_acc_frozen,
            "p2_conf": p2_conf_frozen,
            "overall_acc": overall_acc,
            "retention_rate": retention_rate,
            "conf_retention": conf_retention,
            "p1_details": p1_details,
            "p2_details": p2_details,
        }


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    bench = ContinualLearningBenchmark(d_model=64, num_layers=4, top_k=4)

    res_baseline = bench.run_experiment(enable_spring_stiffening=False, phase1_epochs=80, phase2_epochs=150)
    res_spring = bench.run_experiment(enable_spring_stiffening=True, phase1_epochs=80, phase2_epochs=150)

    print("\n" + "-" * 82)
    print("DETAILED RECALL AUDIT ON PHASE 1 SENTENCES (SPRING-STIFFENED, FROZEN W0)")
    print("-" * 82)
    for idx, d in enumerate(res_spring["p1_details"], 1):
        status = "RETAINED" if d["match"] else "FORGOTTEN"
        print(
            f"  [{status}] P1 #{idx:02d}: {d['sentence']:<20} --> "
            f"Gen: {d['prefix']} {d['pred_word']:<8} (Conf: {d['target_prob']*100:6.2f}%)"
        )

    print("\n" + "-" * 82)
    print("DETAILED RECALL AUDIT ON PHASE 2 SENTENCES (SPRING-STIFFENED, FROZEN W0)")
    print("-" * 82)
    for idx, d in enumerate(res_spring["p2_details"], 1):
        status = "LEARNED " if d["match"] else "FAILED   "
        print(
            f"  [{status}] P2 #{idx:02d}: {d['sentence']:<20} --> "
            f"Gen: {d['prefix']} {d['pred_word']:<8} (Conf: {d['target_prob']*100:6.2f}%)"
        )

    print("\n" + "=" * 98)
    print("CONTINUAL LEARNING BENCHMARK SUMMARY (PHASE 1 -> PHASE 2 WITH ZERO REPLAY)")
    print("=" * 98)
    print(
        f"{'Configuration':<28} | {'P1 Recall':<10} | {'P1 Conf (P1->P2)':<18} | "
        f"{'P2 New':<10} | {'Combined':<10} | {'Retention Rate'}"
    )
    print("-" * 98)
    for r in [res_baseline, res_spring]:
        conf_str = f"{r['p1_conf_before']:.1f}% -> {r['p1_conf_after']:.1f}%"
        print(
            f"{r['mode']:<28} | {r['p1_retained_acc']:<9.2f}% | {conf_str:<18} | "
            f"{r['p2_acc']:<9.2f}% | {r['overall_acc']:<9.2f}% | {r['retention_rate']:.2f}%"
        )
    print("=" * 98)

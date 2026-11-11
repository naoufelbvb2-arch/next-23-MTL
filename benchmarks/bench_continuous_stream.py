"""
benchmarks/bench_continuous_stream.py
Phase 2 of Engineering Maturity Roadmap:
Continuous Token Streams & Full Autoregressive Rollout Benchmark.

Tests the unified ContinuousSpringNetwork (SparseCausalAttention + ALiBi Positional
Context + 4 DynamicSpringLayers with Dormant Reserve) on 8 connected Arabic narrative/scientific
paragraphs (17 words each = 136 continuous tokens, 128 next-token transitions) with:
  1. Zero-Autograd AST Verification across all project files.
  2. Strict Causal Masking Verification (Future Leakage = 0.0).
  3. Step-wise Causal Local Learning & Dynamic Neurogenesis Growth Log.
  4. Teacher-Forced Next-Token Accuracy after Production Freeze (W0 only).
  5. Free-Running Autoregressive Rollout from a 2-word Seed Prefix for all 8 paragraphs.
  6. Hub-Word Disambiguation Audit ("في", "من", "ثم", "على", "إلى", "حيث").
"""

import sys
import os
import ast
import pathlib
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_continuous import ContinuousSpringNetwork
from benchmarks.corpus_data import CONTINUOUS_PARAGRAPHS_CORPUS


def verify_all_files_zero_autograd() -> tuple[bool, list[str]]:
    root = pathlib.Path(__file__).resolve().parent.parent
    files = [
        root / "engine" / "spring_layer.py",
        root / "engine" / "dynamic_spring_layer.py",
        root / "engine" / "dfa_router.py",
        root / "engine" / "causal_context.py",
        root / "engine" / "sparse_attention.py",
        root / "model.py",
        root / "model_v2.py",
        root / "model_dynamic.py",
        root / "model_continuous.py",
        pathlib.Path(__file__).resolve(),
    ]
    for fpath in files:
        content = fpath.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("backward", "grad"):
                    return False, [str(fpath)]
    return True, [f.name for f in files]


@torch.no_grad()
def verify_stream_causal_invariance(model: ContinuousSpringNetwork, paragraphs_ids: list[list[int]]) -> dict:
    """
    Verifies mathematically that:
    1. Attention weight matrix A_ij is strictly 0.0 for all j > i (upper triangle).
    2. Modifying future tokens at t > k has ZERO effect on logits at position k.
    """
    sample_batch = torch.tensor(paragraphs_ids, dtype=torch.long)  # [8, 17]
    logits_orig = model(sample_batch)  # [8, 17, V]

    attn_w = model.sparse_attention._last_attn_weights  # [8, 17, 17]
    seq_len = attn_w.shape[-1]
    upper_tri = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    future_leakage = float(attn_w[:, upper_tri].abs().max().item())

    # Perturb tokens from index 8..16 and check that logits at 0..7 are 100% invariant
    perturbed = sample_batch.clone()
    perturbed[:, 8:] = (perturbed[:, 8:] + 7) % model.vocab_size
    logits_pert = model(perturbed)
    prefix_diff = float((logits_orig[:, :8, :] - logits_pert[:, :8, :]).abs().max().item())

    return {
        "future_leakage": future_leakage,
        "prefix_diff": prefix_diff,
        "passed": (future_leakage == 0.0) and (prefix_diff < 1e-6),
    }


class ContinuousStreamBenchmark:
    def __init__(self, paragraphs: list[str], d_initial: int = 32, d_max: int = 128, delta_d: int = 8):
        self.paragraphs = paragraphs
        self.d_initial = d_initial
        self.d_max = d_max
        self.delta_d = delta_d

        self.tokenized_paragraphs = [p.strip().split() for p in paragraphs]
        tokens = sorted(list(set(w for p in self.tokenized_paragraphs for w in p)))
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for i, t in enumerate(tokens)}
        self.vocab_size = len(tokens)

        self.paragraphs_ids = [
            [self.token_to_id[w] for w in words] for words in self.tokenized_paragraphs
        ]
        self.total_tokens = sum(len(p) for p in self.paragraphs_ids)
        # Step-wise transitions from t=1..T-1 (starting from 1-word or 2-word prefix)
        self.total_transitions = sum(len(p) - 1 for p in self.paragraphs_ids)

    @torch.no_grad()
    def evaluate_teacher_forced_and_rollout(
        self, model: ContinuousSpringNetwork
    ) -> tuple[float, int, float, int, list[dict], list[dict]]:
        """
        Evaluates:
        1. Teacher-Forced Next-Token Accuracy across all (w_1..w_t) -> w_{t+1} transitions.
        2. Free-Running Autoregressive Rollout starting from ONLY the first 2 words (Seed Prefix).
        3. Hub-Word Disambiguation across repeated prepositions/conjunctions ("في", "من", "ثم", "حيث").
        """
        tf_correct = 0
        tf_total = 0
        hub_words = {"في", "من", "ثم", "على", "إلى", "حيث"}
        hub_audit_records = []

        for p_idx, seq_ids in enumerate(self.paragraphs_ids):
            # Step-wise evaluation for t = 1 .. len(seq_ids)-1
            for t in range(1, len(seq_ids)):
                prefix = seq_ids[:t]
                target_id = seq_ids[t]
                logits = model.forward(prefix)
                pred_id = int(torch.argmax(logits).item())
                is_ok = pred_id == target_id
                if is_ok:
                    tf_correct += 1
                tf_total += 1

                last_word = self.id_to_token[prefix[-1]]
                if last_word in hub_words:
                    context_str = " ".join(self.id_to_token[i] for i in prefix[max(0, t - 3) : t])
                    hub_audit_records.append({
                        "paragraph": p_idx + 1,
                        "position": t,
                        "hub_word": last_word,
                        "recent_context": context_str,
                        "target_next": self.id_to_token[target_id],
                        "pred_next": self.id_to_token[pred_id],
                        "match": is_ok,
                    })

        tf_acc = (tf_correct / float(tf_total)) * 100.0

        # Free-Running Autoregressive Rollout from 2-word Seed Prefix
        rollout_correct = 0
        rollout_details = []
        for p_idx, seq_ids in enumerate(self.paragraphs_ids):
            seed_ids = seq_ids[:2]
            gen_ids = model.autoregressive_rollout(seed_prefix_ids=seed_ids, total_length=len(seq_ids))
            is_exact = gen_ids == seq_ids
            if is_exact:
                rollout_correct += 1

            matched_words = sum(1 for a, b in zip(gen_ids, seq_ids) if a == b)
            rollout_details.append({
                "paragraph_idx": p_idx + 1,
                "seed_prefix": " ".join(self.id_to_token[i] for i in seed_ids),
                "target_text": " ".join(self.id_to_token[i] for i in seq_ids),
                "generated_text": " ".join(self.id_to_token[i] for i in gen_ids),
                "word_matches": f"{matched_words}/{len(seq_ids)}",
                "exact_match": is_exact,
            })

        rollout_acc = (rollout_correct / float(len(self.paragraphs_ids))) * 100.0
        return tf_acc, tf_correct, rollout_acc, rollout_correct, rollout_details, hub_audit_records

    @torch.no_grad()
    def run_benchmark(self, max_epochs: int = 80):
        print("=" * 92)
        print("PHASE 2 MATURITY BENCHMARK: CONTINUOUS TOKEN STREAMS & AUTOREGRESSIVE ROLLOUT")
        print("=" * 92)
        print(
            f"Paragraphs: {len(self.paragraphs)} | Length per Paragraph: {len(self.paragraphs_ids[0])} words | "
            f"Total Stream Tokens: {self.total_tokens} | Causal Transitions: {self.total_transitions} | "
            f"Vocab Size: {self.vocab_size}"
        )
        print("-" * 92)

        model = ContinuousSpringNetwork(
            vocab_size=self.vocab_size,
            d_max=self.d_max,
            d_initial=self.d_initial,
            delta_d=self.delta_d,
            num_layers=4,
            top_k=4,
            attn_top_k=5,
            max_seq_len=32,
            eta_w=0.055,
            eta_0=0.018,
            gamma=0.10,
            beta_k=0.02,
            enable_neurogenesis=True,
            saturation_threshold=0.85,
            stiffness_threshold=0.0022,
            error_threshold=0.005,
            cooldown_epochs=3,
            seed=42,
        )

        # 1. Verify Zero-Autograd AST & Strict Causal Masking
        ast_ok, checked_files = verify_all_files_zero_autograd()
        causal_check = verify_stream_causal_invariance(model, self.paragraphs_ids)
        print(f"[Check 1] AST Zero-Autograd Verification : {'PASS' if ast_ok else 'FAIL'} ({len(checked_files)} files)")
        print(
            f"[Check 2] Stream Causal Mask Invariance  : {'PASS' if causal_check['passed'] else 'FAIL'} "
            f"(Future Leakage={causal_check['future_leakage']:.1e}, Prefix Diff={causal_check['prefix_diff']:.1e})"
        )
        print("-" * 92)

        # 2. Step-wise Causal Local Learning Loop across Continuous Token Streams
        converged_epoch = -1
        for epoch in range(1, max_epochs + 1):
            epoch_loss = 0.0
            step_correct = 0

            for seq_ids in self.paragraphs_ids:
                # Step-wise causal training: at each position t, read prefix (w_1..w_t) and predict w_{t+1}
                for t in range(1, len(seq_ids)):
                    prefix = seq_ids[:t]
                    target_id = seq_ids[t]

                    logits = model.forward_step_prefix(prefix)
                    probs = F.softmax(logits, dim=-1)

                    target_t = torch.tensor(target_id, dtype=torch.long)
                    y_onehot = F.one_hot(target_t, num_classes=self.vocab_size).float()
                    e_t = y_onehot - probs

                    loss = 0.5 * float(torch.sum(e_t ** 2).item())
                    epoch_loss += loss

                    if int(torch.argmax(logits).item()) == target_id:
                        step_correct += 1

                    model.local_step_update(e_t)

            avg_loss = epoch_loss / float(self.total_transitions)
            step_acc = (step_correct / float(self.total_transitions)) * 100.0
            token_err_rate = 1.0 - (step_correct / float(self.total_transitions))

            # Check Dynamic Neurogenesis triggers at end of epoch
            events = model.step_neurogenesis(
                epoch=epoch,
                mean_epoch_error=avg_loss,
                token_error_rate=token_err_rate,
            )
            for ev in events:
                print(
                    f"  [NEUROGENESIS WAKE-UP | Epoch {ev['epoch']:03d}] Layer {ev['layer_id']}: "
                    f"D_active {ev['old_d']} -> {ev['new_d']} (+{ev['recruited']} cells) | "
                    f"Reason: {ev['reason']} (Sat={ev['saturation_before']:.1f}%, k={ev['winner_k']:.4f})"
                )

            if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
                dims = model.get_active_dimensions()
                print(
                    f"  Epoch {epoch:03d}/{max_epochs:03d} | "
                    f"Step Next-Token Acc: {step_acc:6.2f}% ({step_correct}/{self.total_transitions}) | "
                    f"MSE Loss: {avg_loss:.6f} | Active Dims: {dims}"
                )

            if step_acc == 100.0 and avg_loss < 2e-4 and converged_epoch == -1:
                converged_epoch = epoch
                dims = model.get_active_dimensions()
                print(
                    f"  [STREAM CONVERGED at Epoch {epoch:03d}] Next-Token Acc: 100.00% | "
                    f"MSE Loss: {avg_loss:.6f} | Active Dims: {dims}"
                )
                break

        # 3. Production Freeze & Dormant Pruning (W_final = W_0, w=None, k=None)
        freeze_stats = model.production_freeze(prune_dormant=True)

        # 4. Evaluate Teacher-Forced Next-Token Accuracy, Free-Running Rollout, and Hub-Word Disambiguation
        (
            tf_acc,
            tf_correct,
            rollout_acc,
            rollout_correct,
            rollout_details,
            hub_records,
        ) = self.evaluate_teacher_forced_and_rollout(model)

        hub_correct = sum(1 for r in hub_records if r["match"])
        hub_acc = (hub_correct / float(len(hub_records))) * 100.0

        print("\n" + "=" * 92)
        print("PRODUCTION FREEZE & PRUNING SUMMARY (W_0 ONLY, w=None, k=None)")
        print("=" * 92)
        print(f"  - Initial Active Dimensions        : [{self.d_initial}, {self.d_initial}, {self.d_initial}, {self.d_initial}]")
        print(f"  - Final Awakened Active Dimensions : {freeze_stats['active_dims']} (Max Latent = [128, 128, 128, 128])")
        print(
            f"  - Dormant Cells Saved & Pruned     : {freeze_stats['dormant_cells_saved']} / {freeze_stats['total_max_cells']} "
            f"({freeze_stats['dormant_saved_pct']:.1f}% capacity saved)"
        )
        print(f"  - Frozen Next-Token Accuracy (W0)  : {tf_acc:.2f}% ({tf_correct}/{self.total_transitions} transitions)")
        print(f"  - Hub-Word Disambiguation Accuracy : {hub_acc:.2f}% ({hub_correct}/{len(hub_records)} hub occurrences)")
        print(f"  - Free-Running Rollout Exact Match : {rollout_acc:.2f}% ({rollout_correct}/{len(self.paragraphs)} full paragraphs)")

        print("\n" + "=" * 92)
        print("FREE-RUNNING AUTOREGRESSIVE ROLLOUT AUDIT (FROM 2-WORD SEED PREFIX -> 17 WORDS)")
        print("=" * 92)
        for d in rollout_details:
            status = "EXACT MATCH" if d["exact_match"] else "MISMATCH"
            print(f"[{status}] Paragraph #{d['paragraph_idx']:02d} ({d['word_matches']} words) | Seed: \"{d['seed_prefix']}\"")
            print(f"    Target   : {d['target_text']}")
            print(f"    Generated: {d['generated_text']}")

        print("\n" + "=" * 92)
        print("HUB-TOKEN DISAMBIGUATION AUDIT (REPEATED PREPOSITIONS & CONJUNCTIONS ACROSS CONTEXTS)")
        print("=" * 92)
        print(f"{'Para#':<6} | {'Pos':<4} | {'Hub Token':<10} | {'Causal Prefix Context':<28} | {'Target Next':<14} | {'Pred Next':<14} | {'Status'}")
        print("-" * 92)
        for r in hub_records[:24]:
            st = "PASS" if r["match"] else "FAIL"
            print(
                f"#{r['paragraph']:02d}    | {r['position']:<4} | {r['hub_word']:<10} | "
                f"{r['recent_context']:<28} | {r['target_next']:<14} | {r['pred_next']:<14} | {st}"
            )
        print("=" * 92)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    bench = ContinuousStreamBenchmark(
        paragraphs=CONTINUOUS_PARAGRAPHS_CORPUS,
        d_initial=32,
        d_max=128,
        delta_d=8,
    )
    bench.run_benchmark(max_epochs=80)

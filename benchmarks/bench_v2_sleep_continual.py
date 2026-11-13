"""
benchmarks/bench_v2_sleep_continual.py
Milestone 2.2 Benchmark: 4-Phase Zero-Replay Continual Learning with Dual-Stage Offline
Sleep Consolidation, Synaptic Recycling, Cubic Armor Locking, and Transition Refractory Gate.

Compares two continual learning configurations across 4 sequential English narrative phases
(Phase 1 -> Phase 2 -> Phase 3 -> Phase 4, Zero-Replay):
  - Setting A (Naive Continual — No Sleep / Unshielded):
    Sequential training across the 4 phases without sleep consolidation, synaptic recycling,
    or mechanical armor locking.
  - Setting B (SS-ENN v2.2 — Dual-Stage Sleep Consolidation, Recycling & Armor Lock):
    Sequential training where each phase is followed by `sleep_consolidate_and_recycle()`:
      1. Signal-Preserving High-Cosine Synaptic Fusion
      2. Low-Utility Awakening Pruning & In-Place Dormant Rejuvenation
      3. Sleep Armor Locking (k in [0.90, 0.96]) & Cubic Anchor Shield (1 - k)^3
      4. Two-Tier Transition Refractory Gate during 30-token free-running generation.

Outputs 3 comprehensive telemetry tables:
  - Table 1: Phase-by-Phase Neurogenesis, Sleep Consolidation & Synaptic Recycling Log
  - Table 2: 4-Phase Zero-Replay Retention & Recall Matrix (Setting A vs. Setting B on W0)
  - Table 3: 30-Token Rollout, Cross-Sentence Entity Consistency & 0% Adjacent/Loop Audit

Strictly zero autograd / zero backward().
"""

import sys
import ast
import math
import time
import pathlib
import torch
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import EnglishSubwordTokenizer
from models import ScaledSpringNetwork
from benchmarks.english_dataset import (
    OPENINGS,
    CHARACTERS,
    MIDDLES,
    CLOSINGS,
    CANONICAL_SEED_PROMPTS,
)
from benchmarks.bench_scale_english_cpu import (
    verify_all_repo_files_zero_autograd,
    build_minibatch_tensors,
    evaluate_batches,
    extract_story_entities,
)


def build_4phase_continual_english_dataset(stories_per_phase: int = 96) -> dict:
    """
    Partitions the English narrative universe into 4 sequential, disjoint domain phases
    for Zero-Replay Continual Learning:
      - Phase 1 (Valley & Village Domain): Openings {0, 2}, Characters {0, 1, 2}
        ('the little bird', 'the young rabbit', 'the clever fox')
      - Phase 2 (River & Lake Domain): Openings {1, 4}, Characters {3, 4}
        ('the brave puppy', 'the gentle deer')
      - Phase 3 (Forest & Meadow Domain): Openings {3, 5}, Characters {5, 6, 7}
        ('the busy squirrel', 'the kind girl', 'the curious boy')
      - Phase 4 (Cottage & Mountain Domain): Openings {6, 7}, Characters {8, 9}
        ('the wise owl', 'the playful kitten')
    """
    phase_specs = [
        {"phase_id": 1, "name": "Phase 1 (Valley & Village)", "openings": [0, 2], "chars": [3, 0]},
        {"phase_id": 2, "name": "Phase 2 (River & Lake)", "openings": [1, 4], "chars": [1, 4]},
        {"phase_id": 3, "name": "Phase 3 (Forest & Meadow)", "openings": [3, 5], "chars": [5, 2]},
        {"phase_id": 4, "name": "Phase 4 (Cottage & Mountain)", "openings": [6, 7], "chars": [8, 9]},
    ]

    phases_data = {}
    all_stories = []

    for spec in phase_specs:
        pid = spec["phase_id"]
        unique_stories = []
        for idx_pair in range(len(spec["openings"])):
            o_idx = spec["openings"][idx_pair]
            c_idx = spec["chars"][idx_pair]
            opening = OPENINGS[o_idx]
            subj, act1, act2, act3 = CHARACTERS[c_idx]
            for m_idx, middle in enumerate(MIDDLES):
                cl_idx = (o_idx + c_idx + m_idx) % len(CLOSINGS)
                closing = CLOSINGS[cl_idx]
                if (idx_pair + m_idx) % 2 == 0:
                    story = (
                        f"{opening}, {subj} {act1}. "
                        f"Then {subj} {act2} and {act3}. "
                        f"{middle} {closing}"
                    )
                else:
                    story = (
                        f"{subj.capitalize()} {act1} {opening.lower()}. "
                        f"Afterward {subj} {act2} and {act3}. "
                        f"{middle} {closing}"
                    )
                unique_stories.append(story)

        # Sample/repeat to balanced phase size
        phase_stories = [unique_stories[i % len(unique_stories)] for i in range(stories_per_phase)]
        phases_data[pid] = {
            "name": spec["name"],
            "stories": phase_stories,
            "unique_stories": unique_stories,
        }
        all_stories.extend(phase_stories)

    return {
        "phases": phases_data,
        "all_stories": all_stories,
    }


def audit_rollout_quality_and_consistency(
    model: ScaledSpringNetwork,
    tokenizer: EnglishSubwordTokenizer,
    prompts: list[str],
    max_new_tokens: int = 30,
) -> dict:
    """
    Table 3 Audit: Generates +30 tokens from each canonical prompt using Frozen W0 only
    and measures:
      1. Cross-Sentence Entity Consistency Rate (Sentence 1 entity == Sentence 2 entity)
      2. Adjacent Duplicate Rate (x_t == x_{t-1}, e.g. 'the the')
      3. Phrase Loop Rate (any 4-gram repeating within the 30-token rollout)
    """
    records = []
    consistent_count = 0
    total_adjacent_dups = 0
    total_transitions = 0
    looped_samples = 0

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
            enable_refractory_gate=True,
        )
        gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)

        # 1. Entity consistency across sentence bridge
        entities = extract_story_entities(gen_text)
        sent1_entity = entities[0] if len(entities) >= 1 else "N/A"
        sent2_entity = entities[1] if len(entities) >= 2 else sent1_entity

        lower_text = gen_text.lower()
        if "afterward" in lower_text:
            bridge = ". afterward"
        elif "then" in lower_text:
            bridge = ". then"
        else:
            bridge = ". (direct)"

        is_consistent = (sent1_entity != "N/A") and (sent1_entity == sent2_entity)
        if is_consistent:
            consistent_count += 1

        # 2. Adjacent duplicate token audit (x_t == x_{t-1})
        content_ids = [tid for tid in out_ids if tid not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)]
        adj_dups = 0
        for i in range(1, len(content_ids)):
            total_transitions += 1
            if content_ids[i] == content_ids[i - 1]:
                adj_dups += 1
                total_adjacent_dups += 1

        # 3. 4-gram phrase loop audit
        fourgrams = {}
        has_loop = False
        for i in range(len(content_ids) - 3):
            fg = tuple(content_ids[i : i + 4])
            fourgrams[fg] = fourgrams.get(fg, 0) + 1
            if fourgrams[fg] >= 2:
                has_loop = True
        if has_loop:
            looped_samples += 1

        records.append({
            "prompt": prompt,
            "generated": gen_text,
            "new_tokens": len(out_ids) - len(prompt_ids),
            "sent1_entity": sent1_entity,
            "bridge": bridge,
            "sent2_entity": sent2_entity,
            "consistent": is_consistent,
            "adj_dups": adj_dups,
            "has_loop": has_loop,
        })

    n_prompts = max(len(prompts), 1)
    return {
        "records": records,
        "entity_consistency_pct": (consistent_count / n_prompts) * 100.0,
        "adjacent_dup_rate_pct": (total_adjacent_dups / max(total_transitions, 1)) * 100.0,
        "phrase_loop_rate_pct": (looped_samples / n_prompts) * 100.0,
        "consistent_count": consistent_count,
        "total_adjacent_dups": total_adjacent_dups,
        "looped_samples": looped_samples,
    }


@torch.no_grad()
def run_v2_sleep_continual_benchmark(
    preset: str = "1.5M",
    epochs_per_phase: int = 8,
    batch_size: int = 32,
    seq_len: int = 36,
):
    torch.set_grad_enabled(False)

    print("=" * 106)
    print("SS-ENN v2.0 MILESTONE 2.2 BENCHMARK: 4-PHASE CONTINUAL LEARNING, DUAL-STAGE SLEEP & SYNAPTIC RECYCLING")
    print("=" * 106)

    ast_ok, n_files = verify_all_repo_files_zero_autograd()
    print(f"[Gate Check] AST Zero-Autograd Verification : {'PASS' if ast_ok else 'FAIL'} ({n_files} Python files verified)")

    dataset = build_4phase_continual_english_dataset(stories_per_phase=96)
    phases = dataset["phases"]
    all_stories = dataset["all_stories"]

    tokenizer = EnglishSubwordTokenizer(max_vocab_size=2048)
    tokenizer.fit(all_stories, pad_to_max_vocab=True)

    phase_batches = {}
    phase_seqs = {}
    phase_vocab_sets = {}
    all_seqs = []

    for pid in (1, 2, 3, 4):
        batches, seqs, n_toks = build_minibatch_tensors(
            phases[pid]["stories"], tokenizer, seq_len=seq_len, batch_size=batch_size
        )
        phase_batches[pid] = batches
        phase_seqs[pid] = seqs
        all_seqs.extend(seqs)
        vset = set()
        for s in seqs:
            for tid in s:
                if tid != tokenizer.pad_id:
                    vset.add(tid)
        phase_vocab_sets[pid] = vset

    # Shared vocabulary IDs appearing in >= 2 phases (syntax, punctuation, middles, closings)
    shared_vocab_ids = set()
    for tid in range(tokenizer.vocab_size):
        appear_count = sum(1 for pid in (1, 2, 3, 4) if tid in phase_vocab_sets[pid])
        if appear_count >= 2:
            shared_vocab_ids.add(tid)

    print(
        f"[Dataset] 4 Sequential Phases x 96 Stories = {len(all_stories)} Stories | "
        f"Vocab={tokenizer.vocab_size:,} (Shared Grammar/Discourse IDs={len(shared_vocab_ids)})"
    )
    print("-" * 106)

    # =========================================================================
    # SETTING A: NAIVE CONTINUAL (NO SLEEP / UNSHIELDED)
    # =========================================================================
    print("[Running Setting A] Naive Continual (No Sleep Consolidation / Unshielded) across Phases 1 -> 4...")
    model_naive = ScaledSpringNetwork(
        preset=preset,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len + 8,
        eta_w=0.14,
        eta_0=0.055,
        gamma=0.0,
        beta_k=0.0,
        eta_b=0.015,
        enable_neurogenesis=True,
        saturation_threshold=0.80,
        stiffness_threshold=0.08,
        error_threshold=0.005,
        cooldown_epochs=2,
        seed=42,
    )
    model_naive.use_cubic_shield = False
    model_naive.calibrate_salience_from_tokens(
        all_seqs, pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id
    )

    naive_phase_dims = {}
    for pid in (1, 2, 3, 4):
        for ep in range(1, epochs_per_phase + 1):
            ep_mse = 0.0
            ep_corr = 0
            ep_val = 0
            for inp, tgt in phase_batches[pid]:
                st = model_naive.train_minibatch(inp, tgt, pad_id=tokenizer.pad_id)
                ep_mse += st["mse_loss"] * st["valid_tokens"]
                ep_corr += st["correct"]
                ep_val += st["valid_tokens"]
            model_naive.step_neurogenesis(
                epoch=(pid - 1) * epochs_per_phase + ep,
                mean_epoch_error=ep_mse / max(ep_val, 1),
                token_error_rate=1.0 - (ep_corr / max(ep_val, 1)),
            )
        naive_phase_dims[pid] = model_naive.get_allocated_dimensions()

    naive_freeze_stats = model_naive.production_freeze(prune_dormant=True)
    naive_post_breakdown = model_naive.count_parameters_breakdown()
    naive_evals = {
        pid: evaluate_batches(model_naive, phase_batches[pid], pad_id=tokenizer.pad_id)
        for pid in (1, 2, 3, 4)
    }

    # =========================================================================
    # SETTING B: SS-ENN v2.2 (DUAL-STAGE SLEEP, SYNAPTIC RECYCLING & ARMOR LOCK)
    # =========================================================================
    print("[Running Setting B] SS-ENN v2.2 (Dual-Stage Sleep Consolidation, Recycling & Cubic Armor Lock)...")
    model_sleep = ScaledSpringNetwork(
        preset=preset,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len + 8,
        eta_w=0.14,
        eta_0=0.048,
        gamma=0.06,
        beta_k=0.022,
        eta_b=0.015,
        enable_neurogenesis=True,
        saturation_threshold=0.80,
        stiffness_threshold=0.08,
        error_threshold=0.005,
        cooldown_epochs=2,
        seed=42,
    )
    model_sleep.use_cubic_shield = True
    model_sleep.calibrate_salience_from_tokens(
        all_seqs, pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id
    )

    phase_telemetry = []
    t_start = time.perf_counter()

    for pid in (1, 2, 3, 4):
        prep_info = model_sleep.prepare_continual_phase(phase_seqs[pid], pad_id=tokenizer.pad_id)
        phase_rejuvenated = prep_info.get("total_rejuvenated", 0)
        for ep in range(1, epochs_per_phase + 1):
            ep_mse = 0.0
            ep_corr = 0
            ep_val = 0
            for inp, tgt in phase_batches[pid]:
                st = model_sleep.train_minibatch(inp, tgt, pad_id=tokenizer.pad_id)
                ep_mse += st["mse_loss"] * st["valid_tokens"]
                ep_corr += st["correct"]
                ep_val += st["valid_tokens"]
            ng_events = model_sleep.step_neurogenesis(
                epoch=ep,
                mean_epoch_error=ep_mse / max(ep_val, 1),
                token_error_rate=1.0 - (ep_corr / max(ep_val, 1)),
            )
            phase_rejuvenated += sum(ev.get("rejuvenated_in_place", 0) for ev in ng_events)

        sleep_info = model_sleep.sleep_consolidate_and_recycle(
            phase_id=pid,
            merge_cosine_threshold=0.72,
            prune_fraction=0.16,
            armor_stiffness=0.94,
            shared_vocab_ids=shared_vocab_ids,
            is_final_phase=(pid == 4),
        )
        phase_telemetry.append({
            "phase_id": pid,
            "name": phases[pid]["name"],
            "rejuvenated_in_place": phase_rejuvenated,
            "allocated_dims": sleep_info["allocated_dims"],
            "effective_dims": sleep_info["effective_active_dims"],
            "merged_cells": sum(sleep_info["merged_per_layer"]),
            "pruned_cells": sum(sleep_info["pruned_per_layer"]),
            "recycled_total": sleep_info["recycled_this_sleep"],
            "locked_cells": sum(sleep_info["locked_per_layer"]),
            "naive_dims": naive_phase_dims[pid],
        })
        print(
            f"  [Sleep Cycle #{pid}] {phases[pid]['name']:<28} | "
            f"Rejuvenated: {phase_rejuvenated:2d} | "
            f"Fused: {sum(sleep_info['merged_per_layer']):2d} | "
            f"Pruned: {sum(sleep_info['pruned_per_layer']):2d} | "
            f"Armor-Locked: {sum(sleep_info['locked_per_layer']):3d} | "
            f"Effective Dims: {sleep_info['effective_active_dims']}"
        )

    t_elapsed = time.perf_counter() - t_start
    sleep_freeze_stats = model_sleep.production_freeze(prune_dormant=True)
    sleep_post_breakdown = model_sleep.count_parameters_breakdown()

    sleep_evals = {
        pid: evaluate_batches(model_sleep, phase_batches[pid], pad_id=tokenizer.pad_id)
        for pid in (1, 2, 3, 4)
    }

    rollout_audit = audit_rollout_quality_and_consistency(
        model_sleep, tokenizer, CANONICAL_SEED_PROMPTS, max_new_tokens=30
    )

    # =========================================================================
    # TABLE 1: PHASE-BY-PHASE NEUROGENESIS, SLEEP & SYNAPTIC RECYCLING LOG
    # =========================================================================
    print("\n" + "=" * 106)
    print("TABLE 1: PHASE-BY-PHASE NEUROGENESIS, SLEEP CONSOLIDATION & SYNAPTIC RECYCLING LOG")
    print("=" * 106)
    print(
        f"{'Phase':<28} | {'Rejuv In-Place':<14} | {'Fused (S1)':<10} | {'Pruned (S2)':<11} | "
        f"{'Armor Locked':<12} | {'v2.2 Effective Dims':<21} | {'Naive Dims'}"
    )
    print("-" * 106)
    for pt in phase_telemetry:
        print(
            f"{pt['name']:<28} | {pt['rejuvenated_in_place']:>10} cells | "
            f"{pt['merged_cells']:>6} cells | {pt['pruned_cells']:>7} cells | "
            f"{pt['locked_cells']:>8} cells | {str(pt['effective_dims']):<21} | {pt['naive_dims']}"
        )
    print("-" * 106)
    print(
        f"FINAL FROZEN W0 SUMMARY      | Total Recycled Across 4 Sleeps: {sleep_freeze_stats['total_recycled_across_sleeps']} cells | "
        f"Compacted Active Cells: {sleep_freeze_stats['total_active_cells']}/{sleep_freeze_stats['total_max_cells']} "
        f"({sleep_freeze_stats['dormant_saved_pct']:.1f}% saved vs {naive_freeze_stats['dormant_saved_pct']:.1f}% in Naive)"
    )
    print(
        f"FROZEN W0 MEMORY FOOTPRINT   | SS-ENN v2.2 (With Sleep): {sleep_post_breakdown['frozen_w0_params']:,} params "
        f"({sleep_post_breakdown['live_buffer_mb']:.2f} MB) vs Naive (No Sleep): "
        f"{naive_post_breakdown['frozen_w0_params']:,} params ({naive_post_breakdown['live_buffer_mb']:.2f} MB)"
    )

    # =========================================================================
    # TABLE 2: 4-PHASE ZERO-REPLAY RETENTION & RECALL MATRIX (SETTING A vs B)
    # =========================================================================
    print("\n" + "=" * 106)
    print("TABLE 2: 4-PHASE ZERO-REPLAY RETENTION & RECALL MATRIX AFTER PHASE 4 (FROZEN W0 ONLY)")
    print("=" * 106)
    print(
        f"{'Continual Phase Evaluated':<30} | {'Setting A: Naive (No Sleep) PPL / Acc':<36} | "
        f"{'Setting B: SS-ENN v2.2 (Sleep+Armor) PPL / Acc':<44}"
    )
    print("-" * 106)
    naive_acc_sum = 0.0
    sleep_acc_sum = 0.0
    naive_ppl_sum = 0.0
    sleep_ppl_sum = 0.0

    for pid in (1, 2, 3, 4):
        na = naive_evals[pid]
        sl = sleep_evals[pid]
        naive_acc_sum += na["acc"]
        sleep_acc_sum += sl["acc"]
        naive_ppl_sum += na["ppl"]
        sleep_ppl_sum += sl["ppl"]
        print(
            f"{phases[pid]['name']:<30} | PPL: {na['ppl']:6.2f}  |  Acc: {na['acc']:6.2f}%          | "
            f"PPL: {sl['ppl']:6.2f}  |  Acc: {sl['acc']:6.2f}%  (+{sl['acc'] - na['acc']:.2f}%)"
        )

    print("-" * 106)
    mean_naive_acc = naive_acc_sum / 4.0
    mean_sleep_acc = sleep_acc_sum / 4.0
    mean_naive_ppl = naive_ppl_sum / 4.0
    mean_sleep_ppl = sleep_ppl_sum / 4.0
    print(
        f"{'4-PHASE OVERALL MEAN RETENTION':<30} | PPL: {mean_naive_ppl:6.2f}  |  Acc: {mean_naive_acc:6.2f}%          | "
        f"PPL: {mean_sleep_ppl:6.2f}  |  Acc: {mean_sleep_acc:6.2f}%  (+{mean_sleep_acc - mean_naive_acc:.2f}%)"
    )

    # =========================================================================
    # TABLE 3: 30-TOKEN ROLLOUT, ENTITY CONSISTENCY & 0% ADJACENT/LOOP AUDIT
    # =========================================================================
    print("\n" + "=" * 106)
    print("TABLE 3: 30-TOKEN FREE ROLLOUT, ENTITY CONSISTENCY & REFRACTORY GATE AUDIT (FROZEN W0 ONLY)")
    print("=" * 106)
    for idx, r in enumerate(rollout_audit["records"], 1):
        ent_status = "PASS (NO DRIFT)" if r["consistent"] else "DRIFT"
        dup_status = "0 ADJ DUPS" if r["adj_dups"] == 0 else f"{r['adj_dups']} DUPS"
        loop_status = "0 LOOPS" if not r["has_loop"] else "LOOP"
        print(
            f"[Sample #{idx:02d}] Prompt: \"{r['prompt']}\" | "
            f"Entities: {r['sent1_entity']} -> ({r['bridge']}) -> {r['sent2_entity']} [{ent_status}] | "
            f"[{dup_status}] | [{loop_status}]"
        )
        print(f"             Rollout: {r['generated']}")
        print("-" * 106)

    print(
        f"SUMMARY AUDIT | Cross-Sentence Entity Consistency : {rollout_audit['entity_consistency_pct']:.2f}% "
        f"({rollout_audit['consistent_count']}/{len(rollout_audit['records'])}) | "
        f"Adjacent Duplicate Rate ('the the') : {rollout_audit['adjacent_dup_rate_pct']:.2f}% | "
        f"Phrase Loop Rate : {rollout_audit['phrase_loop_rate_pct']:.2f}%"
    )
    print("=" * 106)


if __name__ == "__main__":
    run_v2_sleep_continual_benchmark(preset="1.5M", epochs_per_phase=8, batch_size=32, seq_len=36)

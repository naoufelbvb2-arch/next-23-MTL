"""
benchmarks/bench_v2_compositional_gen.py
SS-ENN v2.0 (Milestone 2.3): Zero-Shot Compositional Generalization Benchmark
(Hierarchical Stiffness Spectrum, Parameter-Free Local Active RMSNorm & Role-Factored Decoupling)

Protocol:
  1. Build a strict Compositional Split over 8 distinct Narrative Triplets (L_i, S_i, A_i):
       - Train Split: Strictly DIAGONAL pairing (L_i, S_i, A_i) across 4 shared Bridges & Closings.
         Each location L_i ONLY ever co-occurs with its owner subject S_i and action A_i during training.
       - Zero-Shot Cross-Recombination Test Split: 24 UNSEEN hybrid stories (L_j, S_i, A_k)
         with i != j and k != i and k != j (0% compositional overlap with training).
  2. Train TWO models identically under Zero-Autograd local spring updates:
       - Baseline Flat (`enable_hierarchical_compositional=False`, v2.1/v2.2 settings)
       - SS-ENN v2.3    (`enable_hierarchical_compositional=True`, Laws 1-3 active)
  3. Freeze BOTH models (`production_freeze()`, W_0 ONLY) and evaluate on Frozen W_0 Only:
       - Table 1: Train Recall, Zero-Shot Full-Sequence Hybrid Acc & PPL,
         Compositional Continuation Acc, Counter-Template Subject Binding Acc (S_i in foreign L_j).
       - Table 2: Layer-wise Hierarchical Top-K, Mean/Max Stiffness k_l, and RMS Energy
         (Pre-Norm vs Post-Active-RMSNorm on Train vs Unseen Hybrid inputs).
       - Table 3: +30-Token Free Rollout Audit on 6 Hybrid Crossed Prompts
         (100% Sentence-2 S_i retention without drifting to L_j's training owner,
         0.0% adjacent duplicate tokens, 0.0% 3-gram phrase loops).
"""
import math
import pathlib
import sys
import time
import torch
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import EnglishSubwordTokenizer
from models import ScaledSpringNetwork
from benchmarks.bench_scale_english_cpu import (
    verify_all_repo_files_zero_autograd,
    build_minibatch_tensors,
    evaluate_batches,
)


LOCATIONS = [
    "under the tall pine tree beside the quiet northern creek where silver ferns softly sway",
    "near the calm blue lake along the wide sandy shore where clear ripples gently gleam",
    "across the warm golden meadow behind the old stone barn where soft breezes lightly blow",
    "inside the cozy wooden cabin upon the snowy mountain ridge where warm lanterns softly glow",
    "around the bright flower garden past the painted red gate where sweet blossoms slowly open",
    "beneath the pale evening moon near the whispering bamboo grove where cool shadows quietly rest",
    "above the deep crystal canyon over the narrow rope bridge where distant echoes softly drift",
    "among the ancient mossy boulders inside the misty green hollow where cool springs quietly flow",
]

SUBJECTS = [
    "little fox",
    "brave puppy",
    "tiny squirrel",
    "gentle rabbit",
    "happy duck",
    "clever owl",
    "young deer",
    "small turtle",
]

ACTIONS = [
    "chased a fluttering yellow butterfly past the blooming wildflowers and tall grass softly .",
    "splashed in a sparkling shallow puddle near the smooth stones with pure delight .",
    "carried a smooth rounded pebble along the winding dirt trail toward the hearth .",
    "climbed up a rough cedar branch high above the mossy bark toward the lookout .",
    "followed a floating white feather swirling gently through the crisp morning autumn breeze .",
    "picked a sweet purple berry hanging low from the tangled thorny hillside bush .",
    "rolled a shiny spotted acorn cleanly down the steep grassy woodland slope today .",
    "watched a tiny twinkling firefly glow brightly across the quiet twilight sky above .",
]

BRIDGE_CLOSINGS = [
    ("afterward , the", "found a cozy sunny spot beneath the leaves and rested peacefully tonight ."),
    ("then the", "smiled with quiet pride along the trail and walked home safely tonight ."),
    ("soon , the", "shared a warm joyful wave with friends and slept soundly tonight ."),
    ("happily , the", "felt quite calm and thankful under the bright starry sky tonight ."),
]


def build_compositional_dataset() -> tuple[list[str], list[str], list[dict]]:
    """
    Constructs:
      - `train_stories`: Strictly paired diagonal combinations (L_i, S_i, A_i) x 4 Bridge-Closings (x2 copies = 64 stories)
      - `recomb_stories`: 24 UNSEEN cross-recombinations (L_j, S_i, A_k) with j != i, k != i, k != j
      - `recomb_meta`: Metadata tracking exact owner S_j vs injected hybrid subject S_i
    """
    base_train: list[str] = []
    for i in range(len(SUBJECTS)):
        for bridge, closing in BRIDGE_CLOSINGS:
            story = f"{LOCATIONS[i]} , a {SUBJECTS[i]} {ACTIONS[i]} {bridge} {SUBJECTS[i]} {closing}"
            base_train.append(story)

    # Duplicate deterministically for balanced minibatch sizing (64 stories total)
    train_stories = base_train * 2

    recomb_stories: list[str] = []
    recomb_meta: list[dict] = []
    n_world = len(SUBJECTS)
    for i in range(n_world):
        for offset_j, offset_k in [(1, 2), (3, 5), (4, 6)]:
            j = (i + offset_j) % n_world
            k = (i + offset_k) % n_world
            m = (i + offset_j) % len(BRIDGE_CLOSINGS)
            bridge, closing = BRIDGE_CLOSINGS[m]
            story = f"{LOCATIONS[j]} , a {SUBJECTS[i]} {ACTIONS[k]} {bridge} {SUBJECTS[i]} {closing}"
            recomb_stories.append(story)
            recomb_meta.append(
                {
                    "story": story,
                    "hybrid_subject": SUBJECTS[i],
                    "template_owner": SUBJECTS[j],
                    "action_owner": SUBJECTS[k],
                    "location_idx": j,
                    "subject_idx": i,
                    "action_idx": k,
                    "bridge": bridge,
                }
            )

    return train_stories, recomb_stories, recomb_meta


@torch.no_grad()
def evaluate_compositional_metrics(
    model: ScaledSpringNetwork,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    meta_list: list[dict],
    tokenizer: EnglishSubwordTokenizer,
    pad_id: int = 0,
) -> dict[str, float]:
    """
    Evaluates a Frozen (W_0) model on Unseen Cross-Recombination sequences, computing:
      1. `full_acc` / `ppl`: Raw unmasked next-token accuracy and perplexity across all valid tokens.
      2. `cont_acc`: Continuation accuracy across all causally conditioned tokens
         (L_j[1:], ', a', S_i[1], A_k[1:], Bridge[1:], S_i Sentence-2, Closing, <eos>).
      3. `binding_acc` & `owner_revert_rate`: Exact Counter-Template Subject Binding accuracy
         on Sentence-2 `S_i` recall (after bridge `"the -> S_i[0]"` and `"S_i[0] -> S_i[1]"`),
         plus how often predictions erroneously reverted to `L_j`'s original owner `S_j`.
    """
    raw_stats = evaluate_batches(model, batches, pad_id=pad_id)

    cont_correct = 0
    cont_total = 0
    bind_correct = 0
    bind_total = 0
    owner_reverts = 0

    meta_idx = 0
    for inp, tgt in batches:
        logits = model(inp)  # [B, T, V]
        preds = logits.argmax(dim=-1)
        bsz, seq_len = inp.shape

        for b in range(bsz):
            meta = meta_list[meta_idx]
            meta_idx += 1

            subj_tokens = tokenizer.encode(meta["hybrid_subject"], add_bos=False, add_eos=False)
            owner_tokens = tokenizer.encode(meta["template_owner"], add_bos=False, add_eos=False)
            loc_tokens = tokenizer.encode(LOCATIONS[meta["location_idx"]], add_bos=False, add_eos=False)
            act_tokens = tokenizer.encode(ACTIONS[meta["action_idx"]], add_bos=False, add_eos=False)
            bridge_tokens = tokenizer.encode(meta["bridge"], add_bos=False, add_eos=False)

            # Locate exact offsets inside tgt[b] (since inp = [<bos>, x_0, x_1, ...], tgt = [x_0, x_1, ...]):
            # 0..len(loc)-1 -> L_j
            # +0..1         -> ", a"
            # +2..3         -> S_i (Sentence 1)
            # +4..4+len(act)-1 -> A_k
            # +len(bridge)  -> Bridge_m
            # +2            -> S_i (Sentence 2 - Counter-Template Subject Binding target!)
            # rest..eos     -> Closing + <eos>
            p_loc_start = 0
            p_s1_start = len(loc_tokens) + 2
            p_act_start = p_s1_start + len(subj_tokens)
            p_bridge_start = p_act_start + len(act_tokens)
            p_s2_start = p_bridge_start + len(bridge_tokens)

            unconditioned_positions = {p_loc_start, p_s1_start, p_act_start, p_bridge_start}
            s2_positions = {p_s2_start + s_off for s_off in range(len(subj_tokens))}

            for t in range(seq_len):
                y = int(tgt[b, t].item())
                if y == pad_id:
                    continue
                p_hat = int(preds[b, t].item())

                if t not in unconditioned_positions:
                    cont_total += 1
                    if p_hat == y:
                        cont_correct += 1

                if t in s2_positions:
                    bind_total += 1
                    if p_hat == y:
                        bind_correct += 1
                    s_off = t - p_s2_start
                    if s_off < len(owner_tokens) and p_hat == owner_tokens[s_off]:
                        owner_reverts += 1

    return {
        "full_acc": raw_stats["acc"],
        "ppl": raw_stats["ppl"],
        "ce_loss": raw_stats["ce_loss"],
        "cont_acc": (cont_correct / max(cont_total, 1)) * 100.0,
        "binding_acc": (bind_correct / max(bind_total, 1)) * 100.0,
        "owner_revert_rate": (owner_reverts / max(bind_total, 1)) * 100.0,
        "bind_correct": bind_correct,
        "bind_total": bind_total,
    }


def inspect_layer_k_and_rms(
    model: ScaledSpringNetwork,
    sample_batch: torch.Tensor,
) -> tuple[list[dict], list[tuple[float, float]]]:
    """
    Captures pre-freeze active stiffness profile per layer AND measures Raw vs Active-RMSNorm
    layer output energy across L1..L4 on `sample_batch` ([B, T]).
    """
    k_stats = []
    for l_idx, layer in enumerate(model.spring_layers):
        d_act = layer.d_active
        k_val = layer.k[:d_act, :] if layer.k is not None else torch.zeros(d_act, model.d_max)
        k_stats.append(
            {
                "layer": l_idx + 1,
                "top_k": model.hierarchical_top_ks[l_idx] if model.enable_hierarchical_compositional else layer.top_k,
                "mean_k": float(k_val.mean().item()),
                "max_k": float(k_val.max().item()),
            }
        )

    _ = model(sample_batch)
    rms_pairs = list(zip(model._last_layer_raw_rms, model._last_layer_normed_rms))
    return k_stats, rms_pairs


def audit_hybrid_rollout(
    model: ScaledSpringNetwork,
    tokenizer: EnglishSubwordTokenizer,
    prompt_spec: dict,
    max_new_tokens: int = 30,
) -> dict:
    """
    Executes +30-token free rollout on a hybrid crossed prompt and audits:
      - Expected Sentence-2 hybrid subject `S_i` preservation (no drift back to `L_j` owner `S_j`)
      - Adjacent duplicate token rate (must be 0.0%)
      - Repeated 3-gram phrase loop rate (must be 0.0%)
    """
    prompt_text = prompt_spec["prompt"]
    prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    gen_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        enable_refractory_gate=True,
    )
    new_ids = gen_ids[len(prompt_ids):]
    # Remove <eos> from text for clean display, inspect full generation
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    new_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Check adjacent duplicates over generated span + boundary
    adj_dups = sum(1 for idx in range(1, len(gen_ids)) if gen_ids[idx] == gen_ids[idx - 1])
    adj_dup_rate = (adj_dups / max(len(gen_ids) - 1, 1)) * 100.0

    # Check repeated non-entity 3-grams within the newly generated tokens
    trigrams: dict[tuple[int, int, int], int] = {}
    loops = 0
    for idx in range(len(new_ids) - 2):
        tg = (new_ids[idx], new_ids[idx + 1], new_ids[idx + 2])
        trigrams[tg] = trigrams.get(tg, 0) + 1
        if trigrams[tg] > 1:
            loops += 1
    loop_rate = (loops / max(len(new_ids) - 2, 1)) * 100.0

    # Locate Sentence-2 subject inside `full_text` (or directly in `new_text` when Sentence-1 prompts generate S2)
    full_lower = full_text.lower()
    found_subjects = []
    for subj in SUBJECTS:
        start = 0
        while True:
            pos = full_lower.find(subj, start)
            if pos == -1:
                break
            found_subjects.append((pos, subj))
            start = pos + len(subj)
    found_subjects.sort(key=lambda x: x[0])
    subj_seq = [s for _, s in found_subjects]

    # Must mention S_i in Sentence 1 and retain S_i in Sentence 2 without reverting to S_j
    expected_s = prompt_spec["hybrid_subject"]
    template_owner = prompt_spec["template_owner"]
    s2_ok = (len(subj_seq) >= 2) and (subj_seq[0] == expected_s) and all(s == expected_s for s in subj_seq[1:])
    drifted_to_owner = template_owner in subj_seq[1:]

    return {
        "name": prompt_spec["name"],
        "prompt": prompt_text,
        "generated_tail": new_text,
        "expected_subject": expected_s,
        "template_owner": template_owner,
        "subjects_seen": subj_seq,
        "s2_consistent": s2_ok and (not drifted_to_owner),
        "adj_dup_rate": adj_dup_rate,
        "loop_rate": loop_rate,
        "generated_len": len(new_ids),
    }


def train_model_variant(
    enable_hierarchical_compositional: bool,
    train_stories: list[str],
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    train_seqs: list[list[int]],
    tokenizer: EnglishSubwordTokenizer,
    epochs: int = 24,
    seed: int = 42,
) -> tuple[ScaledSpringNetwork, list[dict], list[tuple[float, float]], float]:
    model = ScaledSpringNetwork(
        preset="1.5M",
        vocab_size=max(256, tokenizer.vocab_size),
        max_seq_len=64,
        eta_w=0.08,
        eta_0=0.025,
        gamma=0.08,
        beta_k=0.02,
        eta_b=0.015,
        enable_neurogenesis=True,
        enable_hierarchical_compositional=enable_hierarchical_compositional,
        seed=seed,
    )
    model.calibrate_salience_from_tokens(
        train_seqs,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
    )

    t0 = time.perf_counter()
    for epoch in range(epochs):
        total_err = 0.0
        total_valid = 0
        total_correct = 0
        for inp, tgt in train_batches:
            stats = model.train_minibatch(inp, tgt, pad_id=tokenizer.pad_id)
            total_err += stats["mse_loss"] * stats["valid_tokens"]
            total_correct += stats["correct"]
            total_valid += stats["valid_tokens"]

        mean_err = total_err / max(total_valid, 1)
        token_err_rate = 1.0 - (total_correct / max(total_valid, 1))
        model.step_neurogenesis(epoch, mean_err, token_err_rate)

    train_time_s = time.perf_counter() - t0
    k_profile, _ = inspect_layer_k_and_rms(model, train_batches[0][0])
    model.production_freeze(prune_dormant=True)
    return model, k_profile, train_batches[0][0], train_time_s


def run_v2_compositional_gen_benchmark(verbose: bool = True) -> dict:
    # 0. Strict AST Zero-Autograd Verification
    zero_ok, num_scanned = verify_all_repo_files_zero_autograd()
    if not zero_ok:
        raise RuntimeError("AST Zero-Autograd check failed!")

    train_stories, recomb_stories, recomb_meta = build_compositional_dataset()
    tokenizer = EnglishSubwordTokenizer(max_vocab_size=512)
    tokenizer.fit(train_stories + recomb_stories)

    train_batches, train_seqs, _ = build_minibatch_tensors(
        train_stories, tokenizer, seq_len=62, batch_size=16
    )
    recomb_batches, _, _ = build_minibatch_tensors(
        recomb_stories, tokenizer, seq_len=62, batch_size=12
    )

    # Train Baseline Flat (enable_hierarchical_compositional=False) vs SS-ENN v2.3 (True)
    flat_model, flat_k_prof, sample_train_inp, flat_train_s = train_model_variant(
        enable_hierarchical_compositional=False,
        train_stories=train_stories,
        train_batches=train_batches,
        train_seqs=train_seqs,
        tokenizer=tokenizer,
        epochs=24,
        seed=42,
    )
    v23_model, v23_k_prof, _, v23_train_s = train_model_variant(
        enable_hierarchical_compositional=True,
        train_stories=train_stories,
        train_batches=train_batches,
        train_seqs=train_seqs,
        tokenizer=tokenizer,
        epochs=24,
        seed=42,
    )

    # Evaluate on Frozen W_0 ONLY
    flat_train_eval = evaluate_batches(flat_model, train_batches, pad_id=tokenizer.pad_id)
    v23_train_eval = evaluate_batches(v23_model, train_batches, pad_id=tokenizer.pad_id)

    flat_recomb_eval = evaluate_compositional_metrics(
        flat_model, recomb_batches, recomb_meta, tokenizer, pad_id=tokenizer.pad_id
    )
    v23_recomb_eval = evaluate_compositional_metrics(
        v23_model, recomb_batches, recomb_meta, tokenizer, pad_id=tokenizer.pad_id
    )

    # Measure Frozen W_0 RMS Energy (Train vs Unseen Hybrid inputs)
    sample_hybrid_inp = recomb_batches[0][0]
    _ = flat_model(sample_train_inp)
    flat_rms_train = list(zip(flat_model._last_layer_raw_rms, flat_model._last_layer_normed_rms))
    _ = flat_model(sample_hybrid_inp)
    flat_rms_hybrid = list(zip(flat_model._last_layer_raw_rms, flat_model._last_layer_normed_rms))

    _ = v23_model(sample_train_inp)
    v23_rms_train = list(zip(v23_model._last_layer_raw_rms, v23_model._last_layer_normed_rms))
    _ = v23_model(sample_hybrid_inp)
    v23_rms_hybrid = list(zip(v23_model._last_layer_raw_rms, v23_model._last_layer_normed_rms))

    # Build 6 Hybrid Crossed Rollout Prompts (3 Sentence-2 bridges + 3 early Sentence-1 clause completions)
    rollout_specs = [
        {
            "name": "H1 (Lake[puppy] + Fox + Canyon-Act -> Bridge)",
            "prompt": f"{LOCATIONS[1]} , a {SUBJECTS[0]} {ACTIONS[6]} afterward , the",
            "hybrid_subject": SUBJECTS[0],
            "template_owner": SUBJECTS[1],
        },
        {
            "name": "H2 (Pine[fox] + Puppy + Moon-Act -> Bridge)",
            "prompt": f"{LOCATIONS[0]} , a {SUBJECTS[1]} {ACTIONS[5]} then the",
            "hybrid_subject": SUBJECTS[1],
            "template_owner": SUBJECTS[0],
        },
        {
            "name": "H3 (Cabin[rabbit] + Squirrel + Meadow-Act -> Bridge)",
            "prompt": f"{LOCATIONS[3]} , a {SUBJECTS[2]} {ACTIONS[4]} soon , the",
            "hybrid_subject": SUBJECTS[2],
            "template_owner": SUBJECTS[3],
        },
        {
            "name": "H4 (Meadow[squirrel] + Rabbit, Early S1 Prompt)",
            "prompt": f"{LOCATIONS[2]} , a {SUBJECTS[3]}",
            "hybrid_subject": SUBJECTS[3],
            "template_owner": SUBJECTS[2],
        },
        {
            "name": "H5 (Canyon[deer] + Duck, Early S1 Prompt)",
            "prompt": f"{LOCATIONS[6]} , a {SUBJECTS[4]}",
            "hybrid_subject": SUBJECTS[4],
            "template_owner": SUBJECTS[6],
        },
        {
            "name": "H6 (Hollow[turtle] + Owl, Early S1 Prompt)",
            "prompt": f"{LOCATIONS[7]} , a {SUBJECTS[5]}",
            "hybrid_subject": SUBJECTS[5],
            "template_owner": SUBJECTS[7],
        },
    ]

    v23_rollouts = [
        audit_hybrid_rollout(v23_model, tokenizer, spec, max_new_tokens=30)
        for spec in rollout_specs
    ]
    flat_rollouts = [
        audit_hybrid_rollout(flat_model, tokenizer, spec, max_new_tokens=30)
        for spec in rollout_specs
    ]

    v23_s2_retain_pct = (sum(1 for r in v23_rollouts if r["s2_consistent"]) / len(v23_rollouts)) * 100.0
    flat_s2_retain_pct = (sum(1 for r in flat_rollouts if r["s2_consistent"]) / len(flat_rollouts)) * 100.0
    v23_mean_adj_dup = sum(r["adj_dup_rate"] for r in v23_rollouts) / len(v23_rollouts)
    v23_mean_loop = sum(r["loop_rate"] for r in v23_rollouts) / len(v23_rollouts)

    if verbose:
        print("=" * 96)
        print("SS-ENN v2.0 (Milestone 2.3): Zero-Shot Compositional Generalization Benchmark (Frozen W_0)")
        print("=" * 96)
        print(f"AST Zero-Autograd Verification : PASS ({num_scanned} Python files scanned, 0 backward/grad)")
        print(f"Compositional Split            : {len(train_stories)} Diagonal Train (L_i,S_i,A_i) | "
              f"{len(recomb_stories)} Unseen Hybrid (L_j,S_i,A_k, j!=i!=k)")
        print(f"Detected Entity Vocab Tokens   : {int(v23_model.is_entity_vocab.sum().item())} tokens "
              f"(multiword={v23_model.has_multiword_entities})")
        print()

        print("--- Table 1: Zero-Shot Cross-Recombination Accuracy & Counter-Template Subject Binding (Frozen W_0) ---")
        print(f"{'Architecture':<22} | {'Train Rec%':>10} | {'Hybrid Full%':>12} | {'Hybrid Cont%':>12} | "
              f"{'Hybrid PPL':>10} | {'Subj Bind%':>10} | {'Owner Rev%':>10} | {'Time(s)':>7}")
        print("-" * 108)
        print(
            f"{'Baseline Flat (v2.1)':<22} | {flat_train_eval['acc']:>9.2f}% | "
            f"{flat_recomb_eval['full_acc']:>11.2f}% | {flat_recomb_eval['cont_acc']:>11.2f}% | "
            f"{flat_recomb_eval['ppl']:>10.3f} | {flat_recomb_eval['binding_acc']:>9.2f}% | "
            f"{flat_recomb_eval['owner_revert_rate']:>9.2f}% | {flat_train_s:>6.2f}s"
        )
        print(
            f"{'SS-ENN v2.3 (Hier+RMS)':<22} | {v23_train_eval['acc']:>9.2f}% | "
            f"{v23_recomb_eval['full_acc']:>11.2f}% | {v23_recomb_eval['cont_acc']:>11.2f}% | "
            f"{v23_recomb_eval['ppl']:>10.3f} | {v23_recomb_eval['binding_acc']:>9.2f}% | "
            f"{v23_recomb_eval['owner_revert_rate']:>9.2f}% | {v23_train_s:>6.2f}s"
        )
        print()

        print("--- Table 2: Layer-Wise Hierarchical Top-K, Stiffness k_l & Local Active RMSNorm Profile ---")
        print(f"{'Layer':<6} | {'Flat K':>6} | {'v2.3 K':>6} | {'Flat k_bar(max)':>15} | {'v2.3 k_bar(max)':>15} | "
              f"{'Flat Hybrid RMS':>15} | {'v2.3 Pre->Post RMS (Hybrid)':>27}")
        print("-" * 108)
        for l_idx in range(v23_model.num_layers):
            fk = flat_k_prof[l_idx]
            vk = v23_k_prof[l_idx]
            f_rms_h = flat_rms_hybrid[l_idx][0]
            v_pre_h, v_post_h = v23_rms_hybrid[l_idx]
            print(
                f"L{l_idx+1:<5} | {fk['top_k']:>6d} | {vk['top_k']:>6d} | "
                f"{fk['mean_k']:>6.4f}({fk['max_k']:>6.3f}) | {vk['mean_k']:>6.4f}({vk['max_k']:>6.3f}) | "
                f"{f_rms_h:>15.4f} | {v_pre_h:>10.4f} -> {v_post_h:<10.4f}"
            )
        print()

        print("--- Table 3: +30-Token Free Rollout Audit on 6 Crossed Hybrid Prompts (SS-ENN v2.3 Frozen W_0) ---")
        print(f"Rollout Consistency: v2.3 = {v23_s2_retain_pct:.2f}% ({sum(1 for r in v23_rollouts if r['s2_consistent'])}/6) "
              f"vs Baseline Flat = {flat_s2_retain_pct:.2f}% ({sum(1 for r in flat_rollouts if r['s2_consistent'])}/6) | "
              f"Adj Dup = {v23_mean_adj_dup:.2f}% | Phrase Loops = {v23_mean_loop:.2f}%")
        print("-" * 108)
        for r in v23_rollouts:
            status = "PASS" if r["s2_consistent"] and r["adj_dup_rate"] == 0.0 and r["loop_rate"] == 0.0 else "FAIL"
            print(f"[{status}] {r['name']}")
            print(f"       Expected S_i='{r['expected_subject']}' (Template Owner S_j='{r['template_owner']}') "
                  f"| Seen={r['subjects_seen']} | Dup={r['adj_dup_rate']:.1f}% Loop={r['loop_rate']:.1f}%")
            print(f"       Tail: {r['generated_tail']}")
        print("=" * 96)

    return {
        "scanned_files": num_scanned,
        "flat_train": flat_train_eval,
        "v23_train": v23_train_eval,
        "flat_recomb": flat_recomb_eval,
        "v23_recomb": v23_recomb_eval,
        "flat_k_profile": flat_k_prof,
        "v23_k_profile": v23_k_prof,
        "flat_rms_train": flat_rms_train,
        "flat_rms_hybrid": flat_rms_hybrid,
        "v23_rms_train": v23_rms_train,
        "v23_rms_hybrid": v23_rms_hybrid,
        "v23_rollouts": v23_rollouts,
        "flat_rollouts": flat_rollouts,
        "v23_s2_retain_pct": v23_s2_retain_pct,
        "flat_s2_retain_pct": flat_s2_retain_pct,
        "v23_mean_adj_dup": v23_mean_adj_dup,
        "v23_mean_loop": v23_mean_loop,
        "flat_train_s": flat_train_s,
        "v23_train_s": v23_train_s,
    }


if __name__ == "__main__":
    run_v2_compositional_gen_benchmark(verbose=True)

"""
tests/test_suite.py
Strict verification suite ensuring:
1. Zero usage of autograd / backward() across all organized packages (engine/, models/, benchmarks/, train_eval.py).
2. Purely local 3-Spring-Law updates with (w, w0, k) state tuple and Top-K sparsity.
3. Production Freeze cleanly discards w and k and retains only W_final = W_0.
4. Pass/Fail Gate: Generative error drops to 0.0 and 100% Exact-Match accuracy on all 12 Arabic sentences.
"""
import ast
import os
import pathlib
import sys
import unittest
import torch

# Add project root to sys.path so `python tests/test_suite.py` or `pytest` resolves packages cleanly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import SpringLayer, DFARouter, CausalContext, DynamicSpringLayer, SparseCausalAttention
from models import (
    SpringGenerativeNetwork,
    SpringGenerativeNetworkV2,
    DynamicSpringGenerativeNetwork,
    ContinuousSpringNetwork,
    ScaledSpringNetwork,
    SSENNConfig,
    SSENNProductionModel,
)
from train_eval import ARABIC_CORPUS, build_dataset, train_and_evaluate


class TestSpringMVPArchitecture(unittest.TestCase):
    def test_01_no_autograd_or_backward_in_source(self):
        """
        Scan all Python source files across engine/, models/, benchmarks/, and train_eval.py
        via AST to guarantee zero calls to .backward() or .grad().
        """
        scan_dirs = [
            PROJECT_ROOT / "engine",
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "trainer",
            PROJECT_ROOT / "benchmarks",
        ]
        source_files = [PROJECT_ROOT / "train_eval.py"]
        for d in scan_dirs:
            source_files.extend(sorted(d.glob("*.py")))

        self.assertGreaterEqual(len(source_files), 36, "Expected at least 36 Python files scanned")

        for file_path in source_files:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        self.assertNotEqual(
                            node.func.attr,
                            "backward",
                            f"Forbidden call to .backward() found in {file_path}",
                        )
                        self.assertNotEqual(
                            node.func.attr,
                            "grad",
                            f"Forbidden call to .grad() found in {file_path}",
                        )

    def test_02_spring_layer_state_and_top_k_sparsity(self):
        """Verify (w, w0, k) state tuple, Top-K sparse activation, and local update laws."""
        layer = SpringLayer(
            in_features=64,
            out_features=64,
            top_k=4,
            eta_w=0.05,
            eta_0=0.01,
            gamma=0.1,
            beta_k=0.02,
            use_residual=False,
            seed=123,
        )
        self.assertIsNotNone(layer.w)
        self.assertIsNotNone(layer.w0)
        self.assertIsNotNone(layer.k)
        self.assertTrue(torch.equal(layer.w, layer.w0))
        self.assertTrue(torch.all(layer.k == 0.0))

        x = torch.randn(5, 64)
        out = layer(x)
        self.assertFalse(out.requires_grad)

        # Exactly Top-K = 4 units active per row
        active_counts = (out != 0).sum(dim=-1)
        for count in active_counts.tolist():
            self.assertLessEqual(count, 4)
            self.assertGreater(count, 0)

        # Perform local spring update
        proj_err = torch.randn(5, 64)
        w_before = layer.w.clone()
        w0_before = layer.w0.clone()
        k_before = layer.k.clone()

        stats = layer.local_spring_update(proj_err)
        self.assertGreater(stats["mean_stretch"], 0.0)
        self.assertFalse(torch.equal(layer.w, w_before))
        self.assertFalse(torch.equal(layer.w0, w0_before))
        self.assertFalse(torch.equal(layer.k, k_before))
        self.assertTrue(torch.all((layer.k >= 0.0) & (layer.k <= 1.0)))

    def test_03_dfa_router_fixed_weights(self):
        """Verify DFA feedback matrices B^(l) are fixed and project V -> D properly."""
        router = DFARouter(num_layers=4, d_model=64, vocab_size=37, seed=42)
        b0_before = router.get_matrix(0).clone()
        err = torch.randn(10, 37)
        proj = router.project_error(0, err)
        self.assertEqual(proj.shape, (10, 64))
        self.assertTrue(torch.equal(router.get_matrix(0), b0_before))

    def test_04_production_freeze_discards_w_and_k(self):
        """Verify Production Freeze discards w and k and runs inference solely on W0."""
        vocab, _, _, input_tensor, target_tensor, _ = build_dataset(ARABIC_CORPUS)
        model = SpringGenerativeNetwork(vocab_size=len(vocab), d_model=64, num_layers=4, top_k=4)
        model.train_step(input_tensor, target_tensor)

        # Freeze model
        model.production_freeze()
        self.assertTrue(model.frozen)
        for layer in model.hidden_layers:
            self.assertIsNone(layer.w)
            self.assertIsNone(layer.k)
            self.assertIsNotNone(layer.w0)
        self.assertIsNone(model.output_head.w)
        self.assertIsNone(model.output_head.k)
        self.assertIsNotNone(model.output_head.w0)

        # Inference still works cleanly using W0
        logits = model(input_tensor)
        self.assertEqual(logits.shape, (12, 3, len(vocab)))

    def test_05_full_corpus_100_percent_exact_match_gate(self):
        """
        Pass/Fail Gate:
        - 12 Arabic 3-word sentences
        - Generative error drops to 0.0
        - 100% Exact-Match accuracy using frozen weights W_0
        """
        model, history, frozen_acc, frozen_err, details = train_and_evaluate(
            epochs=200, verbose=False
        )
        self.assertEqual(len(details), 12)
        self.assertEqual(frozen_err, 0.0, f"Expected generative error 0.0, got {frozen_err}")
        self.assertEqual(frozen_acc, 100.0, f"Expected 100.0% Exact Match, got {frozen_acc}%")
        for item in details:
            self.assertTrue(item["match"], f"Mismatch on sentence: {item}")

    def test_06_milestone_2_2_sleep_recycling_and_refractory_gate(self):
        """
        Verify Milestone 2.2 Dual-Stage Sleep Consolidation, Synaptic Recycling,
        In-Place Rejuvenation, Synchronous Compaction, and Transition Refractory Gate.
        """
        model = ScaledSpringNetwork(preset="1.5M", vocab_size=128, max_seq_len=16, seed=42)
        seqs_p1 = [[1, 10, 11, 12, 13, 14, 15, 2], [1, 10, 12, 14, 16, 17, 18, 2]]
        seqs_p2 = [[1, 20, 21, 22, 23, 24, 25, 2], [1, 20, 22, 24, 26, 27, 28, 2]]

        prep1 = model.prepare_continual_phase(seqs_p1, pad_id=0)
        self.assertEqual(prep1["total_rejuvenated"], 0)
        inp1 = torch.tensor([s[:-1] for s in seqs_p1], dtype=torch.long)
        tgt1 = torch.tensor([s[1:] for s in seqs_p1], dtype=torch.long)
        for _ in range(3):
            model.train_minibatch(inp1, tgt1, pad_id=0)

        sleep1 = model.sleep_consolidate_and_recycle(phase_id=1, shared_vocab_ids={1, 2})
        self.assertGreater(sleep1["recycled_this_sleep"], 0)
        self.assertEqual(model.get_active_dimensions(), [32, 32, 32, 32])

        prep2 = model.prepare_continual_phase(seqs_p2, pad_id=0)
        self.assertEqual(prep2["total_rejuvenated"], sleep1["recycled_this_sleep"])
        inp2 = torch.tensor([s[:-1] for s in seqs_p2], dtype=torch.long)
        tgt2 = torch.tensor([s[1:] for s in seqs_p2], dtype=torch.long)
        for _ in range(3):
            model.train_minibatch(inp2, tgt2, pad_id=0)

        sleep2 = model.sleep_consolidate_and_recycle(phase_id=2, shared_vocab_ids={1, 2}, is_final_phase=True)
        self.assertEqual(model.get_active_dimensions(), [60, 60, 60, 60])

        freeze_stats = model.production_freeze(prune_dormant=True)
        self.assertEqual(freeze_stats["active_dims"], [60, 60, 60, 60])

        gen_ids = model.generate([1, 10, 11], max_new_tokens=10, enable_refractory_gate=True)
        for i in range(1, len(gen_ids)):
            self.assertNotEqual(gen_ids[i], gen_ids[i - 1], "Refractory gate must prevent adjacent token duplicates")

    def test_07_milestone_2_3_hierarchical_compositional_generalization(self):
        """
        Verify Milestone 2.3 Hierarchical Top-K & Stiffness Spectrum,
        Parameter-Free Local Active RMSNorm, and Zero-Shot Cross-Recombination
        on Frozen W_0 Only.
        """
        from benchmarks.bench_v2_compositional_gen import run_v2_compositional_gen_benchmark

        results = run_v2_compositional_gen_benchmark(verbose=False)
        self.assertGreaterEqual(results["scanned_files"], 25)
        self.assertGreaterEqual(results["v23_train"]["acc"], 90.0)
        self.assertGreater(results["v23_recomb"]["full_acc"], 90.0)
        self.assertGreaterEqual(results["v23_recomb"]["binding_acc"], 95.0)
        self.assertEqual(results["v23_s2_retain_pct"], 100.0)
        self.assertEqual(results["v23_mean_adj_dup"], 0.0)
        self.assertEqual(results["v23_mean_loop"], 0.0)
        # Verify hierarchical Top-K tapers with depth [24, 24, 20, 16] and RMS stabilized ~1.20
        top_ks = [k["top_k"] for k in results["v23_k_profile"]]
        self.assertEqual(top_ks, [24, 24, 20, 16])
        for _, post_rms in results["v23_rms_hybrid"]:
            self.assertAlmostEqual(post_rms, 1.20, places=2)

    def test_08_milestone_2_4_deep_12layer_block_sparse_capsules(self):
        """
        Verify Milestone 2.4 Contiguous Block-Sparse Elastic Capsules (B=16),
        Depth-Scaled Residual Highway (12-layer stability), Highway-Anchored Deep DFA,
        and Active-Slice CPU Execution Speedup on Frozen W_0 Only.
        """
        from benchmarks.bench_v2_deep_block_sparse import run_v2_deep_block_sparse_benchmark

        results = run_v2_deep_block_sparse_benchmark(verbose=False)
        self.assertTrue(results["ast_ok"])
        self.assertGreaterEqual(results["scanned_files"], 26)

        elem_4l = results["elem_4l"]
        block_4l = results["block_4l"]
        deep_12l = results["deep_12l"]
        naive_12l = results["naive_12l"]
        audit = results["rollout_audit"]
        deep_model = results["deep_12l_model"]

        # 1. Block-Sparse 4L accuracy & genuine CPU throughput speedup over Element-wise 4L
        self.assertGreaterEqual(block_4l["frozen_val"]["acc"], 92.0)
        self.assertGreater(block_4l["train_tok_per_sec"], elem_4l["train_tok_per_sec"])
        self.assertGreater(block_4l["eval_tok_per_sec"], elem_4l["eval_tok_per_sec"])

        # 2. 12-Layer Deep SS-ENN v2.4 stability, stiffness health & capsule entropy
        self.assertEqual(len(deep_model.spring_layers), 12)
        self.assertEqual(deep_model.block_size, 16)
        self.assertGreaterEqual(deep_12l["frozen_val"]["acc"], 92.0)
        self.assertGreater(deep_12l["frozen_val"]["acc"], naive_12l["frozen_val"]["acc"])
        self.assertLessEqual(abs(deep_12l["ratio_post_end_l1"] - 1.0), 0.05)
        self.assertLessEqual(abs(deep_12l["ratio_pre_end_l1"] - 1.0), 0.05)
        self.assertGreaterEqual(deep_12l["mean_train_ent"], 0.90)
        self.assertGreaterEqual(deep_12l["mean_frozen_ent"], 0.90)
        self.assertGreaterEqual(deep_12l["mean_k"], 0.12)
        self.assertLessEqual(deep_12l["mean_k"], 0.35)

        # 3. Contiguous block_size=16 alignment on compacted W_0 matrices
        for layer in deep_model.spring_layers:
            self.assertEqual(layer.w0.shape[0] % 16, 0)

        # 4. Clean 30-token rollout audit on Frozen W_0 Only
        self.assertEqual(audit["entity_consistency_pct"], 100.0)
        self.assertEqual(audit["adjacent_dup_rate_pct"], 0.0)
        self.assertEqual(audit["phrase_loop_rate_pct"], 0.0)

    def test_09_milestone_2_5_reward_modulated_three_factor_alignment(self):
        """
        Verify Milestone 2.5 Reward-Modulated Three-Factor Spring Plasticity:
        1. Critic-free, Zero-Autograd post-training behavioral flip (<=20% -> 100%).
        2. Selective thermal k-melting on rejected paths vs reward locking on preferred paths
           with minimal spillover onto untouched general vocabulary stiffness.
        3. Unseen General Knowledge preservation, Local Active RMSNorm = 1.20, and zero extra W_0 MB.
        4. Single-trajectory scalar reward + EMA baseline mode verification.
        """
        from benchmarks.bench_v2_reward_alignment import run_v2_reward_alignment_benchmark
        from models import ScaledSpringNetwork

        results = run_v2_reward_alignment_benchmark(verbose=False)
        self.assertTrue(results["ast_ok"])
        self.assertGreaterEqual(results["scanned_files"], 27)

        un_beh = results["unaligned_behavior"]
        al_beh = results["aligned_behavior"]
        self.assertLessEqual(un_beh["aligned_rate_pct"], 20.0)
        self.assertGreaterEqual(un_beh["rejected_rate_pct"], 75.0)
        self.assertEqual(al_beh["aligned_rate_pct"], 100.0)
        self.assertEqual(al_beh["rejected_rate_pct"], 0.0)
        self.assertEqual(al_beh["entity_consistency_pct"], 100.0)
        self.assertEqual(al_beh["adjacent_dup_rate_pct"], 0.0)
        self.assertEqual(al_beh["phrase_loop_rate_pct"], 0.0)

        self.assertGreaterEqual(results["aligned_gen_val"]["acc"], 88.0)
        self.assertGreaterEqual(
            results["aligned_pref_eval"]["acc"],
            results["unaligned_pref_eval"]["acc"] + 15.0,
        )
        self.assertEqual(
            results["aligned_mem"]["frozen_footprint_mb"],
            results["unaligned_mem"]["frozen_footprint_mb"],
        )

        stiff = results["stiffness"]
        self.assertLess(stiff["k_rej_vocab_post"], stiff["k_rej_vocab_pre"] * 0.35)
        self.assertGreater(stiff["k_pref_vocab_post"], stiff["k_pref_vocab_pre"] * 3.0)
        self.assertLessEqual(abs(stiff["k_gen_vocab_post"] - stiff["k_gen_vocab_pre"]), 0.02)
        self.assertLess(stiff["k_hid_rej_melted"], stiff["k_hid_rej_before"])
        self.assertGreater(stiff["k_hid_pref_locked"], stiff["k_hid_pref_before"])

        for rms_val in results["post_freeze_normed_rms"]:
            self.assertAlmostEqual(rms_val, 1.20, places=2)

        # Standalone scalar-reward + EMA baseline mode unit test (block-sparse model)
        torch.manual_seed(11)
        mini = ScaledSpringNetwork(
            preset="1.5M",
            vocab_size=64,
            num_layers=2,
            d_init=64,
            d_max=64,
            top_k=32,
            block_size=16,
            enable_block_sparse=True,
        )
        mini.output_head.k.fill_(0.25)
        for layer in mini.spring_layers:
            layer.k.fill_(0.25)
        seq_batch = torch.tensor(
            [
                [1, 5, 6, 7, 10, 11, 2],
                [1, 5, 6, 7, 20, 21, 2],
            ],
            dtype=torch.long,
        )
        stat = mini.align_with_reward_modulation(
            seq_ids=seq_batch,
            prefix_lens=4,
            rewards=[1.0, -0.5],
            use_reward_baseline=True,
        )
        self.assertGreater(stat["valid_decisions"], 0)
        self.assertEqual(stat["mode"], "scalar_reward")
        self.assertGreater(mini.reward_baseline, 0.0)
        self.assertLess(stat["k_head_rej_melted"], stat["k_head_rej_before"])
        self.assertGreater(stat["k_head_pref_locked"], stat["k_head_pref_before"])

    def test_10_phase_3_1_fused_kernels_and_stateful_cache(self):
        """
        Verify v3.0 Phase 3.1 Hardware-Accelerated Kernels, Factorized Highway DFA,
        O(1) Stateful SpringInferenceCache, and Mixed-Precision Kahan Safety:
        1. AST Zero-Autograd across >= 29 Python source files.
        2. Fused in-place (w, w0, k) spring kernels eliminate update matrix churn (0.00 MB),
           reduce full-step ATen allocations >= 40%, speed up training with exact Val parity,
           and execute indexed active-block slice GEMM accurately (< 1e-5 max diff).
        3. FactorizedHighwayDFA cuts 12-layer DFA buffer memory >= 87.5% (99.2% at V=32K),
           accelerates broadcast >= 2x, and converges end-to-end on 12L (> 90% Val Acc).
        4. Stateful O(1) SpringInferenceCache achieves 100.00% exact token sequence parity
           against O(T^2) rollout at +30 and +64 tokens for both 12L Block-Sparse and
           4L Role-Bridge Compositional models.
        5. BF16 + Kahan Compensator prevents sub-ULP underflow & maintains 0 NaN/Inf.
        """
        from benchmarks.bench_v3_kernels_speed import run_v3_kernels_benchmark

        results = run_v3_kernels_benchmark(verbose=False)
        self.assertTrue(results["ast_ok"])
        self.assertGreaterEqual(results["scanned_files"], 29)

        t1_rows = results["table1"]["rows"]
        unfused, fused, fused_fac = t1_rows[0], t1_rows[1], t1_rows[2]
        self.assertGreater(unfused["update_alloc_mb"], 20.0)
        self.assertEqual(fused["update_alloc_mb"], 0.0)
        self.assertEqual(fused_fac["update_alloc_mb"], 0.0)
        self.assertLess(fused_fac["step_alloc_mb"], unfused["step_alloc_mb"] * 0.60)
        self.assertGreaterEqual(fused_fac["speedup_x"], 1.25)
        self.assertGreaterEqual(fused["val_acc_pct"], 88.0)
        self.assertGreaterEqual(fused_fac["val_acc_pct"], 88.0)
        self.assertLessEqual(abs(fused["val_acc_pct"] - unfused["val_acc_pct"]), 1.5)

        sg = results["slice_gemm"]
        self.assertLess(sg["max_abs_diff"], 1e-5)
        self.assertGreaterEqual(sg["flop_reduction_pct"], 50.0)

        t2 = results["table2"]
        s2048 = t2["scaling_rows"][0]
        s32000 = t2["scaling_rows"][-1]
        self.assertGreaterEqual(s2048["mem_saved_pct"], 85.0)
        self.assertGreaterEqual(s2048["speedup_x"], 2.0)
        self.assertGreaterEqual(s32000["mem_saved_pct"], 99.0)
        self.assertGreaterEqual(s32000["speedup_x"], 4.0)
        self.assertGreaterEqual(t2["e2e_12l_val_acc"], 90.0)

        t3 = results["table3"]
        self.assertTrue(t3["all_exact_match"])
        for row in t3["rows"]:
            self.assertEqual(row["exact_match_pct"], 100.0)
            self.assertLess(row["max_logit_diff"], 1e-4)
            if row["horizon"] == 64 and "Role-Bridge" in row["model_label"]:
                self.assertGreaterEqual(row["speedup_x"], 3.0)
            if row["horizon"] == 64 and "12L" in row["model_label"]:
                self.assertGreaterEqual(row["speedup_x"], 1.5)

        t4_rows = results["table4"]["rows"]
        for row in t4_rows:
            self.assertEqual(row["nan_inf_count"], 0)
        bf16_naive, bf16_kahan = t4_rows[1], t4_rows[2]
        self.assertGreaterEqual(bf16_kahan["cos_vs_fp32"], 0.90)
        self.assertGreater(bf16_kahan["cos_vs_fp32"], bf16_naive["cos_vs_fp32"] + 0.30)
        self.assertGreater(bf16_kahan["mean_delta_k"], 0.001)

    def test_11_phase_3_2_manual_config_and_production_model(self):
        """
        Phase 3.2 Gate: Verify explicit manual SSENNConfig validation, zero-alloc
        footprint estimation (Small -> 7B), unified SSENNProductionModel (Zero-[D,V]
        DFA), O(1) parity, BF16 Kahan learning, sleep/alignment lifecycle, and 100%
        bitwise mid-training checkpoint save/resume parity.
        """
        from benchmarks.bench_v3_production_model import run_v3_production_benchmark

        # 1. Config validation error checks
        with self.assertRaises(ValueError):
            SSENNConfig(d_init=50, d_max=128, delta_d=16, block_size=16)
        with self.assertRaises(ValueError):
            SSENNConfig(d_init=64, d_max=128, delta_d=16, block_size=16, top_k_blocks=8)
        with self.assertRaises(ValueError):
            SSENNConfig(d_init=256, d_max=128, delta_d=16, block_size=16)
        with self.assertRaises(ValueError):
            SSENNConfig(dtype="int4")
        with self.assertRaises(ValueError):
            SSENNConfig(device="tpu_unknown")

        cfg_fp32 = SSENNConfig(dtype="float32")
        cfg_bf16 = SSENNConfig(dtype="bfloat16")
        cfg_bf16_ak = SSENNConfig(dtype="bfloat16", kahan_mode="anchor_k")
        cfg_bf16_off = SSENNConfig(dtype="bfloat16", kahan_mode="off")
        self.assertFalse(cfg_fp32.uses_kahan_compensator)
        self.assertEqual(cfg_fp32.effective_kahan_mode, "off")
        self.assertTrue(cfg_bf16.uses_kahan_compensator)
        self.assertEqual(cfg_bf16.effective_kahan_mode, "full")
        self.assertTrue(cfg_bf16_ak.uses_kahan_compensator)
        self.assertEqual(cfg_bf16_ak.effective_kahan_mode, "anchor_k")
        self.assertFalse(cfg_bf16_off.uses_kahan_compensator)
        self.assertAlmostEqual(
            cfg_bf16_ak.estimate_memory_footprint()["kahan_savings_vs_full_pct"],
            100.0 / 3.0,
            places=2,
        )

        # 2. Pure forward side-effect check + sleep + reward alignment + dtype move + Zero-[D,V] DFA check
        unit_cfg = SSENNConfig(
            vocab_size=128, num_layers=3, d_init=32, d_max=64, delta_d=16,
            block_size=16, top_k_blocks=1, max_seq_len=32, attn_top_k=4,
            device="cpu", dtype="float32", seed=17,
        )
        m_unit = SSENNProductionModel(unit_cfg)
        for name, buf in m_unit.named_buffers():
            if buf is not None and buf.ndim == 2 and unit_cfg.vocab_size in buf.shape:
                self.assertTrue(
                    any(k in name for k in ("token_embedding", "head_w", "head_w0", "head_k")),
                    f"Unexpected [D, V] buffer found: {name} {tuple(buf.shape)}",
                )

        x_seq = torch.randint(3, 120, (4, 12), dtype=torch.long)
        _ = m_unit.train_minibatch(x_seq[:, :-1], x_seq[:, 1:], pad_id=0)
        wins_before = m_unit.spring_layers[0].win_counts.clone()
        _ = m_unit.forward(x_seq[:, :-1])
        self.assertTrue(torch.equal(wins_before, m_unit.spring_layers[0].win_counts))

        sleep_rec = m_unit.sleep_consolidate(phase_id=1)
        self.assertGreater(sum(sleep_rec["locked_per_layer"]), 0)
        align_rec = m_unit.align_with_reward_modulation(x_seq, prefix_lens=3, rewards=[1.0, -1.0, 1.0, -1.0], pad_id=0)
        self.assertGreater(align_rec["valid_decisions"], 0)

        m_unit.to_device_and_dtype("cpu", "bfloat16")
        self.assertEqual(m_unit.token_embedding.dtype, torch.bfloat16)
        self.assertEqual(m_unit.spring_layers[0].w.dtype, torch.bfloat16)
        self.assertEqual(m_unit.token_salience.dtype, torch.float32)
        self.assertEqual(m_unit.spring_layers[0].win_counts.dtype, torch.float32)

        # 3. Full Phase 3.2 Benchmark Audit
        res = run_v3_production_benchmark(epochs=16, batch_size=32, seq_len=36, verbose=False)
        self.assertTrue(res["ast_zero_autograd"])
        self.assertGreaterEqual(res["ast_verified_files"], 31)

        t1 = res["table1"]
        self.assertEqual(len(t1), 6)
        params_seq = [r["footprint"]["max_train_params"] for r in t1]
        self.assertEqual(params_seq, sorted(params_seq))
        self.assertGreater(t1[-1]["footprint"]["max_train_params"], 5_000_000_000)
        self.assertAlmostEqual(t1[-1]["footprint"]["kahan_savings_vs_full_pct"], 100.0 / 3.0, places=2)
        for r in t1:
            fp = r["footprint"]
            self.assertGreaterEqual(fp["dfa_memory_savings_pct"], 75.0)
            self.assertLess(fp["frozen_w0_mb"], fp["max_train_mb"])

        t2 = res["table2"]
        self.assertGreaterEqual(t2["frozen_train_acc"], 85.0)
        self.assertGreaterEqual(t2["frozen_val_acc"], 92.5)
        self.assertLess(t2["frozen_val_ppl"], 2.30)
        self.assertGreaterEqual(t2["memory_saved_pct"], 45.0)
        self.assertEqual(t2["reloaded_frozen_logit_diff"], 0.0)
        self.assertEqual(t2["gen_parity_pct"], 100.0)
        self.assertGreater(t2["gen_speedup"], 1.2)
        self.assertGreaterEqual(t2["entity_consistency_pct"], 80.0)
        self.assertEqual(t2["adjacent_dup_rate_pct"], 0.0)
        self.assertEqual(t2["phrase_loop_rate_pct"], 0.0)
        self.assertTrue(t2["bf16_finite"])
        self.assertLess(t2["bf16_final_ce"], t2["bf16_ep1_ce"])
        self.assertGreater(t2["bf16_kahan_mb"], 0.0)
        self.assertGreater(t2["bf16_mean_k"], 0.0)
        self.assertGreater(t2["bf16_ak_kahan_mb"], 0.0)
        self.assertGreater(t2["bf16_ak_saved_pct"], 30.0)
        self.assertEqual(t2["bf16_ak_res_w_count"], 0)
        self.assertTrue(t2["bf16_ak_finite"])

        t3 = res["table3"]
        self.assertEqual(len(t3), 2)
        for row in t3:
            self.assertEqual(row["pre_freeze_weight_diff"], 0.0)
            self.assertEqual(row["frozen_weight_diff"], 0.0)
            self.assertEqual(row["logit_diff"], 0.0)
            self.assertEqual(row["rollout_parity_pct"], 100.0)

    def test_12_phase_3_3_atomic_vault_mmap_and_fault_tolerant_resume(self):
        """
        Phase 3.3 Gate: Verify O(1) RAM MMapTokenDataset & ResumableBatchStreamer
        cursor parity, 0.00e+00 mid-epoch sudden kill resume parity across FP32
        & BF16 (anchor_k), self-healing backup fallback on truncated/bit-flipped
        checkpoints, real SIGTERM preemption save, and compacted best_frozen_w0.pt
        export alongside Zero-Autograd AST verification across >= 36 files.
        """
        from benchmarks.bench_v3_trainer_fault_tolerance import (
            run_v3_trainer_fault_tolerance_benchmark,
        )

        res = run_v3_trainer_fault_tolerance_benchmark(verbose=False)
        self.assertTrue(res["ast_zero_autograd_ok"])
        self.assertGreaterEqual(res["ast_files_verified"], 36)

        t1 = res["table1"]
        self.assertEqual(len(t1), 2)
        for row in t1:
            self.assertTrue(row["sha256_ok"])
            self.assertEqual(row["cursor_parity_pct"], 100.0)
            self.assertLess(row["meta_ram_kb"], 16.0)
            self.assertGreater(row["stream_tok_s"], 10_000.0)

        t2 = res["table2"]
        self.assertEqual(len(t2), 2)
        for row in t2:
            self.assertEqual(row["pre_freeze_diff"], 0.0)
            self.assertEqual(row["frozen_w0_diff"], 0.0)
            self.assertEqual(row["val_parity_pct"], 100.0)
        self.assertGreaterEqual(t2[0]["val_acc_pct"], 85.0)
        self.assertGreater(t2[1]["val_acc_pct"], 15.0)

        t3 = res["table3"]
        self.assertEqual(len(t3), 2)
        for row in t3:
            self.assertTrue(row["primary_corrupted"])
            self.assertEqual(row["resume_source"], "backup")
            self.assertEqual(row["recovered_step"], 20)
            self.assertTrue(row["finite_ok"])
            self.assertGreaterEqual(row["final_val_acc"], 85.0)

        t4 = res["table4"]
        self.assertEqual(t4["preempt_status"], "preempted")
        self.assertEqual(t4["preempt_step"], 22)
        self.assertEqual(t4["emergency_saves"], 1)
        self.assertEqual(t4["post_resume_source"], "latest")
        self.assertEqual(t4["post_resume_step"], 22)
        self.assertTrue(t4["best_frozen_w0_exists"])
        self.assertTrue(t4["best_frozen_is_frozen"])
        self.assertTrue(t4["best_frozen_has_no_train_bufs"])
        self.assertGreaterEqual(t4["best_frozen_val_acc"], 85.0)
        self.assertIn("emergency_preemption_save", t4["telemetry_events"])
        self.assertIn("checkpoint_resumed", t4["telemetry_events"])
        self.assertIn("best_frozen_w0_exported", t4["telemetry_events"])
        self.assertIn("epoch_end", t4["telemetry_events"])

    def test_13_phase_3_4_unified_cli_workflow_end_to_end(self):
        """
        Phase 3.4 Gate: Exercise all 4 subcommands of `python -m trainer.cli`
        (`estimate`, `prepare-data`, `train`, `generate`) end-to-end on clean
        temporary storage alongside full AST Zero-Autograd verification across
        >= 38 Python files.
        """
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from engine.tokenizer import EnglishSubwordTokenizer
        from models.config_v3 import SSENNConfig
        from models.model_scaled import ScaledSpringNetwork
        from models.model_v3_production import SSENNProductionModel
        from trainer.cli import main as cli_main
        from trainer.checkpoint_vault import AtomicCheckpointVault
        from trainer.mmap_dataset import BinaryShardBuilder, MMapTokenDataset
        from benchmarks.bench_scale_english_cpu import verify_all_repo_files_zero_autograd

        ast_ok, n_files = verify_all_repo_files_zero_autograd()
        self.assertTrue(ast_ok)
        self.assertGreaterEqual(n_files, 38)

        with tempfile.TemporaryDirectory(prefix="ssenn_cli_e2e_") as tmp_dir_str:
            tmp_root = Path(tmp_dir_str)
            data_dir = tmp_root / "shards"
            vault_dir = tmp_root / "vault"

            # 1. `estimate` (verify JSON analytical table + recommendation for 7B anchor_k)
            buf_est = io.StringIO()
            with redirect_stdout(buf_est):
                rc = cli_main(
                    [
                        "estimate",
                        "--vocab-size", "32000",
                        "--num-layers", "32",
                        "--d-init", "3072",
                        "--d-max", "6656",
                        "--delta-d", "512",
                        "--block-size", "16",
                        "--top-k-blocks", "32",
                        "--max-seq-len", "2048",
                        "--attn-top-k", "16",
                        "--dtype", "bfloat16",
                        "--kahan-mode", "anchor_k",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            est_data = json.loads(buf_est.getvalue())
            self.assertEqual(est_data["kahan_mode"], "anchor_k")
            self.assertAlmostEqual(est_data["kahan_savings_vs_full_pct"], 100.0 / 3.0, places=2)
            self.assertGreater(est_data["max_train_params"], 7_000_000_000)
            self.assertIn("A100", est_data["recommended_hardware"])

            # 2. `prepare-data` (tokenize built-in TinyStories & build SHA-256 binary shards)
            buf_prep = io.StringIO()
            with redirect_stdout(buf_prep):
                rc = cli_main(
                    [
                        "prepare-data",
                        "--builtin-tinystories",
                        "--out-dir", str(data_dir),
                        "--vocab-size", "2048",
                        "--seq-len", "36",
                        "--target-min-tokens", "12000",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue((data_dir / "tokenizer.json").exists())
            self.assertTrue((data_dir / "train_shard.bin").exists())
            self.assertTrue((data_dir / "train_shard.meta.json").exists())
            self.assertTrue((data_dir / "val_shard.bin").exists())
            self.assertTrue((data_dir / "val_shard.meta.json").exists())

            # 3. `train` (run Phase 1 up to 3 epochs, then auto-resume to 12 epochs)
            buf_tr1 = io.StringIO()
            with redirect_stdout(buf_tr1):
                rc = cli_main(
                    [
                        "train",
                        "--data-dir", str(data_dir),
                        "--vault-dir", str(vault_dir),
                        "--num-layers", "6",
                        "--d-init", "64",
                        "--d-max", "128",
                        "--delta-d", "16",
                        "--block-size", "16",
                        "--top-k-blocks", "2",
                        "--max-seq-len", "64",
                        "--attn-top-k", "8",
                        "--dtype", "float32",
                        "--kahan-mode", "off",
                        "--epochs", "3",
                        "--batch-size", "32",
                        "--ckpt-every-steps", "5",
                        "--quiet",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertIn("resumed=False", buf_tr1.getvalue())

            buf_tr2 = io.StringIO()
            with redirect_stdout(buf_tr2):
                rc = cli_main(
                    [
                        "train",
                        "--data-dir", str(data_dir),
                        "--vault-dir", str(vault_dir),
                        "--num-layers", "6",
                        "--d-init", "64",
                        "--d-max", "128",
                        "--delta-d", "16",
                        "--block-size", "16",
                        "--top-k-blocks", "2",
                        "--max-seq-len", "64",
                        "--attn-top-k", "8",
                        "--dtype", "float32",
                        "--kahan-mode", "off",
                        "--epochs", "12",
                        "--batch-size", "32",
                        "--ckpt-every-steps", "5",
                        "--auto-resume",
                        "--quiet",
                    ]
                )
            self.assertEqual(rc, 0)
            tr2_out = buf_tr2.getvalue()
            self.assertIn("resumed=True", tr2_out)
            self.assertTrue((vault_dir / "best_frozen_w0.pt").exists())
            telemetry = AtomicCheckpointVault(vault_dir).read_telemetry()
            self.assertGreater(len(telemetry), 0)

            # 4. `generate` (O(1) stateful cache inference from compacted best_frozen_w0.pt)
            buf_gen = io.StringIO()
            with redirect_stdout(buf_gen):
                rc = cli_main(
                    [
                        "generate",
                        "--vault-dir", str(vault_dir),
                        "--prompt", "Once upon a time",
                        "--max-new-tokens", "20",
                    ]
                )
            self.assertEqual(rc, 0)
            gen_out = buf_gen.getvalue()
            self.assertIn("O(1) Stateful Cache", gen_out)
            self.assertIn("Frozen=True", gen_out)
            self.assertIn("Once upon a time", gen_out)
            self.assertIn("tok/s", gen_out)

            # 5. Hotfix v3.0.1 Gate: Tensor vs List salience calibration parity & low hub vs high entity scores
            tok_loaded = EnglishSubwordTokenizer.load(data_dir / "tokenizer.json")
            with MMapTokenDataset(data_dir, shard_name="train_shard", verify_checksum=True) as ds_cal:
                cal_n = min(len(ds_cal), 256)
                x_cal, y_cal = ds_cal.get_batch(list(range(cal_n)))
                tensor_seqs = torch.cat([x_cal, y_cal[:, -1:]], dim=1)
                list_seqs = tensor_seqs.tolist()

            cal_cfg = SSENNConfig(
                vocab_size=tok_loaded.vocab_size,
                num_layers=2,
                d_init=32,
                d_max=32,
                delta_d=0,
                block_size=16,
                top_k_blocks=1,
                max_seq_len=64,
                attn_top_k=4,
                dtype="float32",
                kahan_mode="off",
                seed=42,
            )
            m_list = SSENNProductionModel(cal_cfg)
            m_tensor = SSENNProductionModel(cal_cfg)
            m_list.calibrate_salience_from_tokens(
                list_seqs, pad_id=tok_loaded.pad_id, bos_id=tok_loaded.bos_id, eos_id=tok_loaded.eos_id
            )
            m_tensor.calibrate_salience_from_tokens(
                tensor_seqs, pad_id=tok_loaded.pad_id, bos_id=tok_loaded.bos_id, eos_id=tok_loaded.eos_id
            )
            self.assertEqual(float((m_list.token_salience - m_tensor.token_salience).abs().max().item()), 0.0)
            self.assertEqual(float((m_list.capacitor_alpha - m_tensor.capacitor_alpha).abs().max().item()), 0.0)

            s_list = ScaledSpringNetwork(
                vocab_size=tok_loaded.vocab_size, num_layers=2, d_init=32, d_max=32,
                delta_d=0, top_k=8, attn_top_k=4, max_seq_len=64, seed=42,
            )
            s_tensor = ScaledSpringNetwork(
                vocab_size=tok_loaded.vocab_size, num_layers=2, d_init=32, d_max=32,
                delta_d=0, top_k=8, attn_top_k=4, max_seq_len=64, seed=42,
            )
            s_list.calibrate_salience_from_tokens(
                list_seqs, pad_id=tok_loaded.pad_id, bos_id=tok_loaded.bos_id, eos_id=tok_loaded.eos_id
            )
            s_tensor.calibrate_salience_from_tokens(
                tensor_seqs, pad_id=tok_loaded.pad_id, bos_id=tok_loaded.bos_id, eos_id=tok_loaded.eos_id
            )
            self.assertEqual(float((s_list.token_salience - s_tensor.token_salience).abs().max().item()), 0.0)

            the_id = tok_loaded.token_to_id["the"]
            the_sal = float(m_tensor.token_salience[the_id].item())
            self.assertLessEqual(the_sal, 0.75)
            for entity_word in ("bird", "rabbit", "squirrel", "kitten"):
                ent_id = tok_loaded.token_to_id[entity_word]
                self.assertGreater(float(m_tensor.token_salience[ent_id].item()), 1.30)

            # 6. Hotfix v3.0.1 Gate: BinaryShardBuilder long-paragraph strided chunking (> 3 * seq_len)
            long_dir = tmp_root / "long_shards"
            long_story = " ".join([
                "Once upon a time in the green valley the little bird sang a sweet song.",
                "Then the young rabbit hopped along the soft grass and found a fresh carrot.",
                "Suddenly a gentle rain began to fall while the wise owl watched from oak.",
                "Finally everyone returned home safely beside the quiet forest river."
            ])
            long_seq_len = 16
            encoded_long = tok_loaded.encode(long_story, add_bos=True, add_eos=True)
            self.assertGreater(len(encoded_long), 3 * long_seq_len)
            long_meta = BinaryShardBuilder.build_from_stories(
                stories=[long_story],
                tokenizer=tok_loaded,
                seq_len=long_seq_len,
                output_dir=long_dir,
                shard_name="long_chunk_shard",
            )
            self.assertGreaterEqual(int(long_meta["num_sequences"]), 4)
            with MMapTokenDataset(long_dir, shard_name="long_chunk_shard", verify_checksum=True) as ds_long:
                self.assertGreaterEqual(len(ds_long), 4)
                x_last, y_last = ds_long[len(ds_long) - 1]
                # Tail chunk preserves tokens from end of story (not discarded after first window)
                self.assertIn(int(tok_loaded.eos_id), y_last.tolist())

            # 7. Hotfix v3.0.1 Gate: CLI --seq-len conflict guard & automatic --max-seq-len raise notice
            with self.assertRaises(ValueError):
                cli_main([
                    "train",
                    "--data-dir", str(data_dir),
                    "--vault-dir", str(tmp_root / "vault_mismatch"),
                    "--seq-len", "99",
                    "--quiet",
                ])

            buf_notice = io.StringIO()
            with redirect_stdout(buf_notice):
                rc = cli_main([
                    "train",
                    "--data-dir", str(data_dir),
                    "--vault-dir", str(tmp_root / "vault_raise_seq"),
                    "--num-layers", "2",
                    "--d-init", "32",
                    "--d-max", "32",
                    "--delta-d", "0",
                    "--block-size", "16",
                    "--top-k-blocks", "1",
                    "--max-seq-len", "16",
                    "--attn-top-k", "4",
                    "--dtype", "float32",
                    "--kahan-mode", "off",
                    "--epochs", "5",
                    "--batch-size", "128",
                    "--sleep-every-epochs", "2",
                    "--sleep-min-epoch", "3",
                    "--quiet",
                ])
            self.assertEqual(rc, 0)
            self.assertIn("Raising --max-seq-len from 16 to 36", buf_notice.getvalue())
            self.assertIn("shard_seq_len=36 | max_seq_len=36", buf_notice.getvalue())
            sleep_telemetry = [
                rec
                for rec in AtomicCheckpointVault(tmp_root / "vault_raise_seq").read_telemetry()
                if rec.get("event") == "sleep_consolidation"
            ]
            self.assertEqual([rec["epoch"] for rec in sleep_telemetry], [4])


if __name__ == "__main__":
    unittest.main(verbosity=2)

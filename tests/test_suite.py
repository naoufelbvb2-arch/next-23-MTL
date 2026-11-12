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
            PROJECT_ROOT / "benchmarks",
        ]
        source_files = [PROJECT_ROOT / "train_eval.py"]
        for d in scan_dirs:
            source_files.extend(sorted(d.glob("*.py")))

        self.assertGreaterEqual(len(source_files), 15, "Expected at least 15 Python files scanned")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)

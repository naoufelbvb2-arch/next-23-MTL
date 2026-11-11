# Architectural & Implementation Spec: Spring-Link Local Learning Network (MVP)

## Objective
Implement and validate a backpropagation-free multi-layer causal neural network trained strictly via local elastic spring dynamics and Direct Feedback Alignment (DFA), targeting 100% exact-match generation on a structured corpus before freezing to static weights.

## System Constraints & Invariants
1. NO use of `torch.autograd.backward()`, `loss.backward()`, or computational graph retention.
2. Parameter updates MUST be purely local: each layer updates using only its pre-synaptic activation, post-synaptic local error projection, and internal spring states.
3. Every trainable connection maintains a state tuple: (w, w0, k).
4. Training terminates with a clean export: W_prod = w0, discarding w and k.

## Hyperparameter Baselines
- vocab_size: Dynamic based on corpus (approx 25 tokens)
- d_model: 64
- num_layers: 4
- top_k: 4 (Winner-Take-All per layer)
- eta_w (stretch rate): 0.05
- eta_0 (anchor drift rate): 0.01
- gamma (spring restoring force): 0.1
- beta_k (stiffening rate): 0.02
- epochs: 200

## Core File Layout
- `engine/spring_layer.py`: Custom PyTorch module holding (w, w0, k) and manual tensor updates (with `torch.no_grad()`).
- `engine/dfa_router.py`: Fixed random feedback matrices B^(l) for direct error injection.
- `engine/causal_context.py`: Autoregressive sequence processor and embedding table.
- `model.py`: Full model wrapper integrating the 4 spring layers and generation head.
- `train_eval.py`: Single-phase training loop, accuracy tracking, and weight-freezing test.
- `test_suite.py`: Strict verification ensuring autograd is disabled and testing production consolidation.

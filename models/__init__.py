"""
models/__init__.py
Clean modular package exporting the 5 SS-ENN model architectures:
  1. SpringGenerativeNetwork         (model_mvp.py)
  2. SpringGenerativeNetworkV2       (model_v2_attn.py)
  3. DynamicSpringGenerativeNetwork  (model_dynamic.py)
  4. ContinuousSpringNetwork         (model_continuous.py)
  5. ScaledSpringNetwork             (model_scaled.py)
"""
from .model_mvp import SpringGenerativeNetwork
from .model_v2_attn import SpringGenerativeNetworkV2
from .model_dynamic import DynamicSpringGenerativeNetwork
from .model_continuous import ContinuousSpringNetwork
from .model_scaled import ScaledSpringNetwork, SCALE_PRESETS

__all__ = [
    "SpringGenerativeNetwork",
    "SpringGenerativeNetworkV2",
    "DynamicSpringGenerativeNetwork",
    "ContinuousSpringNetwork",
    "ScaledSpringNetwork",
    "SCALE_PRESETS",
]

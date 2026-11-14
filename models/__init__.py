"""
models/__init__.py
Clean modular package exporting the SS-ENN model architectures and v3.0 production config:
  1. SpringGenerativeNetwork         (model_mvp.py)
  2. SpringGenerativeNetworkV2       (model_v2_attn.py)
  3. DynamicSpringGenerativeNetwork  (model_dynamic.py)
  4. ContinuousSpringNetwork         (model_continuous.py)
  5. ScaledSpringNetwork             (model_scaled.py)
  6. SSENNConfig                     (config_v3.py)
  7. SSENNProductionModel            (model_v3_production.py)
"""
from .model_mvp import SpringGenerativeNetwork
from .model_v2_attn import SpringGenerativeNetworkV2
from .model_dynamic import DynamicSpringGenerativeNetwork
from .model_continuous import ContinuousSpringNetwork
from .model_scaled import ScaledSpringNetwork, SCALE_PRESETS
from .config_v3 import SSENNConfig
from .model_v3_production import SSENNProductionModel

__all__ = [
    "SpringGenerativeNetwork",
    "SpringGenerativeNetworkV2",
    "DynamicSpringGenerativeNetwork",
    "ContinuousSpringNetwork",
    "ScaledSpringNetwork",
    "SCALE_PRESETS",
    "SSENNConfig",
    "SSENNProductionModel",
]

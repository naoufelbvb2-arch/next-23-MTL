"""Spring-Link Local Learning Engine Package."""
from .spring_layer import SpringLayer
from .dynamic_spring_layer import DynamicSpringLayer
from .sparse_attention import SparseCausalAttention
from .dfa_router import DFARouter
from .causal_context import CausalContext
from .tokenizer import EnglishSubwordTokenizer
from .kernels import (
    TensorAllocationTracker,
    SpringScratchBuffer,
    MixedPrecisionCompensator,
    fused_spring_update_inplace,
    indexed_block_activate_highway,
    indexed_block_matmul_slice,
    batch_active_block_union_update,
    FactorizedHighwayDFA,
    SpringInferenceCache,
    normalize_token_sequences,
)

__all__ = [
    "SpringLayer",
    "DynamicSpringLayer",
    "SparseCausalAttention",
    "DFARouter",
    "CausalContext",
    "EnglishSubwordTokenizer",
    "TensorAllocationTracker",
    "SpringScratchBuffer",
    "MixedPrecisionCompensator",
    "fused_spring_update_inplace",
    "indexed_block_activate_highway",
    "indexed_block_matmul_slice",
    "batch_active_block_union_update",
    "FactorizedHighwayDFA",
    "SpringInferenceCache",
    "normalize_token_sequences",
]

"""Fault-Tolerant Cloud/Colab Trainer & Memory-Mapped Data Streaming (v3.0 Phase 3.3)."""
from .mmap_dataset import BinaryShardBuilder, MMapTokenDataset, ResumableBatchStreamer
from .checkpoint_vault import AtomicCheckpointVault, PreemptionSignalHandler
from .colab_orchestrator import ElasticCloudTrainer, evaluate_validation_batches

__all__ = [
    "BinaryShardBuilder",
    "MMapTokenDataset",
    "ResumableBatchStreamer",
    "AtomicCheckpointVault",
    "PreemptionSignalHandler",
    "ElasticCloudTrainer",
    "evaluate_validation_batches",
]

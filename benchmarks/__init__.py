"""Benchmarks and tiered Arabic/English evaluation corpora for SS-ENN."""
from .corpus_data import CORPUS_TIERS, CONTINUOUS_PARAGRAPHS_CORPUS
from .english_dataset import build_english_tinystories_corpus, CANONICAL_SEED_PROMPTS
from .bench_v2_sleep_continual import run_v2_sleep_continual_benchmark
from .bench_v2_compositional_gen import run_v2_compositional_gen_benchmark

__all__ = [
    "CORPUS_TIERS",
    "CONTINUOUS_PARAGRAPHS_CORPUS",
    "build_english_tinystories_corpus",
    "CANONICAL_SEED_PROMPTS",
    "run_v2_sleep_continual_benchmark",
    "run_v2_compositional_gen_benchmark",
]

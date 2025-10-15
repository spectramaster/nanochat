"""Typed runtime configuration schemas for NanoChat scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseTrainConfig:
    """Hyperparameter bundle for ``scripts/base_train.py``."""

    run: str = "dummy"
    depth: int = 20
    max_seq_len: int = 2048
    num_iterations: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: int = 20
    device_batch_size: int = 32
    total_batch_size: int = 524_288
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    weight_decay: float = 0.0
    matrix_lr: float = 0.02
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_tokens: int = 20 * 524_288
    core_metric_every: int = 2_000
    core_metric_max_per_task: int = 500
    sample_every: int = 2_000
    model_tag: str = ""
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.2
    final_lr_frac: float = 0.0

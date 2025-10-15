"""Dataloader utilities built on top of the dataset module."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Tuple

import torch

from nanochat.data.dataset import DatasetConfig, parquets_iter_batched
from nanochat.utils import get_dist_info

TokenizerFactory = Callable[[], "TokenizerProtocol"]


class TokenizerProtocol:
    """Protocol defining the tokenizer methods used by the dataloader."""

    def get_bos_token_id(self) -> int:  # pragma: no cover - interface declaration
        raise NotImplementedError

    def encode(
        self, texts: Iterable[str], prepend: int | None = None, num_threads: int = 4
    ) -> Iterable[Iterable[int]]:  # pragma: no cover - interface declaration
        raise NotImplementedError


def tokenizing_distributed_data_loader(
    batch_size: int,
    seq_len: int,
    split: str,
    tokenizer_factory: TokenizerFactory,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    config: DatasetConfig | None = None,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Stream text from parquet shards, tokenize, and yield training batches."""

    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    needed_tokens = batch_size * seq_len + 1
    tokenizer = tokenizer_factory()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)

    # infinite iterator over document batches
    def document_batches():
        while True:
            # batch will iterate in group size of the parquet files, usually e.g. 1024 rows
            for batch in parquets_iter_batched(
                split=split, start=ddp_rank, step=ddp_world_size, config=config
            ):
                # for the tokenizer we might want to go in usually smaller batches, e.g. 128 rows
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(
                doc_batch, prepend=bos_token, num_threads=tokenizer_threads
            )
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(batch_size, seq_len).to(
            device="cuda", dtype=torch.int32, non_blocking=True
        )
        targets = targets_cpu.view(batch_size, seq_len).to(
            device="cuda", dtype=torch.int64, non_blocking=True
        )
        yield inputs, targets

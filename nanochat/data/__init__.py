"""Public accessors for NanoChat data utilities."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

_MODULE_ALIASES: Dict[str, str] = {
    "dataset": "nanochat.data.dataset",
    "dataloader": "nanochat.data.dataloader",
    "tokenizer": "nanochat.data.tokenizer",
    "DatasetConfig": "nanochat.data.dataset",
    "DEFAULT_CONFIG": "nanochat.data.dataset",
    "list_parquet_files": "nanochat.data.dataset",
    "parquets_iter_batched": "nanochat.data.dataset",
    "download_single_file": "nanochat.data.dataset",
    "download_dataset": "nanochat.data.dataset",
    "tokenizing_distributed_data_loader": "nanochat.data.dataloader",
    "TokenizerProtocol": "nanochat.data.dataloader",
    "RustBPETokenizer": "nanochat.data.tokenizer",
    "load_tokenizer": "nanochat.data.tokenizer",
    "get_token_bytes": "nanochat.data.tokenizer",
}

__all__ = list(_MODULE_ALIASES)


def __getattr__(name: str) -> Any:
    target = _MODULE_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module 'nanochat.data' has no attribute {name!r}")

    module = import_module(target)
    if name in {"dataset", "dataloader", "tokenizer"}:
        value = module
    else:
        value = getattr(module, name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - trivial wrapper
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - static typing helpers
    from nanochat.data import dataloader, dataset, tokenizer
    from nanochat.data.dataset import (
        DatasetConfig,
        DEFAULT_CONFIG,
        download_dataset,
        download_single_file,
        list_parquet_files,
        parquets_iter_batched,
    )
    from nanochat.data.dataloader import TokenizerProtocol, tokenizing_distributed_data_loader
    from nanochat.data.tokenizer import RustBPETokenizer, get_token_bytes, load_tokenizer

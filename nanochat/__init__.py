"""Public NanoChat Python API with optional, lazily-loaded submodules.

The original monolithic package imported heavyweight dependencies such as
``torch`` on import, which broke tooling (e.g. ``pytest``) whenever optional
runtime dependencies were unavailable.  To keep backwards compatibility while
allowing lightweight imports for configuration utilities, we expose the same
module level attributes as before but load them on demand.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

_MODULE_ALIASES: Dict[str, str] = {
    "gpt": "nanochat.core.gpt",
    "loss_eval": "nanochat.core.loss_eval",
    "adamw": "nanochat.core.adamw",
    "dataset": "nanochat.data.dataset",
    "dataloader": "nanochat.data.dataloader",
    "tokenizer": "nanochat.data.tokenizer",
    "engine": "nanochat.runtime.engine",
    "execution": "nanochat.runtime.execution",
    "muon": "nanochat.runtime.muon",
    "ColoredFormatter": "nanochat.utils.logging",
    "setup_default_logging": "nanochat.utils.logging",
    "DummyWandb": "nanochat.utils.distributed",
    "compute_cleanup": "nanochat.utils.distributed",
    "compute_init": "nanochat.utils.distributed",
    "get_dist_info": "nanochat.utils.distributed",
    "is_ddp": "nanochat.utils.distributed",
    "get_base_dir": "nanochat.utils.paths",
    "print0": "nanochat.utils.display",
    "print_banner": "nanochat.utils.display",
}

__all__ = list(_MODULE_ALIASES)


def __getattr__(name: str) -> Any:
    """Dynamically import optional submodules only when accessed."""

    target = _MODULE_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module 'nanochat' has no attribute {name!r}")

    module = import_module(target)
    if name in {"ColoredFormatter", "setup_default_logging"}:
        value = getattr(module, name)
    elif name in {
        "DummyWandb",
        "compute_cleanup",
        "compute_init",
        "get_dist_info",
        "is_ddp",
    }:
        value = getattr(module, name)
    elif name in {"get_base_dir", "print0", "print_banner"}:
        value = getattr(module, name)
    else:
        value = module

    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - trivial wrapper
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    from nanochat.core import adamw, gpt, loss_eval
    from nanochat.data import dataloader, dataset, tokenizer
    from nanochat.runtime import engine, execution, muon
    from nanochat.utils import (
        ColoredFormatter,
        DummyWandb,
        compute_cleanup,
        compute_init,
        get_base_dir,
        get_dist_info,
        is_ddp,
        print0,
        print_banner,
        setup_default_logging,
    )

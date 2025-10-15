"""Lazily-resolved utility helpers for NanoChat."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

_MODULE_ALIASES: Dict[str, str] = {
    "logging": "nanochat.utils.logging",
    "distributed": "nanochat.utils.distributed",
    "paths": "nanochat.utils.paths",
    "display": "nanochat.utils.display",
    "ColoredFormatter": "nanochat.utils.logging",
    "setup_default_logging": "nanochat.utils.logging",
    "compute_cleanup": "nanochat.utils.distributed",
    "compute_init": "nanochat.utils.distributed",
    "get_dist_info": "nanochat.utils.distributed",
    "is_ddp": "nanochat.utils.distributed",
    "DummyWandb": "nanochat.utils.distributed",
    "get_base_dir": "nanochat.utils.paths",
    "print0": "nanochat.utils.display",
    "print_banner": "nanochat.utils.display",
}

__all__ = list(_MODULE_ALIASES)


def __getattr__(name: str) -> Any:
    target = _MODULE_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module 'nanochat.utils' has no attribute {name!r}")

    module = import_module(target)
    if name in {"logging", "distributed", "paths", "display"}:
        value = module
    else:
        value = getattr(module, name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - trivial wrapper
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from nanochat.utils import display, distributed, logging, paths
    from nanochat.utils.display import print0, print_banner
    from nanochat.utils.distributed import (
        DummyWandb,
        compute_cleanup,
        compute_init,
        get_dist_info,
        is_ddp,
    )
    from nanochat.utils.logging import ColoredFormatter, setup_default_logging
    from nanochat.utils.paths import get_base_dir

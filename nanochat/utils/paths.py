"""Path utilities used across the project."""

from __future__ import annotations

import os


def get_base_dir() -> str:
    """Return the base directory for NanoChat caches and intermediates."""

    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ["NANOCHAT_BASE_DIR"]
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _missing_modules(names: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip expensive tokenizer tests when optional deps are unavailable."""

    optional = ["regex", "rustbpe", "tiktoken"]
    missing = _missing_modules(optional)
    if not missing:
        return

    skip_marker = pytest.mark.skip(
        reason=f"optional dependencies missing: {', '.join(missing)}",
    )
    for item in items:
        if item.nodeid.startswith("tests/test_rustbpe.py"):
            item.add_marker(skip_marker)

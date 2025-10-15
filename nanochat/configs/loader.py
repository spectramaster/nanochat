"""Structured configuration loader for NanoChat scripts."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from dataclasses import fields, is_dataclass, replace
from typing import Any, Dict, Iterable, Mapping, MutableMapping

try:  # pragma: no cover - optional dependency
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    tomllib = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - yaml is optional
    yaml = None  # type: ignore


def _parse_cli_value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return raw


def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".json"}:
        return json.loads(path.read_text())
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML config files")
        return yaml.safe_load(path.read_text()) or {}
    if suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("tomllib is required to load TOML config files")
        return tomllib.loads(path.read_text())
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _coerce_defaults(
    defaults: Mapping[str, Any] | MutableMapping[str, Any] | Any | None,
) -> tuple[Dict[str, Any], Any | None]:
    """Return a mutable mapping representation and the original dataclass (if any)."""

    if defaults is None:
        return {}, None
    if is_dataclass(defaults):
        data = {field.name: getattr(defaults, field.name) for field in fields(defaults)}
        return data, defaults
    if isinstance(defaults, MutableMapping):
        return dict(defaults), None
    if isinstance(defaults, Mapping):
        return dict(defaults), None
    raise TypeError("defaults must be a mapping or dataclass instance")


def load_runtime_config(
    defaults: Mapping[str, Any] | Any | None = None,
    argv: Iterable[str] | None = None,
) -> Dict[str, Any] | Any:
    """Load runtime configuration from defaults, files and CLI overrides."""

    defaults_dict, defaults_dataclass = _coerce_defaults(defaults)
    argv = list(sys.argv[1:] if argv is None else argv)

    merged: Dict[str, Any] = dict(defaults_dict)
    config_paths = []
    overrides: Dict[str, Any] = {}

    expect_config_path = False
    for arg in argv:
        if expect_config_path:
            config_paths.append(Path(arg))
            expect_config_path = False
            continue
        if arg.startswith("--"):
            if arg.startswith("--config="):
                config_paths.append(Path(arg.split("=", 1)[1]))
                continue
            if "=" not in arg:
                if arg == "--config":
                    expect_config_path = True
                    continue
                config_paths.append(Path(arg[2:]))
                continue
            key, raw_value = arg[2:].split("=", 1)
            overrides[key] = _parse_cli_value(raw_value)
        else:
            config_paths.append(Path(arg))

    for path in config_paths:
        if not path.name:
            continue
        data = _load_config_file(path)
        if not isinstance(data, Mapping):
            raise TypeError(f"Config file {path} must contain a mapping at the top level")
        for key, value in data.items():
            if key not in merged:
                raise KeyError(f"Unknown config key: {key}")
            merged[key] = value

    if expect_config_path:
        raise ValueError("--config flag provided without a following file path")

    torchrun_keys = {"local_rank", "rank", "world_size"}
    for key, value in overrides.items():
        if key in torchrun_keys:
            continue
        if key not in merged:
            raise KeyError(f"Unknown config key: {key}")
        expected = defaults_dict.get(key)
        if expected is not None and not isinstance(value, type(expected)):
            raise TypeError(
                f"Type mismatch for '{key}': expected {type(expected).__name__}, got {type(value).__name__}"
            )
        merged[key] = value

    if defaults_dataclass is not None:
        # ``replace`` preserves dataclass type while applying overrides.
        return replace(defaults_dataclass, **merged)

    return merged

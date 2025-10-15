import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from nanochat.configs.loader import load_runtime_config


def test_returns_defaults_when_no_overrides() -> None:
    defaults = {"foo": 1, "bar": "baz"}
    config = load_runtime_config(defaults=defaults, argv=[])
    assert config == defaults


def test_cli_override_respects_types() -> None:
    defaults = {"foo": 1, "bar": "baz"}
    config = load_runtime_config(defaults=defaults, argv=["--foo=2", "--bar='qux'"])
    assert config["foo"] == 2
    assert config["bar"] == "qux"


def test_unknown_key_raises() -> None:
    defaults = {"foo": 1}
    with pytest.raises(KeyError):
        load_runtime_config(defaults=defaults, argv=["--baz=3"])


def test_config_file_merges(tmp_path: Path) -> None:
    defaults = {"foo": 1, "bar": "baz"}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"foo": 5}))
    config = load_runtime_config(defaults=defaults, argv=[str(config_file)])
    assert config["foo"] == 5
    assert config["bar"] == "baz"


def test_torchrun_arguments_are_ignored() -> None:
    defaults = {"foo": 1}
    config = load_runtime_config(
        defaults=defaults,
        argv=["--foo=3", "--local_rank=0", "--world_size=1"],
    )
    assert config["foo"] == 3


@dataclass
class _ExampleConfig:
    foo: int = 1
    bar: str = "baz"


def test_dataclass_defaults_are_supported() -> None:
    config = load_runtime_config(defaults=_ExampleConfig(), argv=["--foo=5"])
    assert isinstance(config, _ExampleConfig)
    assert config.foo == 5
    assert config.bar == "baz"

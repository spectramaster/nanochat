"""Regression tests for dataset module dependency guards."""

from __future__ import annotations

import pytest

from nanochat.data import dataset


def test_parquets_iter_requires_pyarrow(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "_pyarrow_parquet", None)
    cfg = dataset.DatasetConfig(data_dir=str(tmp_path))
    iterator = dataset.parquets_iter_batched(split="train", config=cfg)
    with pytest.raises(RuntimeError, match="pyarrow"):
        next(iterator)


def test_download_single_file_requires_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "requests", None)
    cfg = dataset.DatasetConfig(data_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="requests"):
        dataset.download_single_file(0, config=cfg)

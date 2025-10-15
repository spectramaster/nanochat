"""Dataset management utilities for NanoChat training workflows."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Iterable, Iterator, List, Sequence

from nanochat.utils.display import print0
from nanochat.utils.paths import get_base_dir

try:  # pragma: no cover - optional dependency wiring
    import pyarrow.parquet as _pyarrow_parquet
except ModuleNotFoundError:  # pragma: no cover - exercised via tests with monkeypatching
    _pyarrow_parquet = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency wiring
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised via tests with monkeypatching
    requests = None  # type: ignore[assignment]

__all__ = [
    "DatasetConfig",
    "DEFAULT_CONFIG",
    "list_parquet_files",
    "parquets_iter_batched",
    "download_single_file",
    "download_dataset",
    "main",
]


def _require_pyarrow():
    if _pyarrow_parquet is None:
        raise RuntimeError(
            "pyarrow is required for parquet streaming; install it via 'pip install pyarrow'."
        )
    return _pyarrow_parquet


def _require_requests():
    if requests is None:
        raise RuntimeError(
            "requests is required for dataset downloads; install it via 'pip install requests'."
        )
    return requests


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing the FineWeb-EDU dataset layout."""

    base_url: str = (
        "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
    )
    max_shard: int = 1822
    data_dir: str | None = None

    def resolve_data_dir(self) -> str:
        directory = self.data_dir or os.path.join(get_base_dir(), "base_data")
        os.makedirs(directory, exist_ok=True)
        return directory

    def index_to_filename(self, index: int) -> str:
        return f"shard_{index:05d}.parquet"


DEFAULT_CONFIG = DatasetConfig()


def list_parquet_files(config: DatasetConfig | None = None, data_dir: str | None = None) -> List[str]:
    """Return sorted parquet file paths for the configured dataset."""

    cfg = config or DEFAULT_CONFIG
    directory = data_dir or cfg.resolve_data_dir()
    parquet_files = sorted(
        f for f in os.listdir(directory) if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(directory, file) for file in parquet_files]


def parquets_iter_batched(
    split: str,
    start: int = 0,
    step: int = 1,
    config: DatasetConfig | None = None,
) -> Iterator[Sequence[str]]:
    """Yield batches of documents from parquet shards.

    Parameters
    ----------
    split:
        Either ``"train"`` or ``"val"``.
    start, step:
        Control sharding in distributed settings (rank/world-size aware).
    config:
        Optional dataset configuration.
    """

    pq = _require_pyarrow()
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(config=config)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


def _download(index: int, config: DatasetConfig) -> bool:
    return download_single_file(index, config=config)


def download_single_file(index: int, config: DatasetConfig | None = None) -> bool:
    """Download a single shard with retry/backoff."""

    requests_mod = _require_requests()
    cfg = config or DEFAULT_CONFIG
    directory = cfg.resolve_data_dir()
    filename = cfg.index_to_filename(index)
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        print0(f"Skipping {filepath} (already exists)")
        return True

    url = f"{cfg.base_url}/{filename}"
    print0(f"Downloading {filename} → {filepath}")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests_mod.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)
            os.replace(temp_path, filepath)
            print0(f"Successfully downloaded {filename}")
            return True
        except (requests_mod.RequestException, OSError) as exc:
            print0(f"Attempt {attempt}/{max_attempts} failed for {filename}: {exc}")
            for path in (filepath + ".tmp", filepath):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                wait_time = 2**attempt
                print0(f"Waiting {wait_time} seconds before retry…")
                time.sleep(wait_time)
            else:
                print0(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False


def download_dataset(
    count: int | None = None,
    workers: int = 4,
    config: DatasetConfig | None = None,
) -> int:
    """Download multiple shards concurrently and return number of successes."""

    _require_requests()
    cfg = config or DEFAULT_CONFIG
    max_count = cfg.max_shard + 1
    target = max_count if count in (None, -1) else min(count, max_count)
    indices = list(range(target))
    print0(f"Downloading {len(indices)} shards into {cfg.resolve_data_dir()}")
    if workers <= 1:
        return sum(download_single_file(index, config=cfg) for index in indices)
    with Pool(processes=workers) as pool:
        results = pool.starmap(_download, [(index, cfg) for index in indices])
    return sum(1 for success in results if success)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1)
    parser.add_argument("-w", "--num-workers", type=int, default=4)
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = DEFAULT_CONFIG
    successful = download_dataset(count=args.num_files, workers=args.num_workers, config=cfg)
    total = cfg.max_shard + 1 if args.num_files in (-1, None) else args.num_files
    print0(f"Done! Downloaded {successful}/{total} shards to {cfg.resolve_data_dir()}")
    return 0 if successful == total else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(main())

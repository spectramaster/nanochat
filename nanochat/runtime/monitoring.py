"""Runtime monitoring utilities for NanoChat."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from nanochat.utils import setup_default_logging

logger = logging.getLogger(__name__)


@dataclass
class MetricEvent:
    name: str
    value: float
    step: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, object] = field(default_factory=dict)


class RuntimeMonitor:
    """Light-weight metric sink that can be extended for external systems."""

    def __init__(self) -> None:
        self._logger = setup_default_logging()

    def log_metric(self, event: MetricEvent) -> None:
        self._logger.info(
            "metric | %s = %.4f (step=%s) %s",
            event.name,
            event.value,
            event.step,
            event.metadata,
        )

    def log_exception(self, err: Exception, context: Optional[Dict[str, object]] = None) -> None:
        self._logger.error("runtime exception: %s | context=%s", err, context)


DEFAULT_MONITOR = RuntimeMonitor()

from abc import ABC, abstractmethod
import pandas as pd
from typing import TypedDict
from enum import Enum
from chain.db.qdrant.qdrant_client_extended import QdrantClientExtended
from chain.db.qdrant.qdrant_db import QdrantMemoryMetrics


class QuantizationType(Enum):
    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"


class QDrantItem(TypedDict):
    uri: str
    chunk_id: str
    content: str
    score: float


class QDrantItemList(TypedDict):
    dense_items: list[QDrantItem]
    sparse_items: list[QDrantItem]


class StrategyBase(ABC):
    NAME = "Base strategy"

    def __init__(self, quantization: QuantizationType = QuantizationType.NONE):
        self._memory_metrics = QdrantMemoryMetrics()
        self._quantization = quantization

    @abstractmethod
    async def execute(
        self, df: pd.DataFrame, qdrant_client: QdrantClientExtended, collection_name: str, *args, **kwargs
    ) -> QDrantItemList:
        pass

    async def get_parameters(self) -> dict:
        ram_usage: str = 'N/A'
        if self.memory_metrics is not None and self.memory_metrics.memory_resident_bytes is not None:
            bytes_used_in_ram = self.memory_metrics.memory_resident_bytes
            ram_usage = f"{bytes_used_in_ram / (1024**2):.2f} MB"
        return {
            'name': self.NAME,
            'real_ram_usage (bytes)': ram_usage,
            'reranker_items': 100,
            'quantization': self._quantization.name,
        }

    @property
    def memory_metrics(self) -> QdrantMemoryMetrics:
        return self._memory_metrics

    @memory_metrics.setter
    def memory_metrics(self, metrics: QdrantMemoryMetrics) -> None:
        self._memory_metrics = metrics

    @property
    def quantization(self) -> QuantizationType:
        return self._quantization

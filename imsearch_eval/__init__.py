"""Benchmarking framework for vector databases and models."""

__version__ = "0.1.0"

# Try to import adapters, but don't fail if dependencies are missing
_AVAILABLE_ADAPTERS = {}

try:
    from .adapters.triton import TritonModelProvider, TritonModelUtils
    _AVAILABLE_ADAPTERS['triton'] = ['TritonModelProvider', 'TritonModelUtils']
except ImportError:
    TritonModelProvider = None
    TritonModelUtils = None

try:
    from .adapters.weaviate import WeaviateAdapter, WeaviateQuery
    _AVAILABLE_ADAPTERS['weaviate'] = ['WeaviateAdapter', 'WeaviateQuery']
except ImportError:
    WeaviateAdapter = None
    WeaviateQuery = None

try:
    from .adapters.milvus import MilvusAdapter, MilvusQuery
    _AVAILABLE_ADAPTERS['milvus'] = ['MilvusAdapter', 'MilvusQuery']
except ImportError:
    MilvusAdapter = None
    MilvusQuery = None

try:
    from .adapters.huggingface import HuggingFaceDataset
    _AVAILABLE_ADAPTERS['huggingface'] = ['HuggingFaceDataset']
except ImportError:
    HuggingFaceDataset = None

# Always import framework (core dependencies)
from .framework import (
    BenchmarkEvaluator, VectorDBAdapter, ModelProvider, QueryResult,
    BenchmarkDataset, DataLoader, Config, Query, BatchedIterator
)

# Build __all__ based on what's available
__all__ = [
    'BenchmarkEvaluator', 'VectorDBAdapter', 'ModelProvider', 'QueryResult',
    'BenchmarkDataset', 'DataLoader', 'Config', 'Query', 'BatchedIterator'
]

if TritonModelProvider is not None:
    __all__.extend(['TritonModelProvider', 'TritonModelUtils'])
if WeaviateAdapter is not None:
    __all__.extend(['WeaviateAdapter', 'WeaviateQuery'])
if MilvusAdapter is not None:
    __all__.extend(['MilvusAdapter', 'MilvusQuery'])
if HuggingFaceDataset is not None:
    __all__.extend(['HuggingFaceDataset'])

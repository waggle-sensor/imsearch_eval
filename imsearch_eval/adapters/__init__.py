"""Shared adapters for vector databases and models."""

# Try to import adapters, but don't fail if dependencies are missing
_AVAILABLE_ADAPTERS = {}

try:
    from .triton import TritonModelProvider, TritonModelUtils
    _AVAILABLE_ADAPTERS['triton'] = ['TritonModelProvider', 'TritonModelUtils']
except ImportError:
    TritonModelProvider = None
    TritonModelUtils = None

try:
    from .weaviate import WeaviateAdapter, WeaviateQuery
    _AVAILABLE_ADAPTERS['weaviate'] = ['WeaviateAdapter', 'WeaviateQuery']
except ImportError:
    WeaviateAdapter = None
    WeaviateQuery = None

try:
    from .milvus import MilvusAdapter, MilvusQuery
    _AVAILABLE_ADAPTERS['milvus'] = ['MilvusAdapter', 'MilvusQuery']
except ImportError:
    MilvusAdapter = None
    MilvusQuery = None

try:
    from .huggingface import HuggingFaceDataset
    _AVAILABLE_ADAPTERS['huggingface'] = ['HuggingFaceDataset']
except ImportError:
    HuggingFaceDataset = None

# Build __all__ based on what's available
__all__ = []
if TritonModelProvider is not None:
    __all__.extend(['TritonModelProvider', 'TritonModelUtils'])
if WeaviateAdapter is not None:
    __all__.extend(['WeaviateAdapter', 'WeaviateQuery'])
if MilvusAdapter is not None:
    __all__.extend(['MilvusAdapter', 'MilvusQuery'])
if HuggingFaceDataset is not None:
    __all__.extend(['HuggingFaceDataset'])

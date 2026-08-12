"""Helpers for building Milvus benchmark collection schemas.

Mirrors the production dual-dense + BM25 schema used by Sage Image Search
(weavmanage/migrations/002_split_dense_vectors.py).
"""

import re
from typing import Any, Dict, List, Optional, Union

try:
    from pymilvus import DataType, FunctionType
    _MILVUS_AVAILABLE = True
except ImportError:
    DataType = None
    FunctionType = None
    _MILVUS_AVAILABLE = False

DENSE_VECTOR_FIELDS = ("caption_vector", "image_vector")
SPARSE_VECTOR_FIELD = "sparse"
SEARCH_TEXT_FIELD = "search_text"
PROTECTED_COLLECTIONS = frozenset({"SageImageSearch", "SageImageSearchDev"})

_POINT_RE = re.compile(
    r"POINT\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)",
    re.IGNORECASE,
)

_DATATYPE_ALIASES = {
    "INT64": "INT64",
    "INT": "INT64",
    "INTEGER": "INT64",
    "FLOAT": "FLOAT",
    "DOUBLE": "DOUBLE",
    "NUMBER": "FLOAT",
    "BOOL": "BOOL",
    "BOOLEAN": "BOOL",
    "VARCHAR": "VARCHAR",
    "TEXT": "VARCHAR",
    "STRING": "VARCHAR",
    "JSON": "JSON",
    "TIMESTAMPTZ": "TIMESTAMPTZ",
    "GEOMETRY": "GEOMETRY",
}


def _check_milvus_available():
    if not _MILVUS_AVAILABLE:
        raise ImportError(
            "Milvus schema helpers require 'pymilvus'. "
            "Install it with: pip install imsearch_eval[milvus]"
        )


def _resolve_datatype(datatype: Union[str, Any]) -> Any:
    """Map a string or DataType enum to a pymilvus DataType."""
    _check_milvus_available()
    if not isinstance(datatype, str):
        return datatype
    key = datatype.strip().upper()
    attr = _DATATYPE_ALIASES.get(key, key)
    resolved = getattr(DataType, attr, None)
    if resolved is None:
        raise ValueError(f"Unknown Milvus DataType: {datatype}")
    return resolved


def parse_wkt_point(wkt) -> tuple:
    """Return (lat, lon) from a Milvus GEOMETRY WKT POINT. Invalid → (0.0, 0.0)."""
    if not wkt:
        return 0.0, 0.0
    match = _POINT_RE.search(str(wkt))
    if not match:
        return 0.0, 0.0
    lon, lat = float(match.group(1)), float(match.group(2))
    return lat, lon


def to_milvus_wkt_point(lon: float, lat: float) -> str:
    """Build WKT POINT(lon lat) for DataType.GEOMETRY."""
    return f"POINT({float(lon)} {float(lat)})"


def to_milvus_timestamptz(value: str) -> str:
    """Normalize an ISO-ish timestamp string to UTC with a Z suffix."""
    if not value:
        return ""
    text = str(value).strip().replace(" ", "T")
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    if not text.endswith("Z") and "+" not in text[10:] and text.count("-") <= 2:
        return text + "Z" if "T" in text else text
    return text


def build_benchmark_schema(
    name: str,
    scalar_fields: List[Dict[str, Any]],
    vector_dim: int = 1024,
    include_location: bool = False,
    include_timestamp: bool = False,
    timestamp_field: str = "timestamp",
    location_field: str = "location",
    hnsw_M: int = 16,
    hnsw_ef_construction: int = 256,
    search_text_max_length: int = 65535,
    analyzer_params: Optional[Dict[str, Any]] = None,
    enable_dynamic_field: bool = False,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
    bm25_inverted_index_algo: str = "DAAT_MAXSCORE",
) -> Dict[str, Any]:
    """
    Build a MilvusAdapter.create_collection schema_config.

    Always includes:
      - id (INT64 PK, auto_id)
      - caption_vector / image_vector (FLOAT_VECTOR, HNSW COSINE)
      - search_text (VARCHAR + analyzer) → BM25 sparse
      - sparse (SPARSE_FLOAT_VECTOR, filled by BM25 function)

    ``scalar_fields`` items are add_field kwargs, e.g.:
      {"field_name": "caption", "datatype": "VARCHAR", "max_length": 65535}
      {"field_name": "relevant", "datatype": "INT64"}
    """
    _check_milvus_available()
    if analyzer_params is None:
        analyzer_params = {"type": "standard"}

    fields: List[Dict[str, Any]] = [
        {
            "field_name": "id",
            "datatype": DataType.INT64,
            "is_primary": True,
            "auto_id": True,
        },
        {
            "field_name": "caption_vector",
            "datatype": DataType.FLOAT_VECTOR,
            "dim": vector_dim,
        },
        {
            "field_name": "image_vector",
            "datatype": DataType.FLOAT_VECTOR,
            "dim": vector_dim,
        },
        {
            "field_name": "search_text",
            "datatype": DataType.VARCHAR,
            "max_length": search_text_max_length,
            "enable_analyzer": True,
            "analyzer_params": analyzer_params,
        },
        {
            "field_name": "sparse",
            "datatype": DataType.SPARSE_FLOAT_VECTOR,
        },
    ]

    reserved = {
        "id",
        "caption_vector",
        "image_vector",
        SEARCH_TEXT_FIELD,
        SPARSE_VECTOR_FIELD,
        location_field if include_location else None,
        timestamp_field if include_timestamp else None,
    }
    reserved.discard(None)

    for field in scalar_fields:
        field_config = dict(field)
        if "field_name" not in field_config:
            raise ValueError(f"scalar field missing field_name: {field}")
        if field_config["field_name"] in reserved:
            continue
        if "datatype" in field_config:
            field_config["datatype"] = _resolve_datatype(field_config["datatype"])
        if field_config.get("datatype") == DataType.VARCHAR and "max_length" not in field_config:
            field_config["max_length"] = 2048
        fields.append(field_config)

    if include_timestamp:
        fields.append(
            {
                "field_name": timestamp_field,
                "datatype": DataType.TIMESTAMPTZ,
            }
        )
    if include_location:
        fields.append(
            {
                "field_name": location_field,
                "datatype": DataType.GEOMETRY,
                "nullable": True,
            }
        )

    index: List[Dict[str, Any]] = [
        {
            "field_name": "caption_vector",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": hnsw_M, "efConstruction": hnsw_ef_construction},
        },
        {
            "field_name": "image_vector",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": hnsw_M, "efConstruction": hnsw_ef_construction},
        },
        {
            "field_name": "sparse",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": bm25_inverted_index_algo,
                "bm25_k1": bm25_k1,
                "bm25_b": bm25_b,
            },
        },
    ]
    if include_timestamp:
        index.append({"field_name": timestamp_field, "index_type": "STL_SORT"})
    if include_location:
        index.append({"field_name": location_field, "index_type": "RTREE"})

    functions = [
        {
            "name": "search_text_bm25",
            "function_type": FunctionType.BM25,
            "input_field_names": [SEARCH_TEXT_FIELD],
            "output_field_names": [SPARSE_VECTOR_FIELD],
        }
    ]

    return {
        "name": name,
        "auto_id": True,
        "enable_dynamic_field": enable_dynamic_field,
        "fields": fields,
        "index": index,
        "functions": functions,
    }

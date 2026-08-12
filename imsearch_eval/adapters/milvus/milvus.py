"""
Milvus-based adapters for benchmarking framework.

This module provides all Milvus-related adapters:
- MilvusQuery: Query class for Milvus with dual-dense + BM25 hybrid search
- MilvusAdapter: VectorDBAdapter implementation for Milvus
"""

import os
import logging
import time
from io import BytesIO
from itertools import islice
from typing import List, Dict, Any, Optional, Callable
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import base64

import pandas as pd
import numpy as np
from PIL import Image

try:
    from pymilvus import (
        AnnSearchRequest,
        WeightedRanker,
        Function,
        FunctionType,
        MilvusClient,
    )
    _MILVUS_AVAILABLE = True
except ImportError:
    AnnSearchRequest = WeightedRanker = Function = FunctionType = MilvusClient = None
    _MILVUS_AVAILABLE = False

try:
    from ..triton import TritonModelUtils
    _TRITON_AVAILABLE = True
except ImportError:
    TritonModelUtils = None
    _TRITON_AVAILABLE = False

from ...framework.interfaces import VectorDBAdapter, QueryResult, Query
from ...framework.model_utils import ModelUtils
from .schema import (
    DENSE_VECTOR_FIELDS,
    PROTECTED_COLLECTIONS,
    SPARSE_VECTOR_FIELD,
    parse_wkt_point,
)


def _check_milvus_available():
    """Check if Milvus dependencies are installed."""
    if not _MILVUS_AVAILABLE:
        raise ImportError(
            "Milvus adapters require 'pymilvus'. "
            "Install it with: pip install imsearch_eval[milvus]"
        )


def _batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def _to_list(embedding) -> list:
    if isinstance(embedding, np.ndarray):
        return embedding.tolist()
    return list(embedding)


class MilvusQuery(Query):
    """
    Query class for Milvus that provides dual-dense + BM25 hybrid search
    matching Sage Image Search production (app/query.py).
    """

    def __init__(
        self,
        milvus_client,
        collection_name: str = None,
        triton_client=None,
        model_utils: ModelUtils = None,
    ):
        """
        Initialize Milvus query instance.

        Args:
            milvus_client: Milvus client
            collection_name: Name of the collection to query
            triton_client: Optional Triton client for generating embeddings
            model_utils: Optional ModelUtils instance (if None and triton_client
                provided, creates TritonModelUtils)
        """
        _check_milvus_available()
        self.milvus_client = milvus_client
        self.collection_name = collection_name
        self.triton_client = triton_client

        if model_utils is None and triton_client is not None:
            if not _TRITON_AVAILABLE:
                raise ImportError(
                    "TritonModelUtils is required but not available. "
                    "Install with: pip install imsearch_eval[triton]"
                )
            self.model_utils = TritonModelUtils(triton_client)
        else:
            self.model_utils = model_utils

        self._output_fields_cache: Dict[str, List[str]] = {}

    def query(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str,
        limit: int = 25,
        query_method: Callable = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform a search query on Milvus.

        Routes to a specific query method. Default is clip_hybrid_query_dual_index.
        """
        if query_method is None:
            query_method = self.clip_hybrid_query_dual_index
        elif isinstance(query_method, str):
            query_method = getattr(self, query_method)

        return query_method(near_text, collection_name, target_vector, limit, **kwargs)

    def _scalar_output_fields(self, collection_name: str) -> List[str]:
        """Return non-vector field names for a collection (cached)."""
        if collection_name in self._output_fields_cache:
            return self._output_fields_cache[collection_name]

        skip = set(DENSE_VECTOR_FIELDS) | {SPARSE_VECTOR_FIELD, "id"}
        names: List[str] = []
        try:
            desc = self.milvus_client.describe_collection(collection_name)
            for field in desc.get("fields", []):
                name = field.get("name")
                if name and name not in skip:
                    names.append(name)
        except Exception as e:
            logging.warning(f"Could not describe collection '{collection_name}': {e}")

        self._output_fields_cache[collection_name] = names
        return names

    def _load_image(self, link: str) -> Optional[Image.Image]:
        """Load an image from a local path or HTTP(S) URL."""
        if not link:
            return None
        try:
            if link.startswith("http://") or link.startswith("https://"):
                headers = {}
                user = os.environ.get("SAGE_USER")
                password = os.environ.get("SAGE_PASS")
                if user and password:
                    token = base64.b64encode(f"{user}:{password}".encode()).decode()
                    headers["Authorization"] = f"Basic {token}"
                request = Request(link, headers=headers)
                with urlopen(request, timeout=30) as response:
                    image = Image.open(BytesIO(response.read()))
            else:
                image = Image.open(link)
            return image.convert("RGB")
        except (HTTPError, URLError, OSError, ValueError) as e:
            logging.debug(f"Failed to load image from {link}: {e}")
            return None
        except Exception as e:
            logging.debug(f"Failed to load image from {link}: {e}")
            return None

    def _extract_hit(
        self,
        hit,
        vector_field: Optional[str] = None,
    ) -> dict:
        """Flatten a Milvus hit into a result dict."""
        entity = hit.get("entity", hit) if hasattr(hit, "get") else getattr(hit, "entity", {})
        if entity is None:
            entity = {}
        if not isinstance(entity, dict):
            try:
                entity = dict(entity)
            except Exception:
                entity = {}

        result = {}
        for key, value in entity.items():
            if key == vector_field:
                result["vector"] = value
                continue
            if key == "location":
                lat, lon = parse_wkt_point(value)
                result["location"] = value
                result["location_lat"] = lat
                result["location_lon"] = lon
                continue
            result[key] = value

        hit_id = hit.get("id") if hasattr(hit, "get") else getattr(hit, "id", "")
        distance = hit.get("distance") if hasattr(hit, "get") else getattr(hit, "distance", 0.0)
        result["uuid"] = str(hit_id if hit_id is not None else "")
        result["score"] = float(distance or 0.0)
        return result

    def _rerank_hits(self, objects: List[dict], near_text: str) -> List[dict]:
        """CLIP image-text rerank using the image at each hit's link field."""
        if not self.model_utils or not hasattr(self.model_utils, "clip_image_text_score"):
            for obj in objects:
                obj["rerank_score"] = 0.0
            return objects

        for obj in objects:
            image = self._load_image(obj.get("link") or "")
            if image is None:
                obj["rerank_score"] = 0.0
            else:
                obj["rerank_score"] = float(
                    self.model_utils.clip_image_text_score(near_text, image)
                )
        objects.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return objects

    def vector_query(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str,
        limit: int = 25,
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        rerank: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """Perform a pure dense vector search."""
        if search_params is None:
            search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        if not self.model_utils:
            raise ValueError("Model utils is required for vector queries with text input")

        embedding = self.model_utils.get_clip_embeddings(near_text, image=None)
        if embedding is None:
            logging.error("Failed to generate embedding")
            return pd.DataFrame()

        fields = list(output_fields) if output_fields is not None else self._scalar_output_fields(collection_name)
        if target_vector not in fields:
            fields = fields + [target_vector]

        results = self.milvus_client.search(
            collection_name=collection_name,
            anns_field=target_vector,
            data=[_to_list(embedding)],
            limit=limit,
            search_params=search_params,
            output_fields=fields,
        )

        objects = []
        hit_list = results[0] if results else []
        for hit in hit_list:
            objects.append(self._extract_hit(hit, vector_field=target_vector))

        if rerank:
            objects = self._rerank_hits(objects, near_text)
        else:
            for obj in objects:
                obj.setdefault("rerank_score", 0.0)

        return pd.DataFrame(objects)

    def clip_hybrid_query_dual_index(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str = "image_vector",
        limit: int = 25,
        query_alpha: float = 0.4,
        clip_alpha: float = 0.7,
        enable_image_vector: bool = True,
        enable_caption_vector: bool = True,
        enable_bm25: bool = True,
        image_vector_field: str = "image_vector",
        caption_vector_field: str = "caption_vector",
        target_sparse_vector: str = "sparse",
        output_fields: Optional[List[str]] = None,
        rerank: bool = True,
        dense_search_params: Optional[dict] = None,
        sparse_search_params: Optional[dict] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Hybrid CLIP image_vector + caption_vector + BM25 sparse search,
        then optional Triton CLIP rerank (query text vs retrieved image).

        Dense vs sparse uses ``query_alpha``. Within dense, ``clip_alpha``
        weights ``image_vector`` vs ``caption_vector``. Ablation flags omit
        legs instead of sending a zero weight.
        """
        if not self.model_utils:
            raise ValueError("Model utils required for CLIP hybrid query")

        if alpha is not None:
            query_alpha = alpha

        dense_embedding = self.model_utils.get_clip_embeddings(near_text, image=None)
        if dense_embedding is None:
            logging.error("Failed to generate CLIP embedding")
            return pd.DataFrame()
        vector = _to_list(dense_embedding)

        dense_search_params = dense_search_params or {
            "metric_type": "COSINE",
            "params": {"ef": 64},
        }
        sparse_search_params = sparse_search_params or {"metric_type": "BM25"}

        reqs = []
        weights = []
        single_search = None
        if enable_image_vector:
            reqs.append(
                AnnSearchRequest(
                    data=[vector],
                    anns_field=image_vector_field,
                    param=dense_search_params,
                    limit=limit,
                )
            )
            weights.append(query_alpha * clip_alpha)
            single_search = {
                "data": [vector],
                "anns_field": image_vector_field,
                "search_params": dense_search_params,
            }
        if enable_caption_vector:
            reqs.append(
                AnnSearchRequest(
                    data=[vector],
                    anns_field=caption_vector_field,
                    param=dense_search_params,
                    limit=limit,
                )
            )
            weights.append(query_alpha * (1.0 - clip_alpha))
            if single_search is None:
                single_search = {
                    "data": [vector],
                    "anns_field": caption_vector_field,
                    "search_params": dense_search_params,
                }
        if enable_bm25:
            reqs.append(
                AnnSearchRequest(
                    data=[near_text],
                    anns_field=target_sparse_vector,
                    param=sparse_search_params,
                    limit=limit,
                )
            )
            weights.append(1.0 - query_alpha)
            if single_search is None:
                single_search = {
                    "data": [near_text],
                    "anns_field": target_sparse_vector,
                    "search_params": sparse_search_params,
                }

        if not reqs:
            raise ValueError(
                "At least one of enable_image_vector, enable_caption_vector, "
                "or enable_bm25 must be true"
            )

        diversity_field = (
            image_vector_field if enable_image_vector else caption_vector_field
        )
        fields = list(output_fields) if output_fields is not None else self._scalar_output_fields(collection_name)
        if diversity_field not in fields:
            fields = fields + [diversity_field]

        if len(reqs) == 1:
            results = self.milvus_client.search(
                collection_name=collection_name,
                data=single_search["data"],
                anns_field=single_search["anns_field"],
                search_params=single_search["search_params"],
                limit=limit,
                output_fields=fields,
            )
        else:
            results = self.milvus_client.hybrid_search(
                collection_name=collection_name,
                reqs=reqs,
                ranker=WeightedRanker(*weights),
                limit=limit,
                output_fields=fields,
            )

        hit_list = results[0] if results else []
        objects = [self._extract_hit(hit, vector_field=diversity_field) for hit in hit_list]

        if rerank:
            objects = self._rerank_hits(objects, near_text)
        else:
            for obj in objects:
                obj.setdefault("rerank_score", 0.0)

        return pd.DataFrame(objects)

    def clip_hybrid_query(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str = "image_vector",
        limit: int = 25,
        **kwargs,
    ) -> pd.DataFrame:
        """Alias for clip_hybrid_query_dual_index (production dual-dense hybrid)."""
        return self.clip_hybrid_query_dual_index(
            near_text,
            collection_name,
            target_vector,
            limit,
            **kwargs,
        )


class MilvusAdapter(VectorDBAdapter):
    """Milvus adapter using framework MilvusQuery implementation."""

    @classmethod
    def init_client(cls, **kwargs):
        """
        Initialize and return a Milvus client.

        Args:
            **kwargs: Connection parameters:
                - uri: Milvus URI (default: MILVUS_URI or "http://localhost:19530")
                - user: Milvus user (default: MILVUS_USER)
                - token: Milvus token (default: MILVUS_TOKEN)
                - password: Milvus password (optional)
                - db_name: Database name (default: MILVUS_DB, then MILVUS_DB_NAME)
        """
        _check_milvus_available()
        uri = kwargs.get("uri", os.getenv("MILVUS_URI", "http://localhost:19530"))
        user = kwargs.get("user", os.getenv("MILVUS_USER", ""))
        token = kwargs.get("token", os.getenv("MILVUS_TOKEN", ""))
        password = kwargs.get("password", os.getenv("MILVUS_PASSWORD", None))
        db_name = kwargs.get(
            "db_name",
            os.getenv("MILVUS_DB") or os.getenv("MILVUS_DB_NAME", ""),
        )

        logging.debug(f"Attempting to connect to Milvus at {uri}")

        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            try:
                milvus_client = MilvusClient(
                    uri=uri,
                    user=user,
                    password=password,
                    db_name=db_name,
                    token=token,
                )
                milvus_client.list_collections()
                logging.debug("Successfully connected to Milvus")
                return milvus_client
            except Exception as e:
                retry_count += 1
                logging.error(
                    f"Failed to connect to Milvus (attempt {retry_count}/{max_retries}): {e}"
                )
                if retry_count < max_retries:
                    logging.debug("Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    raise

    def __init__(
        self,
        milvus_client=None,
        collection_name: str = None,
        triton_client=None,
        query_instance: Query = None,
        query_class=None,
        **client_kwargs
    ):
        """
        Initialize Milvus adapter.

        Args:
            milvus_client: Pre-initialized Milvus client (optional)
            collection_name: Default collection name to use for queries
            triton_client: Pre-initialized Triton client (optional)
            query_instance: Pre-initialized Query instance (optional)
            query_class: Query class to use if query_instance is None
            **client_kwargs: Passed to init_client if milvus_client is None
        """
        _check_milvus_available()
        if milvus_client is None:
            milvus_client = self.init_client(**client_kwargs)

        self.milvus_client = milvus_client
        self.default_collection_name = collection_name
        self.triton_client = triton_client

        if query_instance is None:
            if query_class is None:
                query_class = MilvusQuery
            query_instance = query_class(
                milvus_client,
                collection_name,
                triton_client,
            )

        self.query_instance = query_instance

    def search(
        self,
        query: str,
        collection_name: str,
        target_vector: str,
        limit: int = 25,
        query_method: Callable = None,
        **kwargs
    ) -> QueryResult:
        """Perform a search query on Milvus."""
        df = self.query_instance.query(
            near_text=query,
            collection_name=collection_name,
            target_vector=target_vector,
            limit=limit,
            query_method=query_method,
            **kwargs
        )
        results = df.to_dict("records") if not df.empty else []
        return QueryResult(results)

    def create_collection(
        self,
        schema_config: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Create a Milvus collection.

        Args:
            schema_config: Dictionary containing schema configuration:
                - name: Collection name
                - auto_id: Whether to auto-generate IDs
                - enable_dynamic_field: Whether to enable dynamic fields
                - fields: List of field definitions (add_field kwargs)
                - functions: Optional list of Function kwargs
                - index: Optional list of add_index kwargs
        """
        collection_name = schema_config.get("name")
        try:
            if "name" not in schema_config:
                raise ValueError("Collection name is required in schema_config")
            if "fields" not in schema_config:
                raise ValueError("Fields are required in schema_config")
            collection_name = schema_config["name"]

            if collection_name in PROTECTED_COLLECTIONS:
                raise ValueError(
                    f"Refusing to drop protected production collection '{collection_name}'"
                )

            if self.milvus_client.has_collection(collection_name):
                logging.debug(f"Collection '{collection_name}' exists. Deleting it first...")
                self.milvus_client.drop_collection(collection_name)
                while self.milvus_client.has_collection(collection_name):
                    time.sleep(1)

            fields_config = schema_config.get("fields", [])
            index_config = schema_config.get("index", [])
            functions_config = schema_config.get("functions", [])

            schema = self.milvus_client.create_schema(
                auto_id=schema_config.get("auto_id", False),
                enable_dynamic_field=schema_config.get("enable_dynamic_field", False),
            )

            for field_config in fields_config:
                try:
                    schema.add_field(**field_config)
                    logging.debug(
                        f"Added field '{field_config.get('field_name')}' "
                        f"to collection '{collection_name}'"
                    )
                except Exception as e:
                    logging.error(f"Failed to add field: {e}")
                    raise

            index_params = None
            if index_config:
                index_params = self.milvus_client.prepare_index_params()
                for index_item in index_config:
                    try:
                        index_params.add_index(**index_item)
                        logging.debug(
                            f"Added index for field '{index_item.get('field_name')}' "
                            f"to collection '{collection_name}'"
                        )
                    except Exception as e:
                        logging.error(f"Failed to add index: {e}")
                        raise

            for func_config in functions_config:
                try:
                    cfg = dict(func_config)
                    ft = cfg.get("function_type")
                    if isinstance(ft, str):
                        cfg["function_type"] = getattr(FunctionType, ft)
                    function = Function(**cfg)
                    schema.add_function(function)
                    logging.debug(
                        f"Added function '{cfg.get('name')}' to collection '{collection_name}'"
                    )
                except Exception as e:
                    logging.error(f"Failed to add function: {e}")
                    raise

            self.milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            self.milvus_client.load_collection(collection_name)

            logging.debug(f"Collection '{collection_name}' successfully created.")
            return True

        except Exception as e:
            logging.error(f"Error creating collection '{collection_name}': {e}")
            return False

    def delete_collection(
        self,
        collection_name: str,
        **kwargs
    ) -> bool:
        """Delete a Milvus collection."""
        try:
            if collection_name in PROTECTED_COLLECTIONS:
                raise ValueError(
                    f"Refusing to drop protected production collection '{collection_name}'"
                )
            if self.milvus_client.has_collection(collection_name):
                self.milvus_client.drop_collection(collection_name)
                logging.debug(f"Collection '{collection_name}' deleted.")
                return True
            logging.debug(f"Collection '{collection_name}' does not exist.")
            return False
        except Exception as e:
            logging.error(f"Error deleting collection '{collection_name}': {e}")
            return False

    def insert_data(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
        batch_size: int = 100,
        **kwargs
    ) -> int:
        """
        Insert data into a Milvus collection.

        Each item is a flat dict keyed by field name. The BM25 ``sparse``
        field is dropped if present (it is generated by the schema function).
        """
        try:
            total_inserted = 0
            for batch in _batched(data, batch_size):
                rows = []
                for item in batch:
                    if item is None:
                        continue
                    row = {k: v for k, v in item.items() if k != SPARSE_VECTOR_FIELD}
                    rows.append(row)
                if not rows:
                    continue
                res = self.milvus_client.insert(
                    collection_name=collection_name,
                    data=rows,
                )
                if isinstance(res, dict):
                    count = res.get("insert_count", len(rows))
                else:
                    count = getattr(res, "insert_count", len(rows))
                if isinstance(count, int):
                    total_inserted += count
                else:
                    total_inserted += len(rows)

            try:
                self.milvus_client.flush(collection_name)
            except Exception as e:
                logging.debug(f"Flush after insert failed (continuing): {e}")

            logging.debug(f"Inserted {total_inserted} items into '{collection_name}'.")
            return total_inserted
        except Exception as e:
            logging.error(f"Error inserting data into '{collection_name}': {e}")
            return 0

    def close(self):
        """Close the Milvus connection."""
        if self.milvus_client:
            self.milvus_client.close()

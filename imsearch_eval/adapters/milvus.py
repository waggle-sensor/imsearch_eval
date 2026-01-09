"""
Milvus-based adapters for benchmarking framework.

This module provides all Milvus-related adapters:
- MilvusQuery: Query class for Milvus with various search methods
- MilvusAdapter: VectorDBAdapter implementation for Milvus
"""

import os
import logging
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Check for optional dependencies
try:
    from pymilvus import (
        DataType,
        AnnSearchRequest,
        WeightedRanker,
        Function,
        MilvusClient
    )
    # Check if SPARSE_FLOAT_VECTOR is available (Milvus 2.3+)
    try:
        SPARSE_FLOAT_VECTOR = DataType.SPARSE_FLOAT_VECTOR
    except AttributeError:
        SPARSE_FLOAT_VECTOR = None
    _MILVUS_AVAILABLE = True
except ImportError:
    DataType = AnnSearchRequest = WeightedRanker = Function = MilvusClient = None
    SPARSE_FLOAT_VECTOR = None
    _MILVUS_AVAILABLE = False

# Try to import TritonModelUtils (optional, but needed for some query methods)
try:
    from .triton import TritonModelUtils
    _TRITON_AVAILABLE = True
except ImportError:
    TritonModelUtils = None
    _TRITON_AVAILABLE = False

from ..framework.interfaces import VectorDBAdapter, QueryResult, Query


def _check_milvus_available():
    """Check if Milvus dependencies are installed."""
    if not _MILVUS_AVAILABLE:
        raise ImportError(
            "Milvus adapters require 'pymilvus'. "
            "Install it with: pip install pymilvus"
        )

class MilvusQuery(Query):
    """
    Query class for Milvus that provides various search methods.
    """
    
    def __init__(self, milvus_client, collection_name: str, triton_client=None, model_utils=None):
        """
        Initialize Milvus query instance.
        
        Args:
            milvus_client: Milvus client
            collection_name: Name of the collection to query
            triton_client: Optional Triton client for generating embeddings
            model_utils: Optional ModelUtils instance (if None and triton_client provided, creates TritonModelUtils)
        """
        _check_milvus_available()
        self.milvus_client = milvus_client
        self.collection_name = collection_name
        
        self.triton_client = triton_client
        
        # Create model_utils if triton_client is provided but model_utils is not
        if model_utils is None and triton_client is not None:
            if not _TRITON_AVAILABLE:
                raise ImportError(
                    "TritonModelUtils is required but not available. "
                    "Install with: pip install imsearch_eval[weaviate]"
                )
            self.model_utils = TritonModelUtils(triton_client)
        else:
            self.model_utils = model_utils
    
    def query(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str,
        limit: int = 25,
        query_method: str = "clip_hybrid_query",
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform a search query on Milvus.
        
        This is the generic query method that routes to specific query methods
        based on the query_method parameter.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search (ignored, uses instance collection_name)
            target_vector: Name of the vector field to search in
            limit: Maximum number of results to return
            query_method: Method name to call (e.g., "clip_hybrid_query", "vector_query")
            **kwargs: Additional search parameters passed to the specific query method
        
        Returns:
            DataFrame with search results
        """
        # Route to the appropriate query method
        if query_method == "clip_hybrid_query":
            return self.clip_hybrid_query(near_text, collection_name, target_vector, limit, **kwargs)
        elif query_method == "vector_query":
            return self.vector_query(near_text, collection_name, target_vector, limit, **kwargs)
        else:
            # Default to clip_hybrid_query if method not recognized
            logging.warning(f"Unknown query method '{query_method}', defaulting to 'clip_hybrid_query'")
            return self.clip_hybrid_query(near_text, collection_name, target_vector, limit, **kwargs)
    
    def get_location_coordinate(self, obj, coordinate_type: str) -> float:
        """
        Helper function to safely fetch latitude or longitude from the location property.
        
        Args:
            obj: Result object from Milvus
            coordinate_type: "latitude" or "longitude"
        
        Returns:
            Coordinate value as float, or 0.0 if not available
        """
        location = obj.get("location", "")
        if location:
            try:
                if isinstance(location, dict):
                    return float(location.get(coordinate_type, 0.0))
                elif isinstance(location, str):
                    # Try to parse if it's a string representation
                    import json
                    loc_dict = json.loads(location)
                    return float(loc_dict.get(coordinate_type, 0.0))
            except (AttributeError, ValueError, TypeError, json.JSONDecodeError):
                logging.warning(f"Invalid {coordinate_type} value found")
        return 0.0
    
    def _extract_object_data(self, hit) -> dict:
        """
        Extract object data from Milvus result.
        
        Args:
            hit: Hit object from Milvus search result
        
        Returns:
            Dictionary with all properties and metadata extracted.
        """
        result = {}
        entity_data = getattr(hit, "entity", {})
        
        # Extract all entity data fields dynamically
        for key, value in entity_data.items():
            # Skip location as it's handled separately
            if key != "location":
                result[key] = value
        
        # Extract metadata from hit object dynamically
        if hit:
            for key, value in hit.items():
                result[key] = value
        
        # Handle location coordinates if present
        if "location" in entity_data:
            location_lat = self.get_location_coordinate(entity_data, "latitude")
            location_lon = self.get_location_coordinate(entity_data, "longitude")
            if location_lat != 0.0 or location_lon != 0.0:
                result["location_lat"] = location_lat
                result["location_lon"] = location_lon
        
        return result
    
    def vector_query(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str,
        limit: int = 25,
        search_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform a pure vector search.
        
        Args:
            near_text: Text query (will be converted to embedding if model_utils available)
            collection_name: Name of the collection to search (ignored, uses instance collection_name)
            target_vector: Name of the vector field to search in
            limit: Maximum number of results to return
            search_params: Optional search parameters for Milvus (e.g., {"metric_type": "COSINE", "params": {"nprobe": 10}})
            **kwargs: Additional search parameters
        
        Returns:
            DataFrame with search results
        """
        # Default search parameters
        if search_params is None:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        # Get embedding if model_utils is available
        if self.model_utils:
            embedding = self.model_utils.get_clip_embeddings(near_text, image=None)
            if embedding is None:
                logging.error("Failed to generate embedding")
                return pd.DataFrame()
        else:
            raise ValueError("Model utils is required for vector queries with text input")
        
        # Ensure embedding is a list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        # Perform vector search        
        results = self.milvus_client.search(
            collection_name=collection_name,
            anns_field=target_vector,
            data=[embedding],
            limit=limit,
            search_params=search_params,
        )
        
        # Extract results - in 2.6.x, results is a list of SearchResult objects
        objects = []
        for result in results:
            for hit in result:             
                obj_data = self._extract_object_data(hit)
                objects.append(obj_data)
        
        return pd.DataFrame(objects)
    
    def clip_hybrid_query(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str,
        target_sparse_vector: str,
        limit: int = 25,
        alpha: float = 0.4,
        clip_alpha: float = 0.7,
        dense_search_params: Optional[dict] = None,
        sparse_search_params: Optional[dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Perform a hybrid dense + sparse search in Milvus.
        
        Args:
            near_text: Text query to embed and search
            collection_name: Name of Milvus collection
            target_vector: Vector field name for dense vectors
            target_sparse_vector: Sparse vector field name
            limit: Number of results
            alpha: Dense vs sparse score weight
            clip_alpha: Weight for fusing CLIP image and text embeddings
            dense_search_params: Milvus search params for dense ANN for target_vector
        """
        # Validate model utils
        if not self.model_utils:
            raise ValueError("Model utils required for CLIP hybrid query")

        # Generate dense embedding via Clip
        dense_embedding = self.model_utils.get_clip_embeddings(near_text, image=None, alpha=clip_alpha)
        if dense_embedding is None:
            logging.error("Failed to generate CLIP embedding")
            return pd.DataFrame()
        
        # Convert to plain list
        if isinstance(dense_embedding, np.ndarray):
            dense_embedding = dense_embedding.tolist()

        # Set defaults if needed
        dense_search_params = dense_search_params or {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # Build search requests
        ## Dense vector search req
        dense_req = AnnSearchRequest(
            data=[dense_embedding],
            anns_field=target_vector,
            param=dense_search_params,
            limit=limit
        )

        ## sparse text search req
        sparse_req = AnnSearchRequest(
            data=[near_text],
            anns_field=target_sparse_vector,
            limit=limit
        )

        # Ranker
        # Weights: `alpha` for dense, `1-alpha` for sparse
        dense_weight = alpha
        sparse_weight = 1.0 - alpha if target_sparse_vector else 0.0
        ranker = WeightedRanker(dense_weight, sparse_weight)

        # Do hybrid search
        results = self.milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=limit,
        )

        # Flatten results
        rows = []
        for hits in results:
            for hit in hits:
                rows.append(self._extract_object_data(hit))

        return pd.DataFrame(rows)

class MilvusAdapter(VectorDBAdapter):
    """Milvus adapter using framework MilvusQuery implementation."""
    
    @classmethod
    def init_client(cls, **kwargs):
        """
        Initialize and return a Milvus connection alias.
        
        Args:
            **kwargs: Connection parameters:
                - uri: Milvus URI (default: from MILVUS_URI env or "http://localhost:19530")
                - user: Milvus user (default: from MILVUS_USER env or "")
                - token: Milvus token (default: from MILVUS_TOKEN env or "")
                - password: Milvus password (optional)
                - db_name: Milvus database name (default: from MILVUS_DB_NAME env or "")
        
        Returns:
            Connection alias string
        """
        _check_milvus_available()
        uri = kwargs.get("uri", os.getenv("MILVUS_URI", "http://localhost:19530"))
        user = kwargs.get("user", os.getenv("MILVUS_USER", ""))
        token = kwargs.get("token", os.getenv("MILVUS_TOKEN", ""))
        password = kwargs.get("password", os.getenv("MILVUS_PASSWORD", None))
        db_name = kwargs.get("db_name", os.getenv("MILVUS_DB_NAME", ""))

        logging.debug(f"Attempting to connect to Milvus at {uri}")

        # Retry logic to connect to Milvus
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
                logging.debug("Successfully connected to Milvus")
                return milvus_client
            except Exception as e:
                retry_count += 1
                logging.error(f"Failed to connect to Milvus (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    logging.debug("Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    raise
    
    def __init__(self, milvus_client=None, collection_name: str = None, triton_client=None, query_class=None, **client_kwargs):
        """
        Initialize Milvus adapter.
        
        Args:
            milvus_client: Pre-initialized Milvus client (optional)
            collection_name: Default collection name to use for queries
            triton_client: Pre-initialized Triton client (optional)
            query_class: Query class to use (defaults to MilvusQuery from adapters)
            **client_kwargs: Additional parameters to pass to init_client if milvus_client is None
        """
        _check_milvus_available()
        if milvus_client is None:
            milvus_client = self.init_client(**client_kwargs)
        
        self.milvus_client = milvus_client
        self.default_collection_name = collection_name
        self.triton_client = triton_client
        
        # Use MilvusQuery by default, or allow custom query class
        if query_class is None:
            query_class = MilvusQuery
        
        self.query_class = query_class
    
    def _get_query_instance(self, collection_name: str):
        """Get or create a query instance for the given collection."""
        if not hasattr(self, '_query_instances'):
            self._query_instances = {}
        
        if collection_name not in self._query_instances:
            self._query_instances[collection_name] = self.query_class(
                self.milvus_client,
                collection_name,
                self.triton_client
            )
        
        return self._query_instances[collection_name]
    
    def search(
        self, 
        query: str, 
        collection_name: str,
        target_vector: str,
        limit: int = 25,
        query_method: str = "clip_hybrid_query",
        **kwargs
    ) -> QueryResult:
        """
        Perform a search query on Milvus.
        
        Args:
            query: Text query string
            collection_name: Name of the collection to search
            target_vector: Name of the vector field to search in
            limit: Maximum number of results to return
            query_method: Method name to use (e.g., "clip_hybrid_query", "vector_query")
            **kwargs: Additional search parameters
            
        Returns:
            QueryResult containing search results
        """
        query_instance = self._get_query_instance(collection_name)
        
        # Use the generic query method from the Query interface
        df = query_instance.query(
            near_text=query,
            collection_name=collection_name,
            target_vector=target_vector,
            limit=limit,
            query_method=query_method,
            **kwargs
        )
        
        # Convert DataFrame to list of dicts for QueryResult
        results = df.to_dict('records')
        
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
                - auto_id: Whether to auto-generate IDs for the collection
                - enable_dynamic_field: Whether to enable dynamic fields for the collection
                - fields: List of field definitions, must follow Milvus field definition schema
                    - https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/CollectionSchema/add_field.md
                - functions: Optional list of function definitions, must follow Milvus function definition schema
                    - https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Function/Function.md
                - index: Optional index definition, must follow Milvus index definition schema
                    - https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Management/add_index.md
            **kwargs: Additional parameters
        
        Returns:
            True if collection was created successfully
        """
        try:
            if "name" not in schema_config:
                raise ValueError("Collection name is required in schema_config")
            if "fields" not in schema_config:
                raise ValueError("Fields are required in schema_config")
            collection_name = schema_config["name"]
            
            # Delete existing collection if it exists
            if self.milvus_client.has_collection(collection_name):
                logging.debug(f"Collection '{collection_name}' exists. Deleting it first...")
                self.milvus_client.drop_collection(collection_name)
                
                # Wait until it's fully deleted
                while self.milvus_client.has_collection(collection_name):
                    time.sleep(1)
            
            # Extract schema components
            fields_config = schema_config.get("fields", [])
            index_config = schema_config.get("index", [])
            functions_config = schema_config.get("functions", [])

            # Create schema
            schema = self.milvus_client.create_schema(
                auto_id=schema_config.get("auto_id", False),
                enable_dynamic_field=schema_config.get("enable_dynamic_field", False),
            )
            
            # Process fields from schema_config
            for field_config in fields_config:
                try:
                    schema.add_field(**field_config)
                    logging.debug(f"Added field '{field_config.get('field_name')}' to collection '{collection_name}'")
                except Exception as e:
                    logging.error(f"Failed to add field: {e}")


            # Add indices if provided
            index_params = None
            if len(index_config) > 0:
                index_params = self.milvus_client.prepare_index_params()
                for index_config in index_config:
                    try:
                        index_params.add_index(**index_config)
                        logging.debug(f"Added index for field '{index_config.get("field_name")}' to collection '{collection_name}'")
                    except Exception as e:
                        logging.error(f"Failed to add index: {e}")

            # Add functions if provided
            if len(functions_config) > 0:
                for func_config in functions_config:
                    try:                                            
                        function = Function(**func_config)
                        schema.add_function(function)
                        logging.debug(f"Added function '{func_config.get("name")}' to collection '{collection_name}'")
                    except Exception as e:
                        logging.error(f"Failed to add function: {e}")
            
            # Create collection
            self.milvus_client.create_collection(
                name=collection_name,
                index_params=index_params,
                schema=schema,
            )
            
            # Load collection to memory
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
        """
        Delete a Milvus collection.
        
        Args:
            collection_name: Name of the collection to delete
            **kwargs: Additional parameters
        
        Returns:
            True if collection was deleted successfully
        """
        try:
            if self.milvus_client.has_collection(collection_name):
                self.milvus_client.drop_collection(collection_name)
                logging.debug(f"Collection '{collection_name}' deleted.")
                return True
            else:
                logging.debug(f"Collection '{collection_name}' does not exist.")
                return False
        except Exception as e:
            logging.error(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def insert_data(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
        **kwargs
    ) -> int:
        """
        Insert data into Milvus collection.
        
        Args:
            collection_name: Name of the collection to insert into
            data: List of dictionaries with keys corresponding to the collection fields and values corresponding to the data to insert
            **kwargs: Additional parameters
        
        Returns:
            Number of items successfully inserted
        """
        try:
            res = self.milvus_client.insert(
                collection_name=collection_name,
                data=data
            )
                    
            total_inserted = len(res["insert_count"])
            
            logging.debug(f"Inserted {total_inserted} items into '{collection_name}'.")

            return total_inserted
        except Exception as e:
            logging.error(f"Error inserting data into '{collection_name}': {e}")
            return 0
    
    def close(self):
        """Close the Milvus connection."""
        if self.milvus_client:
            self.milvus_client.close()


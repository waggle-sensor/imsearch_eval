"""
Weaviate-based adapters for benchmarking framework.

This module provides all Weaviate-related adapters:
- WeaviateQuery: Query class for Weaviate with various search methods
- WeaviateAdapter: VectorDBAdapter implementation for Weaviate
"""

import os
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Callable

# Check for optional dependencies
try:
    import weaviate
    from weaviate.classes.query import MetadataQuery, HybridFusion, Rerank
    from weaviate.client import WeaviateClient
    _WEAVIATE_AVAILABLE = True
except ImportError:
    weaviate = None
    MetadataQuery = HybridFusion = Rerank = None
    _WEAVIATE_AVAILABLE = False

# Try to import TritonModelUtils (optional, but needed for some query methods)
try:
    from .triton import TritonModelUtils
    _TRITON_AVAILABLE = True
except ImportError:
    TritonModelUtils = None
    _TRITON_AVAILABLE = False

from ..framework.interfaces import VectorDBAdapter, QueryResult, Query
from ..framework.model_utils import ModelUtils

def _check_weaviate_available():
    """Check if Weaviate dependencies are installed."""
    if not _WEAVIATE_AVAILABLE:
        raise ImportError(
            "Weaviate adapters require 'weaviate-client'. "
            "Install it with: pip install imsearch_eval[weaviate]"
        )


class WeaviateQuery(Query):
    """
    Query class for Weaviate that provides various search methods.
    """
    
    def __init__(self, weaviate_client: WeaviateClient, triton_client=None, model_utils: ModelUtils = None):
        """
        Initialize Weaviate query instance.
        
        Args:
            weaviate_client: Weaviate client connection
            triton_client: Optional Triton client for generating embeddings
            model_utils: Optional ModelUtils instance (if None and triton_client provided, creates TritonModelUtils)
        """
        _check_weaviate_available()
        self.weaviate_client = weaviate_client
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
        query_method: Callable = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform a search query on Weaviate.
        
        This is the generic query method that routes to specific query methods
        based on the query_method parameter.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            target_vector: Name of the vector space to search in
            limit: Maximum number of results to return
            query_method: Method name to call (e.g., "clip_hybrid_query", "hybrid_query", "colbert_query", custom callable function)
            **kwargs: Additional search parameters passed to the specific query method
        
        Returns:
            DataFrame with search results
        """
        # Route to the appropriate query method
        if query_method is None:
            query_method = self.clip_hybrid_query
        return query_method(near_text, collection_name, target_vector, limit, **kwargs)
    
    def get_location_coordinate(self, obj, coordinate_type: str) -> float:
        """
        Helper function to safely fetch latitude or longitude from the location property.
        
        Args:
            obj: Weaviate object
            coordinate_type: "latitude" or "longitude"
        
        Returns:
            Coordinate value as float, or 0.0 if not available
        """
        location = obj.properties.get("location", "")
        if location:
            try:
                if coordinate_type in ["latitude", "longitude"]:
                    return float(getattr(location, coordinate_type, 0.0))
            except (AttributeError, ValueError):
                logging.warning(f"Invalid {coordinate_type} value found for obj {obj.uuid}")
        return 0.0
    
    def _extract_object_data(self, obj) -> dict:
        """
        Extract object data from Weaviate result.
        
        Args:
            obj: Weaviate object from query result
        
        Returns:
            Dictionary with all properties and metadata extracted.
        """
        result = {}
        
        # Always include UUID
        result["uuid"] = str(obj.uuid)
        
        # Extract all properties dynamically
        if hasattr(obj, 'properties') and obj.properties:
            for key, value in obj.properties.items():
                # Skip location as it's handled separately
                if key != "location":
                    result[key] = value
        
        # Extract metadata fields dynamically
        if hasattr(obj, 'metadata') and obj.metadata:
            for key, value in obj.metadata.items():
                result[key] = value
        
        # Handle location coordinates if present
        if hasattr(obj, 'properties') and obj.properties and "location" in obj.properties:
            location_lat = self.get_location_coordinate(obj, "latitude")
            location_lon = self.get_location_coordinate(obj, "longitude")
            if location_lat != 0.0 or location_lon != 0.0:
                result["location_lat"] = location_lat
                result["location_lon"] = location_lon
        
        return result
    
    def hybrid_query(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str,
        limit: int = 25,
        alpha: float = 0.4,
        query_properties: list = [],
        autocut_jumps: int = 0,
        rerank_prop: str = "caption",
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform a hybrid vector and keyword search.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            target_vector: Name of the vector space to search in
            limit: Maximum number of results to return
            alpha: Balance between vector and keyword search (0.0 = keyword only, 1.0 = vector only)
            query_properties: List of properties to search in keyword search
            autocut_jumps: Number of jumps for autocut (0 to disable)
            rerank_prop: Property to rerank by against the query (default: "caption")
            **kwargs: Additional parameters to pass to the weaviate's collection.query.hybrid method
        Returns:
            DataFrame with search results
        """
        collection = self.weaviate_client.collections.get(collection_name)
        
        # Perform hybrid search
        res = collection.query.hybrid(
            query=near_text,
            target_vector=target_vector,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            auto_limit=autocut_jumps if autocut_jumps > 0 else None,
            limit=limit,
            alpha=alpha,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            query_properties=query_properties,
            rerank=Rerank(
                prop=rerank_prop,
                query=near_text
            ),
            **kwargs
        )
        
        # Extract results
        objects = []
        for obj in res.objects:
            obj_data = self._extract_object_data(obj)
            objects.append(obj_data)
        
        return pd.DataFrame(objects)
    
    def colbert_query(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str,
        limit: int = 25,
        autocut_jumps: int = 0,
        rerank_prop: str = "caption",
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform a vector search using ColBERT embeddings.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            target_vector: Name of the vector space to search in
            limit: Maximum number of results to return
            autocut_jumps: Number of jumps for autocut (0 to disable)
            rerank_prop: Property to rerank by against the query (default: "caption")
            **kwargs: Additional parameters to pass to the weaviate's collection.query.near_vector method
        Returns:
            DataFrame with search results
        """
        if not self.model_utils:
            raise ValueError("Model utils is required for ColBERT queries")
        
        collection = self.weaviate_client.collections.get(collection_name)
        
        # Generate ColBERT embedding
        colbert_embedding = self.model_utils.get_colbert_embedding(near_text)
        if colbert_embedding is None:
            logging.error("Failed to generate ColBERT embedding")
            return pd.DataFrame()
        
        # For ColBERT, we need to use the mean of token embeddings for vector search
        # ColBERT returns token-level embeddings, so we average them
        if len(colbert_embedding.shape) > 1:
            colbert_vector = colbert_embedding.mean(axis=0)
        else:
            colbert_vector = colbert_embedding
        
        # Perform vector search
        res = collection.query.near_vector(
            near_vector=colbert_vector,
            target_vector=target_vector,
            auto_limit=autocut_jumps if autocut_jumps > 0 else None,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            rerank=Rerank(
                prop=rerank_prop,
                query=near_text
            ),
            **kwargs
        )
        
        # Extract results
        objects = []
        for obj in res.objects:
            obj_data = self._extract_object_data(obj)
            objects.append(obj_data)
        
        return pd.DataFrame(objects)
    
    def clip_hybrid_query(
        self,
        near_text: str,
        collection_name: str,
        target_vector: str,
        limit: int = 25,
        alpha: float = 0.4,
        clip_alpha: float = 0.7,
        query_properties: list = [],
        autocut_jumps: int = 0,
        rerank_prop: str = "caption",
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform a hybrid search using CLIP embeddings.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            target_vector: Name of the vector space to search in
            limit: Maximum number of results to return
            alpha: Balance between vector and keyword search (0.0 = keyword only, 1.0 = vector only)
            clip_alpha: Weight for fusing CLIP image and text embeddings
            query_properties: List of properties to search in keyword search
            autocut_jumps: Number of jumps for autocut (0 to disable)
            rerank_prop: Property to rerank by against the query (default: "caption")
            **kwargs: Additional parameters to pass to the weaviate's collection.query.hybrid method
        Returns:
            DataFrame with search results
        """
        if not self.model_utils:
            raise ValueError("Model utils is required for CLIP hybrid queries")
        
        collection = self.weaviate_client.collections.get(collection_name)
        
        # Get CLIP embedding
        clip_embedding = self.model_utils.get_clip_embeddings(near_text, image=None, alpha=clip_alpha)
        if clip_embedding is None:
            logging.error("Failed to generate CLIP embedding")
            return pd.DataFrame()
        
        # Perform hybrid search
        res = collection.query.hybrid(
            query=near_text,
            target_vector=target_vector,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            auto_limit=autocut_jumps if autocut_jumps > 0 else None,
            limit=limit,
            alpha=alpha,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            query_properties=query_properties,
            vector=clip_embedding,
            rerank=Rerank(
                prop=rerank_prop,
                query=near_text
            ),
            **kwargs
        )
        
        # Extract results
        objects = []
        for obj in res.objects:
            obj_data = self._extract_object_data(obj)
            objects.append(obj_data)
        
        return pd.DataFrame(objects)


class WeaviateAdapter(VectorDBAdapter):
    """Weaviate adapter using framework WeaviateQuery implementation."""
    
    @classmethod
    def init_client(cls, **kwargs):
        """
        Initialize and return a Weaviate client connection.
        
        Args:
            **kwargs: Connection parameters:
                - host: Weaviate host (default: from WEAVIATE_HOST env or "127.0.0.1")
                - port: Weaviate REST port (default: from WEAVIATE_PORT env or "8080")
                - grpc_port: Weaviate GRPC port (default: from WEAVIATE_GRPC_PORT env or "50051")
        
        Returns:
            Weaviate client connection
        """
        _check_weaviate_available()
        host = kwargs.get("host", os.getenv("WEAVIATE_HOST", "127.0.0.1"))
        port = kwargs.get("port", os.getenv("WEAVIATE_PORT", "8080"))
        grpc_port = kwargs.get("grpc_port", os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        
        logging.debug(f"Attempting to connect to Weaviate at {host}:{port}")
        
        # Retry logic to connect to Weaviate
        while True:
            try:
                client = weaviate.connect_to_local(
                    host=host,
                    port=port,
                    grpc_port=grpc_port
                )
                logging.debug("Successfully connected to Weaviate")
                return client
            except weaviate.exceptions.WeaviateConnectionError as e:
                logging.error(f"Failed to connect to Weaviate: {e}")
                logging.debug("Retrying in 10 seconds...")
                time.sleep(10)
    
    def __init__(self, weaviate_client: WeaviateClient = None, triton_client=None, query_instance: Query = None, **client_kwargs):
        """
        Initialize Weaviate adapter.
        
        Args:
            weaviate_client: Pre-initialized Weaviate client (optional)
            triton_client: Pre-initialized Triton client (optional)
            query_instance: Pre-initialized Query instance (optional, defaults to WeaviateQuery instance)
            **client_kwargs: Additional parameters to pass to init_client if weaviate_client is None
        """
        _check_weaviate_available()
        if weaviate_client is None:
            weaviate_client = self.init_client(**client_kwargs)
        
        self.weaviate_client = weaviate_client
        self.triton_client = triton_client
        
        # Use WeaviateQuery by default, or allow custom query instance
        if query_instance is None:
            query_instance = WeaviateQuery(weaviate_client, triton_client)
        
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
        """
        Perform a search query on Weaviate.
        
        Args:
            query: Text query string
            collection_name: Name of the collection to search
            target_vector: Name of the vector space to search in
            limit: Maximum number of results to return
            query_method: Method/type of query to perform (e.g., "clip_hybrid_query", "hybrid_query", "colbert_query", custom callable function)
            **kwargs: Additional search parameters passed to the specific query method
            
        Returns:
            QueryResult containing search results
        """
        # Use the generic query method from the Query interface
        df = self.query_instance.query(
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
        Create a Weaviate collection.
        
        Args:
            schema_config: Dictionary containing schema configuration, must follow weaviate collection creation schema
            **kwargs: Additional parameters
        
        Returns:
            True if collection was created successfully
        """
        try:
            if "name" not in schema_config:
                raise ValueError("Collection name is required in schema_config")
            collection_name = schema_config["name"]
            
            # Delete existing collection if it exists
            if collection_name in self.weaviate_client.collections.list_all():
                logging.debug(f"Collection '{collection_name}' exists. Deleting it first...")
                self.weaviate_client.collections.delete(collection_name)
                
                # Wait until it's fully deleted
                while collection_name in self.weaviate_client.collections.list_all():
                    time.sleep(1)
            
            # Create collection - unpack schema_config directly
            self.weaviate_client.collections.create(**schema_config)
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
        Delete a Weaviate collection.
        
        Args:
            collection_name: Name of the collection to delete
            **kwargs: Additional parameters
        
        Returns:
            True if collection was deleted successfully
        """
        try:
            if collection_name in self.weaviate_client.collections.list_all():
                self.weaviate_client.collections.delete(collection_name)
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
        batch_size: int = 100,
        **kwargs
    ) -> int:
        """
        Insert data into Weaviate collection.
        
        Args:
            collection_name: Name of the collection to insert into
            data: List of dictionaries with 'properties' and optionally 'vector' keys
            batch_size: Size of batches for insertion
            **kwargs: Additional parameters
        
        Returns:
            Number of items successfully inserted
        """
        from itertools import islice
        
        try:
            collection = self.weaviate_client.collections.get(collection_name)
            inserted_count = 0
            
            # Helper to batch data
            def batched(iterable, n):
                it = iter(iterable)
                while batch := list(islice(it, n)):
                    yield batch
            
            # Insert in batches
            with collection.batch.fixed_size(batch_size=batch_size) as batch:
                for item in data:
                    if item is None:
                        continue
                    
                    properties = item.get("properties", {})
                    vector = item.get("vector", {})
                    
                    batch.add_object(properties=properties, vector=vector)
                    inserted_count += 1
                    
                    # Stop if too many errors
                    if batch.number_errors > 5:
                        logging.error("Batch import stopped due to excessive errors.")
                        break
            
            logging.debug(f"Inserted {inserted_count} items into '{collection_name}'.")
            return inserted_count
            
        except Exception as e:
            logging.error(f"Error inserting data into '{collection_name}': {e}")
            return 0
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.weaviate_client:
            self.weaviate_client.close()


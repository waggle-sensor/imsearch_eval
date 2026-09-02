"""Abstract interfaces for vector database and model providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Iterable, Tuple
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
import logging
from tqdm import tqdm

from .image_utils import ensure_rgb

# Soft-failure sentinel returned by process_item (caption empty); not inserted.
DLQ_SOFT_KEY = "__dlq_soft__"


def load_dlq_item(dataset: Any, dataset_idx: int) -> Dict[str, Any]:
    """
    Reload a dataset row for DLQ retry.

    Failures store only ``dataset_idx`` (no image payload) so the DLQ stays
    small; this fetches the row from the on-disk/memory-mapped HF dataset.
    """
    row = dataset[int(dataset_idx)]
    if isinstance(row, dict):
        item = dict(row)
    elif hasattr(row, "to_dict"):
        item = dict(row.to_dict())
    else:
        item = dict(row)

    image = item.get("image")
    if isinstance(image, Image.Image):
        item["image"] = ensure_rgb(image)
    return item


class QueryResult:
    """Container for query results from a vector database."""
    
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize query result.
        
        Args:
            results: List of dictionaries containing result data
        """
        self.results = results
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame(self.results)


class VectorDBAdapter(ABC):
    """Abstract interface for vector database adapters."""
    
    @classmethod
    @abstractmethod
    def init_client(cls, **kwargs):
        """
        Initialize and return a client connection to the vector database.
        
        Args:
            **kwargs: Connection parameters (host, port, etc.)
            
        Returns:
            Client connection object
        """
        pass
    
    @abstractmethod
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
        Perform a search query on the vector database.
        
        Args:
            query: Text query string
            collection_name: Name of the collection/index to search
            target_vector: Name of the vector space to search in
            limit: Maximum number of results to return
            query_method: Method/type of query to perform (implementation-specific, e.g., "hybrid", "vector", "keyword", custom callable function)
            **kwargs: Additional search parameters passed to the specific query method (implementation-specific)
            
        Returns:
            QueryResult containing search results
        """
        pass
    
    @abstractmethod
    def create_collection(
        self,
        schema_config: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Create a collection/index in the vector database.
        
        Args:
            schema_config: Dictionary containing schema configuration
            **kwargs: Additional collection-specific parameters
            
        Returns:
            True if collection was created successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_collection(
        self,
        collection_name: str,
        **kwargs
    ) -> bool:
        """
        Delete a collection/index from the vector database.
        
        Args:
            collection_name: Name of the collection to delete
            **kwargs: Additional parameters
            
        Returns:
            True if collection was deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def insert_data(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
        batch_size: int = 100,
        **kwargs
    ) -> int:
        """
        Insert data into the vector database collection.
        
        Args:
            collection_name: Name of the collection to insert into
            data: List of dictionaries containing data to insert
                   Each dict should have 'properties' and optionally 'vector' keys
            batch_size: Size of batches for insertion
            **kwargs: Additional insertion parameters
            
        Returns:
            Number of items successfully inserted
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ModelProvider(ABC):
    """Abstract interface for model providers."""
    
    @abstractmethod
    def get_embedding(
        self, 
        text: str, 
        image: Optional[Image.Image] = None,
        model_name: str = "default"
    ) -> Any:
        """
        Get embedding for text and/or image.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            model_name: Name of the model to use
            
        Returns:
            Embedding vector (numpy array or similar)
        """
        pass
    
    @abstractmethod
    def generate_caption(self, image: Image.Image, prompt: str , model_name: str = "default", enable_thinking: bool = True) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image to caption
            prompt: Prompt to use for the model
            model_name: Name of the model to use
            enable_thinking: Whether to enable thinking (default: True). Not all models support thinking.
        Returns:
            Generated caption string
        """
        pass


class BenchmarkDataset(ABC):
    """Abstract interface for benchmark datasets."""
    
    @abstractmethod
    def load(self, split: str = "test", sample_size: int = None, seed: int = None, **kwargs) -> pd.DataFrame:
        """
        Load the dataset.
        
        Args:
            split: Dataset split to load (e.g., "test", "train", "val")
            sample_size: Number of samples to load from the dataset (if None, load all samples)
            seed: Seed for random number generator (if None, use a random seed)
            **kwargs: Additional dataset-specific parameters
            
        Returns:
            DataFrame containing the dataset
        """
        pass
    
    @abstractmethod
    def get_query_column(self) -> str:
        """
        Get the name of the column containing the query text.
        
        Returns:
            Column name for queries
        """
        pass
    
    @abstractmethod
    def get_query_id_column(self) -> str:
        """
        Get the name of the column containing the query ID.
        
        Returns:
            Column name for query IDs
        """
        pass
    
    @abstractmethod
    def get_relevance_column(self) -> str:
        """
        Get the name of the column containing relevance labels.
        
        Returns:
            Column name for relevance (1 = relevant, 0 = irrelevant)
        """
        pass
    
    def get_metadata_columns(self) -> List[str]:
        """
        Get optional metadata columns to include in evaluation stats.
        
        Returns:
            List of column names for metadata (e.g., ["category", "supercategory"])
        """
        return []

class Config(ABC):
    """
    Abstract interface for configuration/hyperparameters. 
    Class variables starting with _ are considered sensitive.
    """
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return getattr(self, key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    def to_csv(self) -> str:
        """
        Convert the configuration to CSV file content. Skip private variables and variables starting with _.
        
        Returns:
            CSV file content with config variable, value, and type
        """ 
        config_data = []
        
        # Get all config variables (excluding private ones starting with _)
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name, None)) and not attr_name.startswith('os'):
                attr_value = getattr(self, attr_name)
                config_data.append({
                    'Config Variable': attr_name,
                    'Value': str(attr_value),
                    'Type': type(attr_value).__name__
                })
        
        config_df = pd.DataFrame(config_data)
        config_df = config_df.sort_values('Config Variable')
        return config_df.to_csv(index=False)

class DataLoader(ABC):
    """
    Abstract interface for loading data into vector databases.
    """
    
    def __init__(self, config: Config, model_provider: ModelProvider, dataset: BenchmarkDataset):
        """
        Initialize the DataLoader with a configuration.
        
        Args:
            config: Configuration object implementing the Config interface
            model_provider: Model provider for generating embeddings/captions
            dataset: Benchmark dataset instance
        """
        self.config = config
        self.model_provider = model_provider
        self.dataset = dataset
    
    @abstractmethod
    def process_item(
        self,
        item: Dict[str, Any],
        *,
        force_insert: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single dataset item and prepare it for vector database insertion.

        Args:
            item: Dictionary containing raw dataset item
            force_insert: When True, skip soft-DLQ for empty captions and insert
                with an empty caption (used after DLQ retries are exhausted).
        Returns:
            Insertable dict; None for hard failure; or a soft-DLQ sentinel dict
            with ``__dlq_soft__=True`` when caption generation failed and
            ``force_insert`` is False.
        """
        pass
    
    @abstractmethod
    def get_schema_config(self) -> Dict[str, Any]:
        """
        Get the schema configuration for creating the collection.
        
        Returns:
            Dictionary containing schema configuration
        """
        pass

    def process_batch(
        self,
        batch_size: int,
        dataset: Iterable = None,
        split: str = "test",
        sample_size: int = None,
        seed: int = None,
        workers: int = 0,
        on_batch: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        on_failure: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process items in parallel with a continuously filled worker pool.

        Keeps up to ``workers`` tasks in flight (no batch barrier). When
        ``on_batch`` is set, completed items are streamed in chunks of
        ``batch_size`` and are not retained in the returned list (saves RAM).

        Hard failures (``None`` / exceptions) and soft DLQ sentinels
        (``__dlq_soft__=True``) are collected and not inserted. Failure entries
        store ``dataset_idx`` (and ids) only — not image payloads — so callers
        can reload rows from ``dataset`` on retry via ``load_dlq_item``.

        Args:
            batch_size: Insert/stream chunk size when ``on_batch`` is set;
                otherwise unused for scheduling (kept for API compatibility)
            dataset: Optional pre-loaded dataset. If None, will load using dataset.load()
            split: Dataset split to use if loading dataset
            sample_size: Number of samples to load from the dataset (if None, load all samples)
            seed: Seed for random number generator (if None, use a random seed)
            workers: Number of workers to use for parallel processing (if 0, use all available CPUs)
            on_batch: Optional callback invoked with each completed success chunk
            on_failure: Optional callback invoked for each hard/soft failure entry
        Returns:
            ``(results, failures)`` where ``results`` is empty when ``on_batch``
            is set, and ``failures`` holds lightweight DLQ entries for retry.
        """
        logging.debug("Starting Data Loader...")

        if dataset is None:
            dataset = self.dataset.load(split=split, sample_size=sample_size, seed=seed)

        total_items = len(dataset)
        num_workers = workers if workers > 0 else os.cpu_count()
        keep_results = on_batch is None
        results: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        chunk: List[Dict[str, Any]] = []
        next_idx = 0
        exhausted = False
        # In-flight map: future -> (dataset_idx, image_id, query_id). Do not
        # retain the row here; the worker already holds the only PIL reference.
        future_to_meta: Dict[Any, Tuple[int, str, str]] = {}

        def _image_id(item: Any) -> str:
            if not isinstance(item, dict):
                return ""
            for key in (
                "image_id",
                "inat24_file_name",
                "inat24_image_id",
            ):
                val = item.get(key)
                if val not in (None, ""):
                    return str(val)
            return ""

        def _query_id(item: Any) -> str:
            if not isinstance(item, dict):
                return ""
            val = item.get("query_id")
            if val not in (None, ""):
                return str(val)
            return ""

        def _record_failure(
            dataset_idx: int,
            reason: str,
            error: str,
            image_id: str = "",
            query_id: str = "",
        ) -> None:
            entry = {
                "dataset_idx": int(dataset_idx),
                "reason": reason,
                "error": error or "",
                "attempt": 0,
                "image_id": image_id or "",
                "query_id": query_id or "",
            }
            failures.append(entry)
            if on_failure is not None:
                on_failure(entry)

        def _submit(executor, pending):
            nonlocal next_idx, exhausted
            if next_idx >= total_items:
                exhausted = True
                return
            dataset_idx = next_idx
            next_idx += 1
            item = load_dlq_item(dataset, dataset_idx)
            meta = (dataset_idx, _image_id(item), _query_id(item))
            future = executor.submit(self.process_item, item)
            future_to_meta[future] = meta
            pending.add(future)

        def _flush_chunk():
            nonlocal chunk
            if on_batch is not None and chunk:
                on_batch(chunk)
                chunk = []

        def _handle_success(processed_item: Dict[str, Any]) -> None:
            if keep_results:
                results.append(processed_item)
            else:
                chunk.append(processed_item)
                if len(chunk) >= batch_size:
                    _flush_chunk()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pbar = tqdm(
                total=total_items,
                desc="Processing items",
                unit="item",
            )
            try:
                pending = set()
                while len(pending) < num_workers and not exhausted:
                    _submit(executor, pending)

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        dataset_idx, image_id, query_id = future_to_meta.pop(
                            future, (-1, "", "")
                        )
                        try:
                            processed_item = future.result()
                        except Exception as exc:
                            logging.error(
                                "Error processing item %s: %s",
                                image_id or "unknown",
                                exc,
                            )
                            _record_failure(
                                dataset_idx,
                                "hard_fail",
                                str(exc),
                                image_id=image_id,
                                query_id=query_id,
                            )
                            pbar.update(1)
                            if not exhausted:
                                _submit(executor, pending)
                            continue

                        if processed_item is None:
                            _record_failure(
                                dataset_idx,
                                "hard_fail",
                                "process_item returned None",
                                image_id=image_id,
                                query_id=query_id,
                            )
                        elif (
                            isinstance(processed_item, dict)
                            and processed_item.get(DLQ_SOFT_KEY)
                        ):
                            _record_failure(
                                dataset_idx,
                                processed_item.get("reason", "caption_failed"),
                                processed_item.get(
                                    "error", "empty caption from provider"
                                ),
                                image_id=str(
                                    processed_item.get("image_id") or image_id
                                ),
                                query_id=str(
                                    processed_item.get("query_id") or query_id
                                ),
                            )
                        else:
                            _handle_success(processed_item)

                        pbar.update(1)
                        if not exhausted:
                            _submit(executor, pending)
                _flush_chunk()
            finally:
                pbar.close()

        return results, failures

class Query(ABC):
    """Abstract interface for query classes used by vector database adapters."""
    
    @abstractmethod
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
        Perform a search query on the vector database.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            target_vector: Name of the vector space to search in
            limit: Maximum number of results to return
            query_method: Method/type of query to perform (implementation-specific, e.g., "hybrid", "vector", "keyword", custom callable function)
            **kwargs: Additional search parameters passed to the specific query method (implementation-specific)
        
        Returns:
            DataFrame with search results
        """
        pass


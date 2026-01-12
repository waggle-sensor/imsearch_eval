"""
Huggingface-based adapters for benchmarking framework.

This module provides all Huggingface-related adapters:
- HuggingFaceDataset: Benchmark dataset class for retrieving benchmark datasets from Hugging Face
"""
import pandas as pd
import random
from datasets import load_dataset
from ..framework.interfaces import BenchmarkDataset


class HuggingFaceDataset(BenchmarkDataset):
    """Benchmark dataset class for retrieving benchmark datasets from Huggingface."""
    
    def __init__(self, dataset_name: str = None):
        """
        Initialize Hugging Face benchmark dataset.
        
        Args:
            dataset_name: Hugging Face dataset name
        """
        self.dataset_name = dataset_name
    
    def load(self, split: str = "test", sample_size: int = None, seed: int = None, **kwargs) -> pd.DataFrame:
        """
        Load Hugging Face dataset.
        
        Args:
            split: Dataset split to load (e.g., "test", "train")
            sample_size: Number of samples to load from the dataset (if None, load all samples)
            **kwargs: Additional parameters to pass to datasets.load_dataset
            
        Returns:
            DataFrame containing the Hugging Face dataset
        """
        dataset = load_dataset(self.dataset_name, split=split, **kwargs)
        if seed is not None:
            random_generator = random.Random(seed)
        else:
            random_generator = random.Random()
        if sample_size is not None:
            sampled_indices = random_generator.sample(range(len(dataset)), sample_size)
            dataset = dataset.select(sampled_indices)
        return dataset.to_pandas()
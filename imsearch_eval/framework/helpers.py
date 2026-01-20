"""Helper functions for the benchmarking framework."""

from itertools import islice
from typing import Any, List, Iterable

def BatchedIterator(iterable: Iterable, batch_size: int) -> Iterable[List[Any]]:
    """
    Yield successive batch_size chunks from iterable.
    Args:
        iterable: An iterable (e.g., list, DataFrame rows)
        batch_size: Size of each batch
        
    Yields:
        Iterable[List[Any]]: A batch of items from the iterable
    """
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch
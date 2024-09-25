"""Base Runners Module.

This module defines the base runner class and worker roles for data processing. 
It includes an abstract base class for runners and an enumeration for worker roles 
during multiprocessing.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable

from datasets import IterableDataset

from hyped.common.typing import Sample


class WorkerRole(Enum):
    """Enumeration of different roles a worker can assume during multiprocessing.

    Workers can dynamically switch between these roles based on the current processing stage
    and system needs.
    """

    PROCESSOR = 1
    """Role where the worker processes a shard of data independently. 

    In this role, the worker is responsible for processing its assigned shard of the dataset
    without interacting with other workers. This occurs in Stage 1 where each worker processes
    a distinct shard.
    """

    PRODUCER = 2
    """Role where the worker produces data and adds it to a shared queue. 

    In this role, the worker reads data from a shard and places it into the queue for further
    processing by other workers. This occurs in Stage 2 when the system shifts to multi-worker
    processing of a single shard.
    """

    CONSUMER = 3
    """Role where the worker consumes data from a shared queue for processing. 

    In this role, the worker retrieves data from the queue (populated by a PRODUCER) and processes
    it. This role is also part of Stage 2, where multiple workers collaborate on processing data
    from a single shard.
    """


class BaseRunner(ABC):
    """Abstract base class for data processing runners.

    This class defines the interface for various types of runners that can process datasets.
    Subclasses must implement the :func:`run` method to provide specific data processing logic.
    """

    @abstractmethod
    def run(ds: IterableDataset, fn: Callable[[Sample], Any]) -> None:
        """Execute data processing on the given dataset.

        Args:
            ds (IterableDataset): The dataset to process.
            fn (Callable[[Sample], Any]): The function to apply to each sample in the dataset.
        """
        ...

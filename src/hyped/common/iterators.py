"""Utility module for enhanced iterator control.

This module provides common iterator functionality to simplify the processing
of iterables and queues, offering more flexible control over iteration.
"""
import operator
import sys
from itertools import islice
from queue import Empty, Queue
from typing import Any, Callable, Generic, Iterable, Iterator, Tuple, TypeVar

from tqdm.std import EMA

from .utils import clock

if sys.version_info >= (3, 12):
    from itertools import batched

else:
    T = TypeVar("T")

    def batched(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
        """
        Yields successive n-sized batches from the given iterable.

        Args:
            iterable (Iterable[T]): An iterable to be batched.
            n (int): The size of each batch. Must be at least one.

        Raises:
            ValueError: If n is less than 1.

        Yields:
            Iterator[Tuple[T, ...]]: A generator that yields tuples containing
            n elements each from the iterable. The last batch may contain fewer
            than n elements if the total number of elements is not a multiple of n.
        """
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


def ith_entries(iterable: Iterable[Tuple], i: int) -> Iterator:
    """
    Returns an iterator that yields the i-th entry from each tuple in the iterable using itemgetter.

    Args:
        iterable (Iterable[Tuple]): An iterable that yields tuples.
        i (int): The index of the entry to extract from each tuple.

    Returns:
        Iterator: An iterator yielding the i-th entry from each tuple.
    """
    return map(operator.itemgetter(i), iterable)


class QueueIterator:
    """An iterator for consuming items from a queue.

    This iterator retrieves items from a queue with an optional timeout and can
    terminate iteration upon encountering a sentinel value.
    """

    def __init__(self, queue: Queue, sentinel: Any = None, timeout: None | float = None) -> None:
        """Initialize the iterator.

        Args:
            queue (Queue): The queue to iterate through.
            sentinel (Any): A special value to signal the end of the queue.
            timeout (float): Timeout for waiting on queue items.
        """
        self.queue = queue
        self.sentinel = sentinel
        self.timeout = timeout

    def __iter__(self) -> Iterable[Any]:
        """Return the iterator object itself.

        Returns:
            Iterable[Any]: The iterator object itself.
        """
        return self

    def __next__(self) -> Any:
        """Retrieve the next item from the queue.

        Returns:
            Any: The next item from the queue.

        Raises:
            StopIteration: If the sentinel value is encountered or if the queue is empty.
        """
        try:
            # Get the next item from the queue with a timeout
            item = self.queue.get(timeout=self.timeout)

            # If the sentinel value is encountered, raise StopIteration to end iteration
            if item == self.sentinel:
                raise StopIteration

            return item
        except Empty:
            raise StopIteration


T = TypeVar("T")


class StoppableIterator(Generic[T]):
    """An iterator that can be stopped manually.

    This iterator allows manual interruption of iteration by calling the :func:`stop`
    method, which causes the iterator to stop yielding items.
    """

    def __init__(self, iterable: Iterable[T]) -> None:
        """Initialize the stoppable iterator.

        Args:
            iterable (Iterable[T]): The iterable to iterate over.
        """
        self.iterable = iter(iterable)
        self.stopped = False

    def stop(self) -> None:
        """Stop the iteration.

        Sets the stopped flag to True, causing the iterator to stop yielding items.
        """
        self.stopped = True

    def __iter__(self) -> Iterable[T]:
        """Return the iterator object itself.

        Returns:
            Iterable[T]: The iterator object itself.
        """
        return self

    def __next__(self) -> T:
        """Retrieve the next item from the iterable.

        Returns:
            T: The next item from the iterable.

        Raises:
            StopIteration: If iteration is stopped or if the iterable is exhausted.
        """
        if self.stopped:
            raise StopIteration

        return next(self.iterable)


T = TypeVar("T")


class TimedIterator(Generic[T]):
    """An iterator wrapper that tracks the total and smoothed average time per iteration.

    An iterator wrapper that tracks the time taken to iterate over elements
    of an iterable. It provides the total time and the smoothed average time
    per iteration using an Exponential Moving Average (EMA).
    """

    def __init__(self, iterable: Iterable[T], smoothing: float = 0.3) -> None:
        """Initializes the :class:`TimedIterator`.

        Args:
            iterable (Iterable[T]): The iterable object to track.
            smoothing (float, optional): The smoothing factor used for the EMA calculation.
                Defaults to 0.3.
        """
        self.iterable = iterable
        self.ema = EMA(smoothing=smoothing)
        self.total = 0

    def smooth_time(self) -> float:
        """Returns the smoothed average time per iteration.

        Returns:
            float: The average time per iteration based on the EMA.
        """
        return self.ema()

    def total_time(self) -> float:
        """Returns the total accumulated time spent iterating.

        Returns:
            float: The total time spent iterating over the iterable.
        """
        return self.total

    def __iter__(self) -> Iterable[T]:
        """Returns an iterator object.

        Returns:
            Iterable[T]: The TimedIterator itself.
        """
        return self

    def __next__(self) -> T:
        """Returns the next element from the iterable and updates the time statistics.

        Tracks the time taken to retrieve the next item, updates the EMA and total time,
        and returns the next item.

        Returns:
            T: The next item from the iterable.

        Raises:
            StopIteration: When the iterable is exhausted.
        """
        # track time required for next item
        st = clock()
        item = next(self.iterable)
        # update average and return the item
        dt = clock() - st
        self.ema(dt)
        self.total += dt
        return item


T = TypeVar("T")
U = TypeVar("U")


class BatchBuffer(Generic[T, U]):
    """A buffer that collects items until a specified batch.

    It applies a function when the target batch size is reached.
    """

    def __init__(self, batch_size: int, apply_function: Callable[[list[T]], U]) -> None:
        """Initialize the :class:`BatchBuffer`.

        Args:
            batch_size (int): The number of items to collect before applying the function.
            apply_function (Callable[[List[T]], U]): The function to apply to the collected items.
        """
        self._batch_size = batch_size
        self._apply_function = apply_function
        self._buffer = []

    def add(self, item: T) -> U | None:
        """Add an item to the buffer and apply the function if the batch size is reached.

        Args:
            item (T): The item to add to the buffer.

        Returns:
            U | None: The result of applying the function if the batch size is reached; otherwise,
            returns :code:`None`.
        """
        self._buffer.append(item)
        if self.is_full():
            return self.flush()

    def flush(self) -> U | None:
        """Apply the function to the collected items and clear the buffer.

        Returns:
            U | None: The result of applying the function to the collected items.
        """
        if len(self._buffer) > 0:
            out = self._apply_function(self._buffer)
            self.clear()
            return out

    def is_full(self) -> bool:
        """Check if the buffer is full.

        Returns:
            bool: True if the buffer has reached the batch size; otherwise, False.
        """
        return len(self._buffer) >= self._batch_size

    def clear(self) -> None:
        """Clear the buffer without applying the function."""
        self._buffer.clear()

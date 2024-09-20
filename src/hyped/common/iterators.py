"""Utility module for enhanced iterator control.

This module provides common iterator functionality to simplify the processing
of iterables and queues, offering more flexible control over iteration.
"""
import sys
from itertools import islice
from queue import Empty, Queue
from typing import Any, Iterable, Iterator, Tuple, TypeVar

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

        Example:
            >>> list(batched('ABCDEFG', 3))
            [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]
        """
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


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


class StoppableIterator(object):
    """An iterator that can be stopped manually.

    This iterator allows manual interruption of iteration by calling the :func:`stop`
    method, which causes the iterator to stop yielding items.
    """

    def __init__(self, iterable: Iterable[Any]) -> None:
        """Initialize the stoppable iterator.

        Args:
            iterable (Iterable[Any]): The iterable to iterate over.
        """
        self.iterable = iter(iterable)
        self.stopped = False

    def stop(self) -> None:
        """Stop the iteration.

        Sets the stopped flag to True, causing the iterator to stop yielding items.
        """
        self.stopped = True

    def __iter__(self) -> Iterable[Any]:
        """Return the iterator object itself.

        Returns:
            Iterable[Any]: The iterator object itself.
        """
        return self

    def __next__(self) -> Any:
        """Retrieve the next item from the iterable.

        Returns:
            Any: The next item from the iterable.

        Raises:
            StopIteration: If iteration is stopped or if the iterable is exhausted.
        """
        if self.stopped:
            raise StopIteration

        return next(self.iterable)

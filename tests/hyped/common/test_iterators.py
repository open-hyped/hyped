import importlib
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

import hyped.common.iterators
from hyped.common.iterators import BatchBuffer, QueueIterator, StoppableIterator


def test_batched():
    mock_version_info = MagicMock()
    mock_version_info.__ge__ = MagicMock(return_value=False)
    # reload iterators module with lower python version to get custom implementation
    with patch("hyped.common.iterators.sys.version_info", mock_version_info):
        importlib.reload(hyped.common.iterators)
        from hyped.common.iterators import batched

    # Test with a simple string
    result = list(batched("ABCDEFG", 3))
    expected = [("A", "B", "C"), ("D", "E", "F"), ("G",)]
    assert result == expected, "The batched output did not match the expected result."

    # Test with a list of numbers
    result = list(batched([1, 2, 3, 4, 5], 2))
    expected = [(1, 2), (3, 4), (5,)]
    assert result == expected, "The batched output for numbers did not match the expected result."

    # Test with n greater than the length of the iterable
    result = list(batched("AB", 5))
    expected = [("A", "B")]
    assert (
        result == expected
    ), "The batched output for n greater than iterable length did not match the expected result."

    # Test ValueError for n less than 1
    with pytest.raises(ValueError, match="n must be at least one"):
        list(batched("ABCDEFG", 0))

    # Test ValueError for negative n
    with pytest.raises(ValueError, match="n must be at least one"):
        list(batched("ABCDEFG", -2))


class TestQueueIterator:
    def test_queue_iterator_basic(self):
        q = Queue()
        items = [1, 2, 3, 4]

        # Put items in the queue
        for item in items:
            q.put(item)
        q.put(None)  # Sentinel value

        iterator = QueueIterator(q, sentinel=None)

        # Ensure that all items are returned in order
        assert list(iterator) == items

    def test_queue_iterator_empty_queue(self):
        q = Queue()
        iterator = QueueIterator(q, sentinel=None, timeout=0.1)

        # Ensure that StopIteration is raised when queue is empty
        with pytest.raises(StopIteration):
            next(iterator)

    def test_queue_iterator_with_sentinel(self):
        q = Queue()
        items = [1, 2, 3, "stop", 5]

        # Put items in the queue
        for item in items:
            q.put(item)

        # Sentinel value is "stop"
        iterator = QueueIterator(q, sentinel="stop")

        # Ensure items before sentinel are returned
        assert list(iterator) == [1, 2, 3]

    def test_queue_iterator_with_timeout(self):
        q = Queue()

        # No items in the queue, should raise StopIteration after timeout
        iterator = QueueIterator(q, timeout=0.1)
        with pytest.raises(StopIteration):
            next(iterator)


class TestStoppableIterator:
    def test_stoppable_iterator_basic(self):
        iterable = [1, 2, 3, 4, 5]
        iterator = StoppableIterator(iterable)

        # Ensure all items are returned in order
        assert list(iterator) == iterable

    def test_stoppable_iterator_stop(self):
        iterable = [1, 2, 3, 4, 5]
        iterator = StoppableIterator(iterable)

        # Retrieve first two items
        assert next(iterator) == 1
        assert next(iterator) == 2

        # Stop the iterator
        iterator.stop()

        # Ensure StopIteration is raised after stop is called
        with pytest.raises(StopIteration):
            next(iterator)

    def test_stoppable_iterator_exhausted(self):
        iterable = [1, 2]
        iterator = StoppableIterator(iterable)

        # Exhaust the iterator
        assert next(iterator) == 1
        assert next(iterator) == 2

        # Ensure StopIteration is raised after all items are exhausted
        with pytest.raises(StopIteration):
            next(iterator)


class TestBatchBuffer:
    def test_initialization(self):
        def dummy_function(batch: list[int]) -> int:
            return sum(batch)

        buffer = BatchBuffer(batch_size=3, apply_function=dummy_function)

        assert buffer._batch_size == 3
        assert buffer._apply_function == dummy_function
        assert buffer._buffer == []

    def test_add_and_flush(self):
        def dummy_function(batch: list[int]) -> int:
            return sum(batch)

        buffer = BatchBuffer(batch_size=3, apply_function=dummy_function)

        # Adding items to the buffer
        result = buffer.add(1)
        assert result is None
        assert buffer._buffer == [1]

        result = buffer.add(2)
        assert result is None
        assert buffer._buffer == [1, 2]

        # This should trigger the flush
        result = buffer.add(3)
        assert result == 6  # 1 + 2 + 3
        assert buffer._buffer == []  # Buffer should be cleared

    def test_partial_flush(self):
        def dummy_function(batch: list[int]) -> int:
            return sum(batch)

        buffer = BatchBuffer(batch_size=3, apply_function=dummy_function)

        buffer.add(1)
        buffer.add(2)

        # This should not flush since the buffer is not full
        result = buffer.add(3)
        assert result == 6  # 1 + 2 + 3
        assert buffer._buffer == []  # Buffer should be cleared

        # Adding again to see partial behavior
        buffer.add(4)
        assert buffer._buffer == [4]  # Buffer should contain 4

    def test_is_full(self):
        f = MagicMock()
        buffer = BatchBuffer(batch_size=3, apply_function=f)

        assert not buffer.is_full()  # Initially empty
        assert not f.called
        buffer.add(1)
        assert not buffer.is_full()  # One item added
        assert not f.called
        buffer.add(2)
        assert not buffer.is_full()  # Two items added
        assert not f.called
        buffer.add(3)
        assert f.called

    def test_clear(self):
        def dummy_function(batch: list[int]) -> int:
            return sum(batch)

        buffer = BatchBuffer(batch_size=3, apply_function=dummy_function)

        buffer.add(1)
        buffer.add(2)
        buffer.clear()  # Manually clear the buffer

        assert buffer._buffer == []  # Buffer should be empty

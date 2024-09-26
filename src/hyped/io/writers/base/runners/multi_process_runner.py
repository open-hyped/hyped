"""Dynamic Multiprocessing Runner Module.

This module implements the :class:`DynamicMultiprocessingRunner` class, which manages and runs 
multiple worker processes for parallel data processing. The runner dynamically assigns tasks 
to workers, facilitating efficient data processing in two distinct stages: 

1. **Single-Shard Single-Worker**: Each worker processes one dataset shard at a time.
2. **Single-Shard Multiple-Workers**: Multiple workers process the same shard, with distinct roles 
   for producers (feeding a queue) and consumers (processing from the queue).

The runner optimizes resource usage and performance by adapting to the workload dynamically, 
ensuring effective parallel processing throughout the data lifecycle.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection  # noqa: F401
import traceback
from copy import copy
from enum import Enum
from functools import partial
from itertools import chain
from typing import Any, Callable, Iterable, TypeAlias, TypeVar

import dill
import orjson
from datasets import IterableDataset
from datasets.iterable_dataset import (
    FilteredExamplesIterable,
    MappedExamplesIterable,
    TypedExamplesIterable,
    _BaseExamplesIterable,
)

from hyped.common._worker import manager as _manager  # noqa: F401
from hyped.common._worker import set_worker_info
from hyped.common.iterators import (
    QueueIterator,
    StoppableIterator,
    TimedIterator,
    batched,
    ith_entries,
)
from hyped.common.logging import get_cls_logger
from hyped.common.typing import Rank, Sample
from hyped.common.utils import clock, compose

from ..callbacks.base import CallbackManager
from ..monitor import ProgressMonitor, ProgressReport, TimeReport
from .base import BaseRunner, WorkerRole

T = TypeVar("T")
U = TypeVar("U")

IndexedSample: TypeAlias = tuple[str, Sample]

ContextTuple: TypeAlias = tuple[
    WorkerRole | None,
    Iterable[T] | None,
    Callable[[Iterable[T]], Iterable[U]] | None,
    Callable[[U], Any] | None,
    bool,
]
"""Type alias representing the context passed to workers.

This tuple defines the context used for setting up a worker's role, producer,
processing function, finalizer function, and a completion flag.

Elements:
    - WorkerRole | None: The role assigned to the worker (e.g., producer, processor),
      or None if the role remains unchanged.
    - Iterable[IndexedSample] | None: The producer for the worker, providing an iterable
      over indexed samples that the worker will process.
    - Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]] | None: The processing
      function that transforms the produced samples. It accepts an iterable of
      indexed samples and returns an iterable of transformed samples. If None,
      no processing function is applied.
    - Callable[[IndexedSample], Any] | None: The finalizer function applied to each
      individual sample after processing, or None if no finalizer is needed.
    - bool: A flag indicating whether the context application is complete and
      the worker should stop processing.
"""


class ExamplesIterablePipeline(list[_BaseExamplesIterable]):
    """Pipeline of huggingface's :class:`_BaseExamplesIterable` blocks.

    This class manages a sequence of processing steps applied to a dataset,
    allowing for chaining and copying of processing pipelines.
    """

    @property
    def src_iterable(self) -> None | _BaseExamplesIterable:
        """Get the source iterable for the pipeline.

        Returns:
            _BaseExamplesIterable: The source iterable for the pipeline.

        Raises:
            AssertionError: If the pipeline is empty.
        """
        assert len(self) > 0, "Pipeline is empty; no source iterable available."
        return self[0].ex_iterable

    def copy(self) -> ExamplesIterablePipeline:
        """Create a copy of the pipeline with each step copied.

        Returns:
            ExamplesIterablePipeline: A new pipeline instance with copied steps.
        """

        first = copy(self[0])
        first.ex_iterable = None

        pipeline = ExamplesIterablePipeline([first])

        for step in map(copy, self[1:]):
            step.ex_iterable = pipeline[-1]
            pipeline.append(step)

        return pipeline

    def __call__(self, ex_iterable: Iterable[IndexedSample]) -> Iterable[IndexedSample]:
        """Run the pipeline on the given example iterable.

        Args:
            ex_iterable (Iterable[IndexedSample]): The example iterable to be processed by the
                pipeline.

        Returns:
            Iterable[IndexedSample]: Processed samples from the pipeline.
        """

        pipeline = self.copy()
        pipeline[0].ex_iterable = ex_iterable

        yield from pipeline[-1]

    def __str__(self) -> str:
        """String representation of the pipeline."""
        return "[" + ", ".join([type(step).__name__ for step in self]) + "]"


class MessageType(Enum):
    """Enum representing the various worker message types."""

    READY = 1
    """
    Indicates that a worker is ready to start processing or receive tasks.
    """

    DONE = 2
    """
    Indicates that a worker has completed processing and terminates soon.
    """

    EXCEPTION = 3
    """
    Indicates that an exception occurred during worker processing.
    Used to signal errors or abnormal terminations in task execution.
    """

    CTX_REQUEST = 4
    """
    Requests a new context for the worker.
    """

    CTX_STARTED = 5
    """
    Signals that a worker has started processing a given context.
    """

    CTX_REPORT = 6
    """
    Provides status updates and progress reports for a worker's current context.
    """

    CTX_SWITCH = 7
    """
    Indicates that the worker is requesting to switch to a different context.
    """

    CTX_COMPLETE = 8
    """
    Signals that the worker has successfully completed its current context.
    """

    CTX_CANCELED = 9
    """
    Indicates that the current context has been canceled before completion.
    """


class Worker(mp.Process):
    """A worker process for parallel data processing.

    This class extends :code:`mp.Process` to handle data processing in a separate
    process, managing context and processing stages for data.
    """

    def __init__(
        self,
        rank: Rank,
        num_workers: int,
        send_msg_conn: mp.connection.Connection,
        progress_report_interval: float,
        worker_init: Callable[[], Any],
        worker_finalize: Callable[[], Any],
    ) -> None:
        """Initialize a worker process for parallel data processing.

        Args:
            rank (Rank): The rank or identifier for the worker, typically representing
                the worker's position or role in the set of workers.
            num_workers (int): The total number of workers involved in the data
                processing pipeline.
            req_ctx_conn (mp.connection.Connection): A connection object used to request
                request a new processing context from the manager process.
            tracker_conn (mp.connection.Connection): A connection object used to
                send updates to the progress tracker thread.
            tracker_update_interval (float): The time interval, in seconds, between sending
                progress updates to the tracker.
            worker_init (Callable[[], Any]): A callable function to initialize the worker's
                state before processing begins.
            worker_finalize (Callable[[], Any]): A callable function to finalize the worker's
                state after processing is complete.
        """
        super(Worker, self).__init__(daemon=True)

        self._progress_report_interval = progress_report_interval

        self._rank = rank
        self._num_workers = num_workers
        # connections
        self._send_msg_conn = send_msg_conn
        self._parent_ctx_conn, self._child_ctx_conn = mp.Pipe(duplex=True)
        # worker initializer and finalizer
        self._worker_init = worker_init
        self._worker_finalize = worker_finalize
        # pipeline to be executed by the worker
        self._role: WorkerRole | None = None
        self._producer: Iterable[IndexedSample] | None = None
        self._processor: Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]] | None = None
        self._finalizer: Callable[[IndexedSample], Any] | None = None

        # create logger
        self._logger = get_cls_logger(type(self))

        # Log initialization details
        self._logger.debug(
            f"Created worker with rank {self._rank} of {self._num_workers} total workers."
        )

    T = TypeVar("T")
    U = TypeVar("U")

    def send_ctx(
        self,
        role: WorkerRole | None,
        producer: Iterable[T] | None,
        processor: Callable[[Iterable[T]], Iterable[U]] | None,
        finalizer: Callable[[U], Any] | None,
        done: bool,
        *,
        blocking: bool = True,
    ) -> bool:
        """
        Send new processing context to the worker.

        Args:
            role (WorkerRole | None): The role of the worker.
            producer (Iterable[T] | None): The producer iterable for generating samples.
            processor (Callable[[Iterable[T]], Iterable[U]] | None): The
                processor function.
            finalizer (Callable[[U], Any] | None): The finalizer function.
            done (bool): Whether the context change is final.
            blocking (bool): Whether to block and wait for acceptance from the worker.

        Returns:
            bool: Boolean indicating whether the worker accepted the new context.
            If not blocking, it always returns True.
        """
        # clear connection
        while self._parent_ctx_conn.poll():
            self._parent_ctx_conn.recv()
        # send serialized context
        ctx = (role, producer, processor, finalizer, done)
        self._parent_ctx_conn.send_bytes(dill.dumps(ctx))

        if blocking:
            # wait for feedback from worker
            accepted = self._parent_ctx_conn.recv()
            self._logger.debug(
                f"Sent new context to worker {self._rank}, worker "
                f"{'accpeted' if accepted else 'refused'}."
            )
            return accepted
        else:
            self._logger.debug(f"Sent new context to worker {self._rank} in non-blocking mode.")
            return True

    def _send_msg(self, msg_type: MessageType, payload: None | Any = None) -> None:
        """Send a message from the worker to the manager process.

        This function serializes and sends a message to the manager process through the
        specified communication connection. The message includes the worker's rank,
        the type of message, and an optional payload.

        Args:
            msg_type (MessageType): The type of message to send, indicating the nature
                of the update or communication.
            payload (None | Any, optional): Additional data or context to be sent along with
                the message. Can be any serializable object. Default is `None`.
        """
        msg = {"rank": self._rank, "type": msg_type.value, "payload": payload}

        msg = orjson.dumps(msg)
        self._send_msg_conn.send_bytes(msg)

    def _request_new_ctx(self) -> bool:
        """Request new processing context from the main process.

        Returns:
            bool: Whether the worker has been instructed to stop.
        """

        # request new context from main process
        self._logger.debug("Requesting new context from main process.")
        self._send_msg(MessageType.CTX_REQUEST)

        # wait for new context to be received
        ctx = self._recv_ctx()

        _, prod, _, _, done = ctx
        if (prod is None) and not done:
            # not accepted, expected new producer
            self._child_ctx_conn.send(False)
            return self._request_new_ctx()

        return self._apply_ctx(ctx)

    def _recv_ctx(self) -> ContextTuple:
        """Receive and update the processing context.

        This method receives a new processing context from the worker's connection,
        updating the worker's pipeline with the new producer, processor, and finalizer
        functions. If :code:`keep_producer` is :code:`True`, the current producer is
        preserved, and only the processing stages are updated.

        Returns:
            ContextTuple: A tuple containing the received context.
        """
        # wait for new context
        while not self._child_ctx_conn.poll(timeout=1.0):
            self._logger.debug("Waiting for new context...")

        # receive context
        while self._child_ctx_conn.poll():
            ctx = self._child_ctx_conn.recv_bytes()
        # deserialize context
        ctx = dill.loads(ctx)
        self._logger.debug("Received new context.")

        return ctx

    def _apply_ctx(self, ctx: ContextTuple) -> bool:
        """Apply context to the worker.

        This method updates the worker's internal attributes based on the provided context.
        Each context element (role, producer, processor, finalizer) is conditionally applied,
        meaning if the element is :code:`None`, the corresponding worker attribute remains
        unchanged.

        The method also triggers an event to signal that the context has been successfully received
        and applied. Additionally, it logs the applied role for debugging purposes.

        Returns:
            bool: The :code:`done` flag, which indicates if the process using this context should
            be completed.
        """
        # unpack the context
        role, prod, proc, fn, done = ctx
        # apply context to worker
        self._role = role if role is not None else self._role
        self._producer = prod if prod is not None else self._producer
        self._processor = proc if proc is not None else self._processor
        self._finalizer = fn if fn is not None else self._finalizer
        # log
        self._logger.debug(
            f"Applied new context with role "
            f"`{self._role.name if self._role is not None else None}`."
        )

        # accepted
        self._child_ctx_conn.send(True)

        return done

    def run(self) -> None:
        """Start the worker process.

        Continuously request new contexts, process data with the current context, and
        apply new contexts as needed until instructed to stop.
        """
        # TODO: set seed

        # set worker info
        set_worker_info(
            rank=self._rank,
            num_workers=self._num_workers,
            seed=None,
        )

        try:
            # initialize worker
            self._worker_init()
            self._send_msg(MessageType.READY)
            self._logger.info("Initialization complete.")

            done = False
            # request a processing context
            while (not done) and (not self._request_new_ctx()):
                # create producer iterator to avoid resetting when
                # new context is received during execution
                producer_iter = iter(self._producer)
                producer_exhausted = False

                # monitor producer
                timed_producer_iter = TimedIterator(producer_iter, smoothing=0.1)

                # exhaust producer
                while not producer_exhausted:
                    self._send_msg(MessageType.CTX_STARTED, payload=self._role.value)
                    self._logger.debug(f"Starting processing with role {self._role}.")

                    num_samples = 0
                    last_report = clock()

                    # create a stoppable producer that allows to dynamically
                    # interrupt the execution and apply the new context
                    stoppable_producer = StoppableIterator(timed_producer_iter)

                    try:
                        # apply the processor to the producer
                        samples_iter = self._processor(stoppable_producer)
                        timed_samples_iter = TimedIterator(samples_iter, smoothing=0.1)
                        # apply the finalizer function to each sample
                        work_iter = map(self._finalizer, timed_samples_iter)
                        timed_work_iter = TimedIterator(work_iter, smoothing=0.1)

                        def _report_progress(now: float, num_samples: int, last_report: float):
                            payload = ProgressReport(
                                timestamp=now,
                                elapsed_time=now - last_report,
                                num_samples=num_samples,
                                average_time=TimeReport(
                                    producer=timed_producer_iter.smooth_time(),
                                    processor=timed_samples_iter.smooth_time(),
                                    finalizer=timed_work_iter.smooth_time(),
                                ),
                                total_time=TimeReport(
                                    producer=timed_producer_iter.total_time(),
                                    processor=timed_samples_iter.total_time(),
                                    finalizer=timed_work_iter.total_time(),
                                ),
                            )
                            self._send_msg(MessageType.CTX_REPORT, payload=payload)

                        # main worker loop
                        for _ in timed_work_iter:
                            num_samples += 1

                            # check if the worker was asked to apply a new context
                            if self._child_ctx_conn.poll():
                                self._logger.debug("Detected context update request.")

                                # receive and unpack context
                                ctx = self._recv_ctx()
                                new_role, prod, _, _, done = ctx

                                if prod is not None:
                                    # new context not accepted
                                    self._child_ctx_conn.send(False)

                                else:
                                    # the producer is not allowed to change
                                    assert prod is None, "Unexpected producer received."

                                    if done:
                                        # stop worker
                                        self._logger.info("Received 'done' signal, stopping.")
                                        raise StopIteration()

                                    # stop the producer from generating further samples
                                    # and exhaust the current samples generated by the producer
                                    stoppable_producer.stop()
                                    for _ in timed_work_iter:
                                        num_samples += 1

                                    # send progress report before switching the context
                                    _report_progress(clock(), num_samples, last_report)
                                    self._send_msg(
                                        MessageType.CTX_SWITCH,
                                        payload=(self._role.value, new_role.value),
                                    )
                                    self._apply_ctx(ctx)
                                    # recreate the work iterable
                                    break

                            now = clock()
                            # send continuous updates to tracker
                            if (num_samples > 0) and (
                                now - last_report > self._progress_report_interval
                            ):
                                # report progress report and reset tracking values
                                _report_progress(now, num_samples, last_report)
                                num_samples, last_report = 0, now

                        else:
                            # producer exhausted
                            producer_exhausted = True
                            self._logger.info("Finished processing current context.")
                            # send final progress update and completion message
                            _report_progress(clock(), num_samples, last_report)
                            self._send_msg(MessageType.CTX_COMPLETE)

                    except StopIteration:
                        # catch stop execution error
                        self._send_msg(MessageType.CTX_CANCELED)
                        producer_exhausted = True
                        done = True

                    except KeyboardInterrupt:
                        self._send_msg(MessageType.CTX_CANCELED)
                        raise

                    except Exception as e:
                        # gracefully handle exceptions without stopping the worker
                        self._logger.error(
                            f"Unexpected error during processing: {str(e)}.", exc_info=True
                        )
                        self._send_msg(
                            MessageType.EXCEPTION,
                            payload={
                                "error_type": str(type(e).__name__),
                                "error_message": str(e),
                                "stack_trace": traceback.format_exc(),
                            },
                        )

        except KeyboardInterrupt:
            self._logger.warning("Worker interrupted by user.")

        except Exception as e:
            # gracefully handle exception
            self._logger.error(f"Unexpected error during processing: {str(e)}.", exc_info=True)
            self._send_msg(
                MessageType.EXCEPTION,
                payload={
                    "error_type": str(type(e).__name__),
                    "error_message": str(e),
                    "stack_trace": traceback.format_exc(),
                },
            )

        finally:
            try:
                # finalize worker
                self._worker_finalize()
                self._logger.info("Worker finalized successfully.")
            except Exception as e:
                # log exception
                self._logger.error(f"Error finalizing worker: {str(e)}.", exc_info=True)
                self._send_msg(
                    MessageType.EXCEPTION,
                    payload={
                        "error_type": str(type(e).__name__),
                        "error_message": str(e),
                        "stack_trace": traceback.format_exc(),
                    },
                )

        # send done message
        self._send_msg(MessageType.DONE)


class Serializer(object):

    """Serializer applied in multiprocessing runner stage 2.

    This serializer collects and serialized a batch of samples into a single
    serialized element using :code:`orjson` to reduce latency. From our experiments
    :code:`orjson` showed to be the fastest serializer for json-compatible objects.
    """

    def __init__(self, batch_size: int):
        """Initialize a new serializer.

        Args:
            batch_size (int): The number of batches to pack together.
        """
        self.batch_size = batch_size

    def serialize(self, it: Iterable[Sample]) -> Iterable[Any]:
        """Serialization wrapper.

        Args:
            it (Iterable[Sample]): An iterable of samples to be serialized.

        Returns:
            Iterable[Any]: An iterable containing the serialized samples,
            where each sample is serialized to a byte format using orjson.dumps.

        """
        return map(orjson.dumps, batched(it, n=self.batch_size))

    def deserialize(self, it: Iterable[Any]) -> Iterable[Sample]:
        """Deserializer wrapper.

        Args:
            it (Iterable[Any]): An iterable of serialized data (e.g., bytes)
            to be deserialized.

        Returns:
            Iterable[Sample]: An iterable of samples, where each serialized
            data is deserialized using orjson.loads.
        """
        return chain.from_iterable(map(orjson.loads, it))


class WorkerController(object):
    """Controller for managing worker processes and their roles.

    Handles the assignment of workers to different roles (processors,
    consumers, and producers) and manages task execution and worker
    state transitions.
    """

    def __init__(self, workers: list[Worker], serializer: Serializer) -> None:
        """Initializes the WorkerController with the provided workers and serializer.

        Args:
            workers (list[Worker]): A list of workers to be controlled.
            serializer (Serializer): A serializer instance for managing data formats.
        """
        global _manager

        self.workers = workers
        self.serializer = serializer
        self.queue = _manager.Queue(maxsize=self.num_workers)
        self.queue_it = QueueIterator(self.queue, sentinel=None, timeout=1.0)
        self.processor_ranks = set()
        self.producer_ranks = set()
        self.consumer_ranks = set()
        self.joined_ranks = set()
        self._logger = get_cls_logger(type(self))

    @property
    def num_workers(self) -> int:
        """Returns the number of workers being managed.

        Returns:
            int: Number of workers.
        """
        return len(self.workers)

    @property
    def any_producers(self) -> bool:
        """Checks if there are any workers assigned as producers.

        Returns:
            bool: True if any workers are assigned to the producer role, False otherwise.
        """
        return len(self.producer_ranks) > 0

    def start(self) -> None:
        """Starts all the worker processes. Logs the start of each worker."""
        for worker in self.workers:
            worker.start()

        self._logger.info("All workers started.")

    def create_processor(
        self,
        rank: Rank,
        shard: Iterable,
        processor: Callable[[Iterable], Iterable] | None,
        fn: Callable[[Any], Any] | None,
    ) -> None:
        """Assigns a worker the role of processor and provides the processing context.

        Args:
            rank (Rank): The rank of the worker to assign.
            shard (Iterable): The data shard to be processed.
            processor (Callable[[Iterable], Iterable] | None): The processing function.
            fn (Callable[[Any], Any] | None): An additional function to be applied.
        """
        self.workers[rank].send_ctx(
            WorkerRole.PROCESSOR, shard, processor, fn, False, blocking=False
        )
        self.processor_ranks.add(rank)
        self._logger.info(f"Assigned worker {rank} as processor.")

    def create_consumer(
        self,
        rank: Rank,
        processor: Callable[[Iterable], Iterable] | None,
        fn: Callable[[Any], Any] | None,
    ) -> None:
        """Assigns a worker the role of consumer and provides the consumer context.

        Args:
            rank (Rank): The rank of the worker to assign.
            processor (Callable[[Iterable], Iterable] | None): The processing function applied
                before consuming.
            fn (Callable[[Any], Any] | None): An additional function to be applied.
        """
        self.workers[rank].send_ctx(
            WorkerRole.CONSUMER,
            self.queue_it,
            compose(processor, self.serializer.deserialize),
            fn,
            False,
            blocking=False,
        )
        self.consumer_ranks.add(rank)
        self._logger.info(f"Assigned worker {rank} as consumer.")

    def try_switch_processor_to_producer(self) -> Rank | None:
        """Attempts to switch a processor to a producer role.

        If successful, the processor rank is removed from the processor set and added to the
        producer set.

        Returns:
            Rank | None: The rank of the worker if the switch is successful, None otherwise.
        """
        ctx = (WorkerRole.PRODUCER, None, self.serializer.serialize, self.queue.put, False)
        for rank in self.processor_ranks:
            if self.workers[rank].send_ctx(*ctx, blocking=True):
                self.processor_ranks.remove(rank)
                self.producer_ranks.add(rank)
                self._logger.info(f"Assigned worker {rank} as producer.")
                return rank
            self._logger.info(f"Worker {rank} did not accept producer context.")

    def try_switch_producer_to_processor(
        self, processor: Callable[[Iterable], Iterable] | None, fn: Callable[[Any], Any] | None
    ) -> Rank | None:
        """Attempts to switch a producer to a processor role.

        If successful, the producer rank is removed from the producer set and added to the
        processor set.

        Args:
            processor (Callable[[Iterable], Iterable] | None): The processing function.
            fn (Callable[[Any], Any] | None): An additional function to be applied.

        Returns:
            Rank | None: The rank of the worker if the switch is successful, None otherwise.
        """
        ctx = (WorkerRole.PROCESSOR, None, processor, fn, False)
        for rank in self.producer_ranks:
            if self.workers[rank].send_ctx(*ctx, blocking=True):
                self.producer_ranks.remove(rank)
                self.processor_ranks.add(rank)
                self._logger.info(f"Assigned worker {rank} as processor.")
                return rank
            self._logger.info(f"Worker {rank} did not accept processor context.")

    def free_worker(self, rank: Rank) -> None:
        """Removes the worker from any active roles.

        Args:
            rank (Rank): The rank of the worker to free.
        """
        self.processor_ranks -= {rank}
        self.producer_ranks -= {rank}
        self.consumer_ranks -= {rank}

    def stop_worker(self, rank: Rank) -> None:
        """Sends a stop signal to a worker, indicating that it should cease operation.

        Args:
            rank (Rank): The rank of the worker to stop.
        """
        self.workers[rank].send_ctx(None, None, None, None, True, blocking=False)

    def join_worker(self, rank: Rank) -> None:
        """Waits for a worker to complete execution and join the main thread.

        Ensures the worker is no longer performing any roles.

        Args:
            rank (Rank): The rank of the worker to join.
        """
        assert rank not in self.processor_ranks
        assert rank not in self.producer_ranks
        assert rank not in self.consumer_ranks
        self.workers[rank].join()
        self.joined_ranks.add(rank)

    def assert_all_workers_joined(self) -> None:
        """
        Asserts that all workers have completed execution and joined the main thread.
        """
        assert len(self.joined_ranks) == len(self.workers)


class ConsumerProducerBalancer(object):
    class Action(Enum):
        NO_ACTION = 1
        ADD_PRODUCER = 2
        REMOVE_PRODUCER = 3

    def __init__(self, controller: WorkerController, monitor: ProgressMonitor) -> None:
        self._controller = controller
        self._monitor = monitor

    def callback(self) -> Action:
        # get the average time blocked for queue get operation
        registered_consumer_workers = self._monitor.get_workers_with_role(WorkerRole.CONSUMER)
        get_queue_avg_time = self._monitor.avg_times(registered_consumer_workers)["producer"]
        # get the average time blocked for queue put operation
        registered_producer_workers = self._monitor.get_workers_with_role(WorkerRole.PRODUCER)
        put_queue_avg_time = self._monitor.avg_times(registered_producer_workers)["finalizer"]
        # compare put and get operation to see if there is a excess
        # of producers or consumers
        if get_queue_avg_time >= 1.3 * put_queue_avg_time:
            # get operations take longer than put operations
            # queue get operation blocks because its empty
            return ConsumerProducerBalancer.Action.ADD_PRODUCER

        elif put_queue_avg_time >= 1.3 * get_queue_avg_time:
            # put operations take longer than get operations
            # queue put operation blocks because its full
            return ConsumerProducerBalancer.Action.REMOVE_PRODUCER

        return ConsumerProducerBalancer.Action.NO_ACTION


class DynamicMultiprocessingRunner(BaseRunner):
    """Manages and runs a set of worker processes to handle parallel data processing.

    This class coordinates multiple worker processes to process data in parallel,
    dynamically assigning tasks and handling context changes.

    The processing is carried out in two distinct stages:

    - **Stage 1: Single-Shard Single-Worker**
      Each worker is assigned one dataset shard at a time, with minimal communication overhead,
      maximizing throughput by keeping the workers busy with their assigned tasks.

    - **Stage 2: Single-Shard Multiple-Workers**
      After all shards are assigned, the system transitions into this stage, where multiple
      workers process the same shard, dividing the roles into producers (feeding a queue) and
      consumers (processing data from the queue).

    By transitioning to Stage 2, the system ensures efficient and parallel processing of data,
    optimizing performance and resource usage.
    """

    def __init__(
        self,
        num_workers: int,
        prefetch_factor: int,
        worker_init: Callable[[], Any],
        worker_finalize: Callable[[], Any],
        progress_report_interval: float,
        callback: CallbackManager,
    ) -> None:
        """Initialize the multiprocessing runner.

        Args:
            num_workers (int): The number of worker processes to create for parallel data
                processing.
            prefetch_factor (int): The number of items per worker that should be prefetched in
                Stage 2 when using a queue.
            worker_init (Callable[[], Any]): A callable that will be invoked to initialize each
                worker. This function will run before the worker starts processing data.
            worker_finalize (Callable[[], Any]): A callable that will be invoked to finalize each
                worker. This function will run after the worker has finished processing all data.
            callback (CallbackManager): A callback manager that will be invoked at various points
                during the data processing lifecycle.
        """

        self._num_workers = num_workers
        self._prefetch = prefetch_factor

        self._worker_init = worker_init
        self._worker_finalize = worker_finalize

        self._report_interval = progress_report_interval
        self._callback = callback

        self._logger = get_cls_logger(type(self))

    def _prepare_dataset(
        self, ds: IterableDataset
    ) -> tuple[_BaseExamplesIterable, Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]]]:
        """Prepare the dataset for processing by separating processing steps.

        Args:
            ds (IterableDataset): The dataset to prepare.

        Returns:
            tuple[IterableDataset, Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]]]:
            A tuple containing the source dataset and a processor function representing the lazy
            operations applied to the dataset.
        """

        if not hasattr(ds, "_ex_iterable") or not isinstance(
            ds._ex_iterable,
            (
                MappedExamplesIterable,
                FilteredExamplesIterable,
                TypedExamplesIterable,
            ),
        ):
            return ds._prepare_ex_iterable_for_iteration(batch_size=self._prefetch), partial(
                ith_entries, i=1
            )

        # collect all processing steps to separate off
        pipeline = ExamplesIterablePipeline([ds._ex_iterable])
        while isinstance(
            pipeline.src_iterable,
            (
                MappedExamplesIterable,
                FilteredExamplesIterable,
                TypedExamplesIterable,
            ),
        ):
            pipeline.insert(0, pipeline.src_iterable)

        self._logger.info(
            f"Separated {len(pipeline)} processing steps from iterable dataset: {str(pipeline)}"
        )

        # create the source dataaset that excludes the pipeline processing steps
        src_ds = IterableDataset(ex_iterable=pipeline.src_iterable)
        ex_iterable = src_ds._prepare_ex_iterable_for_iteration(batch_size=self._prefetch)
        # pipeline iterator yields (key, sample)-tuples, drop the key
        processor = compose(partial(ith_entries, i=1), pipeline.copy())

        return ex_iterable, processor

    def run(self, ds: IterableDataset, fn: Callable[[Sample], Any]) -> None:
        """Execute data processing using the worker processes.

        Args:
            ds: (IterableDataset): The dataset to process.
            fn (Callable[[Sample], Any]): The function to apply to each sample in the dataset.
        """

        self._logger.info("Starting data processing.")

        # prepare the dataset
        src_ds, processor = self._prepare_dataset(ds)
        self._logger.info(f"Dataset prepared with {src_ds.n_shards} shards.")

        # create connection for workers to request new context
        recv_msg_conn, worker_msg_conn = mp.Pipe(duplex=False)
        # create all workers
        workers = [
            Worker(
                rank=rank,
                num_workers=self._num_workers,
                send_msg_conn=worker_msg_conn,
                progress_report_interval=self._report_interval,
                worker_init=self._worker_init,
                worker_finalize=self._worker_finalize,
            )
            for rank in range(self._num_workers)
        ]

        # create the serializer used to serialize samples
        # before putting them into the queue
        serializer = Serializer(batch_size=self._prefetch)
        # create controller
        controller = WorkerController(workers, serializer)
        controller.start()

        # create the progress monitor, note that the serializer dumps a batch of samples
        # into a single queue element with a batch size set to the prefetch factor
        monitor = ProgressMonitor(
            src_ds.n_shards, self._num_workers, controller.queue, controller.serializer.batch_size
        )

        # create the consumer producer balancer
        balancer = ConsumerProducerBalancer(controller, monitor)

        # run callbacks
        self._callback.on_start(monitor, ds)

        # mark a specific worker as switching
        # used to rate limit the context switches of workers
        switching_worker: None | Rank = None
        last_switch = clock()

        done = False
        while not done:
            # receive message from worker
            msg = recv_msg_conn.recv_bytes()
            msg = orjson.loads(msg)
            # unpack message
            rank: Rank = msg["rank"]
            msg_type = MessageType(msg["type"])
            payload = msg["payload"]

            # handle message
            if msg_type is MessageType.READY:
                monitor._mark_worker_ready(rank)
                self._logger.debug(f"Worker {rank} ready.")

            elif msg_type is MessageType.DONE:
                controller.join_worker(rank)
                monitor._mark_worker_done(rank)
                self._logger.debug(f"Worker {rank} done.")
                # only keep going if there are any workers left
                done = not monitor.any_worker_alive

            elif msg_type is MessageType.CTX_STARTED:
                # update monitor state
                role = WorkerRole(payload)
                monitor._mark_worker_busy(rank, role)
                # log
                self._logger.info(f"Worker {rank} started running role {role.name}.")

                if rank == switching_worker:
                    # reset switching worker
                    switching_worker = None

            elif msg_type is MessageType.CTX_COMPLETE:
                if monitor.get_worker_role(rank) is WorkerRole.PRODUCER:
                    # try to start another producer shard to replace this one
                    controller.try_switch_processor_to_producer()

                # get the shard that was processed by the worker
                shard_id = monitor.get_worker_shard(rank)
                # update the controller and monitor state
                controller.free_worker(rank)
                monitor._mark_worker_completed(rank)
                monitor._mark_worker_idling(rank)

                if shard_id is not None:
                    # run the callback
                    self._callback.on_shard_completed(monitor, shard_id)

            elif msg_type is MessageType.CTX_CANCELED:
                if monitor.get_worker_role(rank) is WorkerRole.PRODUCER:
                    # try to start another producer shard to replace this one
                    controller.try_switch_processor_to_producer()

                # get the shard that was processed by the worker
                shard_id = monitor.get_worker_shard(rank)
                # update the controller and monitor state
                controller.free_worker(rank)
                monitor._mark_worker_canceled(rank)
                monitor._mark_worker_idling(rank)

                if shard_id is not None:
                    # run the callback
                    self._callback.on_shard_canceled(monitor, shard_id)

            elif msg_type is MessageType.CTX_SWITCH:
                # parse payload
                old_role, new_role = payload
                old_role, new_role = WorkerRole(old_role), WorkerRole(new_role)
                # update monitor state
                monitor._mark_worker_idling(rank)
                monitor._mark_worker_busy(rank, new_role)
                # log
                self._logger.info(
                    f"Worker {rank} switched context from {old_role.name} to {new_role.name}"
                )

            elif msg_type is MessageType.CTX_REPORT:
                monitor._report_progress(rank, report=payload)

                now = clock()
                if (switching_worker is None) and (now - last_switch > 5):
                    # call balancer whenever there is a progress report update
                    action = balancer.callback()

                    if action is ConsumerProducerBalancer.Action.ADD_PRODUCER:
                        # try to convert an active processor to a producer
                        switching_worker = self._controller.try_switch_processor_to_producer()
                        last_switch = now

                    elif (action is ConsumerProducerBalancer.Action.REMOVE_PRODUCER) and (
                        len(self._controller.producer_ranks) > 1
                    ):
                        # try to convert an active producer back to a processor
                        switching_worker = self._controller.try_switch_producer_to_processor(
                            processor, fn
                        )
                        last_switch = now

            elif msg_type is MessageType.CTX_REQUEST:
                # worker must be idling
                assert rank in monitor.alive_workers
                assert rank in monitor.idle_workers

                if monitor.is_stopping:
                    # send stop singal
                    controller.stop_worker(rank)

                elif monitor.any_pending_shards:
                    # Stage 1
                    shard_id = monitor.pending_shards.pop()
                    # get shard and send processor context to worker
                    shard = src_ds.shard_data_sources(shard_id, src_ds.n_shards)
                    controller.create_processor(rank, shard, processor, fn)
                    # run callback
                    self._callback.on_shard_in_progress(monitor, shard_id)

                    # mark shard as assigned to worker
                    monitor._mark_shard_in_progress(rank, shard_id)
                    self._logger.info(f"Assigned shard {shard_id} to worker {rank}.")

                else:
                    # Stage 2

                    # check if there is a producer
                    if not controller.any_producers:
                        controller.try_switch_processor_to_producer()

                    # assign worker as consumer
                    controller.create_consumer(rank, processor, fn)

                    # evenutally all workers are consumers
                    if monitor.alive_workers == controller.consumer_ranks:
                        # this is the signal that gracefully stops the workers
                        monitor._mark_as_stopping()
                        self._callback.on_stopping(monitor)
                        self._logger.info("Stopping criteria reached, gracefully stopping workers.")

        # shutdown
        monitor._mark_as_done()
        self._callback.on_done(monitor)
        controller.assert_all_workers_joined()

        self._logger.info("Runner shutdown complete.")

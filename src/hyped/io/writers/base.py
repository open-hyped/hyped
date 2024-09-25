"""Base Module for dynamic multiprocessing and dataset consumption.

This module provides high-level functionality for processing large datasets using a dynamic
multiprocessing system. It includes utilities for managing worker processes, tracking progress,
and implementing custom dataset writers.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import multiprocessing.connection  # noqa: F401
import os
import shutil
import threading
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from copy import copy
from dataclasses import asdict
from enum import Enum
from functools import partial
from itertools import chain, count
from time import time
from typing import Any, Callable, Iterable, TypeAlias, TypeVar

import datasets
import dill
import orjson
from datasets import Dataset, DatasetDict, DatasetInfo, IterableDataset, IterableDatasetDict
from datasets.iterable_dataset import (
    FilteredExamplesIterable,
    MappedExamplesIterable,
    TypedExamplesIterable,
    _BaseExamplesIterable,
)
from tqdm.auto import tqdm
from tqdm.std import EMA

from hyped.common._worker import manager as _manager  # noqa: F401
from hyped.common._worker import reset_worker_info, set_worker_info
from hyped.common.feature_key import FeatureKey
from hyped.common.iterators import (
    BatchBuffer,
    QueueIterator,
    StoppableIterator,
    TimedIterator,
    batched,
    ith_entries,
)
from hyped.common.logging import Logger, get_cls_logger
from hyped.common.typing import DatasetType, Rank, Sample
from hyped.common.utils import TimeWeightedEMA, chdir, compose, run_all

IndexedSample: TypeAlias = tuple[str, Sample]


T = TypeVar("T")


def _do_nothing():
    """A no-operation function that returns nothing."""
    return


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


T = TypeVar("T")
U = TypeVar("U")

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
    READY = 1
    DONE = 2
    EXCEPTION = 3
    CTX_REQUEST = 4
    CTX_STARTED = 5  # TODO: rename to CTX_RUNNING
    CTX_REPORT = 6
    CTX_SWITCH = 7
    CTX_COMPLETE = 8
    CTX_CANCELED = 9


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
        progress_report_interval: float = 0.1,
        worker_init: Callable[[], Any] = _do_nothing,
        worker_finalize: Callable[[], Any] = _do_nothing,
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
                progress updates to the tracker. Defaults to 0.1.
            worker_init (Callable[[], Any]): A callable function to initialize the worker's
                state before processing begins. Defaults to :func:`do_nothing`.
            worker_finalize (Callable[[], Any]): A callable function to finalize the worker's
                state after processing is complete. Defaults to :func:`do_nothing`.
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
                    last_report = time()

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

                        def _report_progress(num_samples: int, last_report: float):
                            payload = {
                                "num_samples": num_samples,
                                "elapsed_time": time() - last_report,
                                "timestamp": time(),
                                "average_time": {
                                    "producer": timed_producer_iter.average_time(),
                                    "processor": timed_samples_iter.average_time(),
                                    "finalizer": timed_work_iter.average_time(),
                                },
                                "total_time": {
                                    "producer": timed_producer_iter.total_time(),
                                    "processor": timed_samples_iter.total_time(),
                                    "finalizer": timed_work_iter.total_time(),
                                },
                            }
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
                                    _report_progress(num_samples, last_report)
                                    self._send_msg(
                                        MessageType.CTX_SWITCH,
                                        payload=(self._role.value, new_role.value),
                                    )
                                    self._apply_ctx(ctx)
                                    # recreate the work iterable
                                    break

                            # send continuous updates to tracker
                            if (num_samples > 0) and (
                                time() - last_report > self._progress_report_interval
                            ):
                                # report progress report and reset tracking values
                                _report_progress(num_samples, last_report)
                                num_samples, last_report = 0, time()

                        else:
                            # producer exhausted
                            producer_exhausted = True
                            self._logger.info("Finished processing current context.")
                            # send final progress update and completion message
                            _report_progress(num_samples, last_report)
                            self._send_msg(MessageType.CTX_COMPLETE)

                    except StopIteration:
                        # catch stop execution error
                        self._send_msg(MessageType.CTX_CANCELED)
                        producer_exhausted = True
                        done = True

                    except KeyboardInterrupt:  # pragma: not covered
                        self._logger.warning("Worker interrupted by user.")
                        self._send_msg(MessageType.CTX_CANCELED)
                        producer_exhausted = True
                        done = True

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

        except KeyboardInterrupt:  # pragma: not covered
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


class TqdmReporter(threading.Thread):
    """Tqdm Reporter Thread.

    A thread that reports progress using the :func:`tqdm` progress bar, based on the state of a
    :class:`ProgressTracker`.
    """

    def __init__(self, monitor: ProgressMonitor, update_interval: float = 0.1) -> None:
        """Initializes the :class:`TqdmReporter` thread.

        Args:
            tracker (ProgressTracker): The :class:`ProgressTracker` instance to track progress.
            update_interval (float): The interval in seconds between progress updates. Defaults
                to 0.1.
        """
        super(TqdmReporter, self).__init__(daemon=True)
        self._monitor = monitor
        self._update_interval = update_interval
        self._logger = get_cls_logger(type(self))

    @property
    def _pbar_desc(self) -> str:
        """Description displayed in tqdm bar.

        Returns a formatted description of the progress bar, including the current number
        of busy workers and their roles (processor, producer, consumer).

        Returns:
            str: A string describing the progress bar status.
        """
        counts = Counter(self._monitor._roles)
        return (
            f"Workers {len(self._monitor.alive_workers)}/{self._monitor.num_workers} "
            f"(S={counts[WorkerRole.PROCESSOR]}, "
            f"P={counts[WorkerRole.PRODUCER]}, "
            f"C={counts[WorkerRole.CONSUMER]})"
        )

    def run(self) -> None:
        """Thread entrypoint.

        The main loop of the :class:`TqdmReporter` thread. Continuously updates the progress bar
        based on the state of the :class:`ProgressTracker` until the tracker is done. After
        completion, performs one final update to ensure the progress bar reflects the latest state.
        """

        self._logger.info("Thread started.")

        ema_dn = EMA(smoothing=0.3)
        ema_dt = EMA(smoothing=0.3)

        with tqdm(total=self._monitor._num_shards, desc=self._pbar_desc) as pbar:
            self._logger.debug(
                f"Initialized tqdm progress bar with {self._monitor.num_shards} total shards."
            )

            prev_total_samples = 0
            prev_update_time = pbar._time()

            def _iter():
                # iterate as long as the tracker is running
                while not self._monitor._done.wait(timeout=self._update_interval):
                    yield
                # do a final update after the tracker finished
                yield

            for _ in _iter():
                # get current state
                total_samples = self._monitor.num_samples_processed
                update_time = pbar._time()

                # compute sample throughput
                dn = total_samples - prev_total_samples
                dt = update_time - prev_update_time
                throughput = ema_dn(dn) / max(ema_dt(dt), 1e-5)

                self._logger.debug(
                    f"Updating progress: Total samples {total_samples}, "
                    f"Throughput {throughput:.02f}ex/s"
                )

                # format total samples
                formatting_string = "%d" if total_samples < 10**6 else "%.2e"
                formatted_total_samples = formatting_string % total_samples
                # update progress bar
                pbar.set_description(self._pbar_desc, refresh=False)
                pbar.set_postfix_str(
                    (
                        f"{self._monitor.num_buffered_samples}q, "
                        f"{throughput:.02f}ex/s, "
                        f"{formatted_total_samples}ex"
                    ),
                    refresh=True,
                )
                # update values
                prev_total_samples = total_samples
                prev_update_time = update_time

                # update the progress bar
                pbar.update(len(self._monitor.completed_shards) - pbar.n)

        self._logger.info("Thread finished.")


class ShardState(Enum):
    PENDING = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    CANCELED = 3


class ProgressMonitor(object):
    def __init__(self, num_shards: int, num_workers: int, queue: mp.Queue, item_size: int) -> None:
        """Initializes the ProgressMonitor.

        Args:
            num_shards (int): The total number of shards to process.
            num_workers (int): The number of workers processing the data.
            queue (mp.Queue): Sample queue filled by producer in stage 2.
            item_size (int): The number of samples contained within a queue item.
        """

        self._stopping = threading.Event()
        self._done = threading.Event()
        # track buffer queue
        self._queue = queue
        self._item_size = item_size
        # track shards
        self._num_shards = num_shards
        self._shard_state = [ShardState.PENDING] * num_shards
        # track workers
        self._num_workers = num_workers
        self._alive = [False] * num_workers
        self._roles = [None] * num_workers
        self._shard = [None] * num_workers

        decay_rate = math.log(2) / 1  # 1s half-life time
        # track samples generated by each worker in different roles
        self._time_ema = [
            {
                "producer": TimeWeightedEMA(decay_rate),
                "processor": TimeWeightedEMA(decay_rate),
                "finalizer": TimeWeightedEMA(decay_rate),
            }
            for _ in range(num_workers)
        ]
        self._num_samples = [
            {
                WorkerRole.PROCESSOR: 0,
                WorkerRole.PRODUCER: 0,
                WorkerRole.CONSUMER: 0,
            }
            for _ in range(num_workers)
        ]

    def get_worker_role(self, rank: Rank) -> None | WorkerRole:
        return self._roles[rank]

    @property
    def num_samples_processed(self) -> int:
        return sum(
            nums[WorkerRole.PROCESSOR] + nums[WorkerRole.CONSUMER] for nums in self._num_samples
        )

    def _report_progress(self, rank: Rank, report: dict[str, Any]) -> None:
        ts = report["timestamp"]
        # save worker report
        self._time_ema[rank]["producer"].update(ts, report["average_time"]["producer"])
        self._time_ema[rank]["processor"].update(ts, report["average_time"]["processor"])
        self._time_ema[rank]["finalizer"].update(ts, report["average_time"]["finalizer"])
        # update processed samples
        role = self._roles[rank]
        self._num_samples[rank][role] += report["num_samples"]

    def avg_times(self, ranks: None | Iterable[Rank] = None) -> dict[str, float]:
        # sort ranks by timestamp
        ranks = ranks if ranks is not None else range(self.num_workers)
        ranks = [r for r in ranks if self._time_ema[r]["producer"].timestamp is not None]
        ranks = sorted(ranks, key=lambda r: self._time_ema[r]["producer"].timestamp)

        times = {}
        for key in ["producer", "processor", "finalizer"]:
            global_ema = TimeWeightedEMA(decay_rate=math.log(2) / 1)

            for rank in ranks:
                local_ema = self._time_ema[rank][key]
                global_ema.update(local_ema.timestamp, local_ema.value)

            times[key] = global_ema.value

        return times

    @property
    def num_shards(self) -> int:
        return self._num_shards

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def num_buffered_samples(self) -> int:
        """Returns the number of samples currently buffered in the queue awaiting processing.

        This property calculates the total number of samples that are waiting in the queue by
        multiplying the number of items in the queue by the item size (the number of samples
        contained in each item).

        Returns:
            int: The total number of samples currently buffered in the queue.
        """
        try:
            return self._queue.qsize() * self._item_size
        except (BrokenPipeError, ConnectionResetError):
            # queue connection closed
            return 0

    def _mark_as_stopping(self) -> None:
        self._stopping.set()

    def _mark_as_done(self) -> None:
        self._done.set()

    @property
    def is_stopping(self) -> bool:
        return self._stopping.is_set()

    def _mark_shard_in_progress(self, rank: Rank, shard_id: int) -> None:
        self._shard_state[shard_id] = ShardState.IN_PROGRESS
        self._shard[rank] = shard_id

    def _mark_shard_completed(self, shard_id: int) -> None:
        self._shard_state[shard_id] = ShardState.COMPLETED

    def _mark_shard_canceled(self, shard_id: int) -> None:
        self._shard_state[shard_id] = ShardState.CANCELED

    @property
    def any_pending_shards(self) -> bool:
        return ShardState.PENDING in set(self._shard_state)

    @property
    def pending_shards(self) -> set[int]:
        return {i for i, state in enumerate(self._shard_state) if state is ShardState.PENDING}

    @property
    def completed_shards(self) -> set[int]:
        return {i for i, state in enumerate(self._shard_state) if state is ShardState.COMPLETED}

    def _mark_worker_ready(self, rank: Rank) -> None:
        self._alive[rank] = True

    def _mark_worker_done(self, rank: Rank) -> None:
        assert self._roles[rank] is None
        self._alive[rank] = False

    def _mark_worker_idling(self, rank: Rank) -> None:
        assert self._alive[rank]
        self._roles[rank] = None

    def _mark_worker_busy(self, rank: Rank, role: WorkerRole) -> None:
        assert self._alive[rank]
        self._roles[rank] = role

    def _check_worker_busy(self, rank: Rank, role: WorkerRole) -> None:
        assert self._alive[rank]
        assert self._roles[rank] is role

    def _mark_worker_completed(self, rank: Rank) -> None:
        shard_id = self._shard[rank]
        if shard_id is not None:
            self._shard_state[shard_id] = ShardState.COMPLETED
            self._shard[rank] = None

    def _mark_worker_canceled(self, rank: Rank) -> None:
        shard_id = self._shard[rank]
        if shard_id is not None:
            self._shard_state[shard_id] = ShardState.CANCELED
            self._shard[rank] = None

    @property
    def any_worker_alive(self) -> bool:
        return any(self._alive)

    @property
    def alive_workers(self) -> set[Rank]:
        return {i for i, alive in enumerate(self._alive) if alive}

    @property
    def idle_workers(self) -> set[Rank]:
        return {i for i, role in enumerate(self._roles) if role is None}

    @property
    def busy_workers(self) -> set[Rank]:
        return {i for i, role in enumerate(self._roles) if role is not None}

    def get_workers_with_role(self, role: WorkerRole) -> set[Rank]:
        return {i for i, r in enumerate(self._roles) if r is role}


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
    def __init__(self, workers: list[Worker], serializer: Serializer) -> None:
        global _manager

        self.workers = workers
        # create the serializer used in stage 2
        self.serializer = serializer
        # create sample queue used in stage 2 but required for tracker
        self.queue = _manager.Queue(maxsize=self.num_workers)
        self.queue_it = QueueIterator(self.queue, sentinel=None, timeout=1.0)

        self.processor_ranks: set[Rank] = set()
        self.producer_ranks: set[Rank] = set()
        self.consumer_ranks: set[Rank] = set()
        self.joined_ranks: set[Rank] = set()

        self._logger = get_cls_logger(type(self))

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    @property
    def any_producers(self) -> bool:
        return len(self.producer_ranks) > 0

    def start(self) -> None:
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
        self.workers[rank].send_ctx(
            WorkerRole.PROCESSOR, shard, processor, fn, False, blocking=False
        )
        # mark worker as processor
        self.processor_ranks.add(rank)
        self._logger.info(f"Assigned worker {rank} as processor.")

    def create_consumer(
        self,
        rank: Rank,
        processor: Callable[[Iterable], Iterable] | None,
        fn: Callable[[Any], Any] | None,
    ) -> None:
        # assign worker as consumer
        self.workers[rank].send_ctx(
            WorkerRole.CONSUMER,
            self.queue_it,
            compose(processor, self.serializer.deserialize),
            fn,
            False,
            blocking=False,
        )
        # mark worker as consumer
        self.consumer_ranks.add(rank)
        self._logger.info(f"Assigned worker {rank} as consumer.")

    def try_switch_to_producer(self) -> Rank | None:
        # create producer context
        ctx = (WorkerRole.PRODUCER, None, self.serializer.serialize, self.queue.put, False)
        # find a worker that accepts the producer context
        for rank in self.processor_ranks:
            # check if candidate accepts producer context
            if self.workers[rank].send_ctx(*ctx, blocking=True):
                self.processor_ranks.remove(rank)
                self.producer_ranks.add(rank)

                self._logger.info(f"Assigned worker {rank} as producer.")
                return rank

            self._logger.info(f"Worker {rank} did not accept producer context.")

    def try_switch_to_processor(
        self, processor: Callable[[Iterable], Iterable] | None, fn: Callable[[Any], Any] | None
    ) -> Rank | None:
        # create producer context
        ctx = (WorkerRole.PROCESSOR, None, processor, fn, False)
        # find a worker that accepts the producer context
        for rank in self.producer_ranks:
            # check if candidate accepts producer context
            if self.workers[rank].send_ctx(*ctx, blocking=True):
                self.producer_ranks.remove(rank)
                self.processor_ranks.add(rank)

                self._logger.info(f"Assigned worker {rank} as processor.")
                return rank

            self._logger.info(f"Worker {rank} did not accept processor context.")

    def join_worker(self, rank: Rank) -> None:
        assert rank not in self.processor_ranks
        assert rank not in self.producer_ranks
        assert rank not in self.consumer_ranks
        self.workers[rank].join()
        self.joined_ranks.add(rank)

    def assert_all_workers_joined(self) -> None:
        assert len(self.joined_ranks) == len(self.workers)

    def free_worker(self, rank: Rank) -> None:
        self.processor_ranks -= {rank}
        self.producer_ranks -= {rank}
        self.consumer_ranks -= {rank}

    def stop_worker(self, rank: Rank) -> None:
        self.workers[rank].send_ctx(None, None, None, None, True, blocking=False)


class DynamicMultiprocessingRunner(object):
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
        prefetch_factor: int = 8,
        worker_init: Callable[[], Any] = _do_nothing,
        worker_finalize: Callable[[], Any] = _do_nothing,
        progress_update_interval: float = 0.1,
        disable_progress_bar: bool = False,
    ) -> None:
        """Initialize the multiprocessing runner.

        Args:
            num_workers (int): The number of worker processes to create for parallel data
                processing.
            prefetch_factor (int): The number of items per worker that should be prefetched in
                Stage 2 when using a queue. Default is 8.
            worker_init (Callable[[], Any]): A callable that will be invoked to initialize each
                worker. This function will run before the worker starts processing data. Default
                is a no-op function.
            worker_finalize (Callable[[], Any]): A callable that will be invoked to finalize each
                worker. This function will run after the worker has finished processing all data.
                Default is a no-op function.
            progress_update_interval (float): The interval (in seconds) at which the progress
                tracker receives updates from the workers. Default is 0.1.
            disable_progress_bar (bool): If set to True, the progress bar (tqdm) will be disabled
                during processing. Default is False.
        """

        self._num_workers = num_workers
        self._prefetch = prefetch_factor

        self._worker_init = worker_init
        self._worker_finalize = worker_finalize

        self._progress_update_interval = progress_update_interval
        self._disable_progress_bar = disable_progress_bar

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
        ds, processor = self._prepare_dataset(ds)
        self._logger.info("Number of shards: %d", ds.n_shards)

        # create connection for workers to request new context
        recv_msg_conn, worker_msg_conn = mp.Pipe(duplex=False)
        # create all workers
        workers = [
            Worker(
                rank=rank,
                num_workers=self._num_workers,
                send_msg_conn=worker_msg_conn,
                progress_report_interval=self._progress_update_interval,
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
            ds.n_shards, self._num_workers, controller.queue, controller.serializer.batch_size
        )
        # create the reporter thread
        reporter = TqdmReporter(monitor, self._progress_update_interval)
        reporter.start()

        # mark a specific worker as switching
        # used to avoid changing the context of too many workers simultaneously
        switching_worker: None | Rank = None
        last_switch = time()

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

            elif msg_type is MessageType.CTX_COMPLETE:
                controller.free_worker(rank)
                monitor._mark_worker_completed(rank)
                monitor._mark_worker_idling(rank)

                if monitor.get_worker_role(rank) is WorkerRole.PRODUCER:
                    controller.try_switch_to_prodcuer()

            elif msg_type is MessageType.CTX_CANCELED:
                controller.free_worker(rank)
                monitor._mark_worker_canceled(rank)
                monitor._mark_worker_idling(rank)
                
                if monitor.get_worker_role(rank) is WorkerRole.PRODUCER:
                    controller.try_switch_to_prodcuer()

            elif msg_type is MessageType.CTX_SWITCH:
                old_role, new_role = payload
                old_role, new_role = WorkerRole(old_role), WorkerRole(new_role)

                monitor._mark_worker_idling(rank)
                monitor._mark_worker_busy(rank, new_role)

                self._logger.info(
                    f"Worker {rank} switched context from {old_role.name} to {new_role.name}"
                )

            elif msg_type is MessageType.CTX_STARTED:
                role = WorkerRole(payload)
                monitor._mark_worker_busy(rank, role)
                self._logger.info(f"Worker {rank} started running role {role.name}.")
                
                if rank == switching_worker:
                    switching_worker = None

            elif msg_type is MessageType.CTX_REPORT:
                monitor._report_progress(rank, report=payload)

                if (switching_worker is None) and (time() - last_switch) >= 5:
                    # get the average time blocked for queue get operation
                    registered_consumer_workers = monitor.get_workers_with_role(WorkerRole.CONSUMER)
                    get_queue_avg_time = monitor.avg_times(registered_consumer_workers)["producer"]
                    # get the average time blocked for queue put operation
                    registered_producer_workers = monitor.get_workers_with_role(WorkerRole.PRODUCER)
                    put_queue_avg_time = monitor.avg_times(registered_producer_workers)["finalizer"]
                    # compare put and get operation to see if there is a excess
                    # of producers or consumers
                    if abs(put_queue_avg_time - get_queue_avg_time) > 0.1:
                        if get_queue_avg_time > put_queue_avg_time:
                            # get operations take longer than put operations
                            # queue get operation blocks because its empty
                            switching_worker = controller.try_switch_to_producer()

                        elif len(controller.producer_ranks) > 1:
                            # put operations take longer than get operations
                            # queue put operation blocks because its full
                            switching_worker = controller.try_switch_to_processor(processor, fn)

                        last_switch = time()


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
                    shard = ds.shard_data_sources(shard_id, ds.n_shards)
                    controller.create_processor(rank, shard, processor, fn)

                    # mark shard as assigned to worker
                    monitor._mark_shard_in_progress(rank, shard_id)
                    self._logger.info(f"Assigned shard {shard_id} to worker {rank}.")

                else:
                    # Stage 2

                    # check if there is a producer
                    if (switching_worker is None) and not controller.any_producers:
                        switching_worker = controller.try_switch_to_producer()

                    # assign worker as consumer
                    controller.create_consumer(rank, processor, fn)

                    # evenutally all workers are consumers
                    if monitor.alive_workers == controller.consumer_ranks:
                        # this is the signal that gracefully stops the workers
                        monitor._mark_as_stopping()
                        self._logger.info("Stopping criteria reached, gracefully stopping workers.")

        monitor._mark_as_done()
        reporter.join()
        controller.assert_all_workers_joined()

        self._logger.info("Data processing completed.")


class MainProcessRunner(object):
    """Runs data processing in the main thread.

    This class implements the data processing pipeline in the main thread, handling data
    consumption and applying a function to each sample. It supports progress reporting
    with optional progress bar display and handles initialization and finalization tasks.
    """

    def __init__(
        self,
        batch_size: int,
        initialize: Callable[[], Any],
        finalize: Callable[[], Any],
        tqdm_update_interval: float = 0.1,
        disable_tqdm: bool = False,
    ) -> None:
        """Initialize the MainProcessRunner.

        Args:
            batch_size (int): The number of samples to process in each batch.
            initialize (Callable[[], Any]): Function to initialize the processing environment.
            finalize (Callable[[], Any]): Function to finalize the processing environment.
            tqdm_update_interval (float, optional): Interval in seconds for updating the tqdm
                progress bar. Defaults to 0.1.
            disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to
                False, meaning the progress bar is enabled.
        """

        self._batch_size = batch_size
        self._initialize = initialize
        self._finalize = finalize

        self._tqdm_update_interval = tqdm_update_interval
        self._disable_tqdm = disable_tqdm

        self._logger = get_cls_logger(type(self))

    def run(self, ds: IterableDataset, fn: Callable[[Sample], Any]) -> None:
        """Execute data processing on the dataset.

        This method prepares the dataset, initializes the processing environment, and processes
        the dataset in batches. It reports progress to a tracker and updates a progress bar
        if not disabled. The method handles initialization and finalization of the processing
        environment and manages progress reporting throughout the processing.

        Args:
            ds (IterableDataset): The dataset to process.
            fn (Callable[[Sample], Any]): The function to apply to each sample in the dataset.
        """
        self._logger.info("Preparing to run data processing on dataset.")

        # prepare the dataset
        ds = ds._prepare_ex_iterable_for_iteration(batch_size=self._batch_size)
        num_shards = ds.n_shards
        self._logger.info(f"Dataset prepared with {num_shards} shards.")

        # create and start the tracker thread
        tracker = None  # ProgressTracker(num_shards, 1, Queue(), 0)
        tracker.start()
        self._logger.info("ProgressTracker started.")

        def _report_progress(num_samples, shard_exhausted, done):
            report = (0, WorkerRole.PROCESSOR, num_samples, shard_exhausted, done)
            tracker._send_prog_conn.send(report)
            self._logger.debug(f"Reported progress: {num_samples} samples processed.")

        # create and start the reporter thread
        reporter: None | TqdmReporter
        if not self._disable_tqdm:
            reporter = TqdmReporter(tracker, update_interval=self._tqdm_update_interval)
            reporter.start()
            self._logger.info("TqdmReporter started.")

        # set the worker info
        set_worker_info(rank=0, num_workers=1, seed=None)

        try:
            # initialize the consumer
            self._initialize()
            self._logger.info("Initialization complete.")

            last_update_samples = 0
            last_update_time = time()

            counter = count(1)
            # consume the dataset one shard at a time
            for i in range(num_shards):
                self._logger.info(f"Processing shard {i + 1}/{num_shards}.")
                # get the current shard and apply the function to each sample
                shard = ds.shard_data_sources(i, num_shards)
                shard = map(fn, ith_entries(shard, i=1))

                # iterate through the shard and track the total number of samples seen
                # using a global counter
                for j, _ in zip(counter, shard):
                    current_time = time()
                    if current_time - last_update_time > self._tqdm_update_interval:
                        # report progress to tracker
                        _report_progress(j - last_update_samples, False, False)
                        last_update_samples = j
                        last_update_time = current_time

                # report shard exhausted to tracker
                _report_progress(j - last_update_samples, True, False)
                last_update_samples = j
                last_update_time = time()

        except KeyboardInterrupt:  # pragma: not covered
            self._logger.warning("Processing interrupted by user.")
            raise

        finally:
            # report worker done
            self._logger.info("Processing completed. Finalizing worker.")
            _report_progress(0, False, True)

            try:
                # finalize the consumer
                self._finalize()
                self._logger.info("Worker finalized successfully.")
            except Exception as e:  # pragma: not covered
                self._logger.error(f"Error during worker finalization: {e}", exc_info=True)

            # reset the worker info
            reset_worker_info()

        # ensure tracker and reporter threads are joined
        tracker.join()
        self._logger.info("ProgressTracker stopped.")
        if not self._disable_tqdm:
            reporter.join()
            self._logger.info("TqdmReporter stopped.")


class DatasetConsumer(object):
    """Consumes and processes a dataset.

    This class prepares a dataset for processing, manages a processing pipeline,
    and executes data consumption, optionally using parallel processing.
    """

    def __init__(
        self,
        fn: Callable[[Sample], Any],
        num_proc: int = mp.cpu_count(),
        prefetch_factor: int = 8,
        initialize: Callable[[], Any] = _do_nothing,
        finalize: Callable[[], Any] = _do_nothing,
        tqdm_update_interval: float = 0.1,
        disable_tqdm: bool = False,
    ) -> None:
        """Initialize the dataset consumer.

        Args:
            fn (Callable[[Sample], Any]): The function to be applied to each sample in the dataset.
            num_proc (int): The number of processes to use for parallel processing. Defaults to the
                number of CPUs. If set to 1, processing will be single-threaded.
            prefetch_factor (int, optional): The number of items to prefetch in the pipeline.
                Default is 8.
            initialize (Callable[[], Any], optional): A callable to initialize the processing
                pipeline. Default is a no-op function.
            finalize (Callable[[], Any], optional): A callable to finalize the processing
                pipeline. Default is a no-op function.
            tqdm_update_interval (float, optional): The interval in seconds at which the tqdm
                progress bar updates. Default is 0.1.
            disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Default is
                False, meaning the progress bar is enabled.
        """
        self._fn = fn
        self._num_proc = num_proc
        self._prefetch = prefetch_factor

        self._initialize = initialize
        self._finalize = finalize

        self._tqdm_update_interval = tqdm_update_interval
        self._disable_tqdm = disable_tqdm

        self._logger = get_cls_logger(type(self))

    def consume(self, ds: IterableDataset) -> None:
        """Process the dataset.

        Args:
            ds (IterableDataset): The dataset to process.
        """

        if self._num_proc > 1:
            self._logger.info("Running in multi-process mode.")
            # create the multiprocessing runner and run it
            runner = DynamicMultiprocessingRunner(
                num_workers=self._num_proc,
                prefetch_factor=self._prefetch,
                worker_init=self._initialize,
                worker_finalize=self._finalize,
                progress_update_interval=self._tqdm_update_interval,
                disable_progress_bar=self._disable_tqdm,
            )
            runner.run(ds, self._fn)

        else:
            self._logger.info("Running in single-process mode.")
            # create the main process runner and run it
            runner = MainProcessRunner(
                batch_size=self._prefetch,
                initialize=self._initialize,
                finalize=self._finalize,
                tqdm_update_interval=self._tqdm_update_interval,
                disable_tqdm=self._disable_tqdm,
            )
            runner.run(ds, self._fn)


def _parse_size(size_str: str) -> int:
    """Convert a string representation of size to bytes.

    Args:
        size_str (str): The size string (e.g., '5GB', '200MB', '1.5KB').

    Returns:
        int: The size in bytes.

    Raises:
        ValueError: If the size_str is not a valid format.
    """
    size_str = size_str.strip().upper()
    size_units = OrderedDict(
        [("KB", 1024), ("MB", 1024**2), ("GB", 1024**3), ("TB", 1024**4), ("B", 1)]
    )

    if len(size_str) < 2 or not any(unit in size_str for unit in size_units):
        raise ValueError(f"Invalid size format: {size_str}")

    for unit, factor in size_units.items():
        if size_str.endswith(unit):
            try:
                size_value = float(size_str[: -len(unit)].strip())
                return int(size_value * factor)
            except ValueError:
                raise ValueError(f"Invalid size value: {size_str}")

    raise ValueError(f"Invalid size format: {size_str}")


class ShardingStrategy(str, Enum):
    """Enum representing different strategies for sharding dataset samples.

    The strategy determines how the dataset will be divided into shards based on
    either the number of samples, an attribute of the sample, or the file size
    of the written data.
    """

    SAMPLE_COUNT = "sample_count"
    """Shard based on the number of samples.

    In this mode, shards are created once a specified number of samples
    has been written. This is the default mode.
    """

    SAMPLE_ITEM = "sample_item"
    """Shard based on an attribute of each sample.

    In this mode, a specific item or attribute of the sample is used to determine
    shard size. The attribute to be measured must be specified as part of the sharding
    configuration (e.g., an attribute that represents the sample size in terms of data).
    """

    FILE_SIZE = "file_size"
    """Shard based on the total size of written files.

    In this mode, the size of the output files is monitored, and a new shard is started
    once a specified file size threshold is reached (e.g., 1 GB).
    """

    NONE = "none"
    """Disable sharding.

    In this mode, all samples are written to a single file or destination
    without splitting them into multiple shards.
    """


class ShardingController(object):
    """Controller responsible for managing dataset sharding during the writing process.

    This class provides an interface to manage sharding strategies, track shard sizes,
    and control the lifecycle of shards, such as initializing and finalizing shards
    when specific thresholds are met.
    """

    def __init__(
        self,
        is_multi_processed: bool,
        sharding_strategy: ShardingStrategy,
        max_shard_size: None | int | str,
        sample_size_key: None | FeatureKey,
        initialize_shard: Callable[[int], Any],
        finalize_shard: Callable[[int], Any],
    ) -> None:
        """Initialize the :class:`ShardingController`.

        Args:
            is_multi_processed (bool): Whether the sharding controller will manage
                multiple processes.
            sharding_strategy (ShardingStrategy): The strategy for determining when to
                create a new shard.
            max_shard_size (None | int | str): The maximum size of a shard. The format
                depends on the sharding strategy (e.g., integer for sample counts or byte sizes,
                or a string for file size such as "5GB").
            sample_size_key (None | FeatureKey): The key to measure sample size, only
                used with the SAMPLE_ITEM strategy.
            initialize_shard (Callable[[int], Any]): A function to initialize a new shard.
            finalize_shard (Callable[[int], Any]): A function to finalize the current shard.
        """
        global _manager

        if (sharding_strategy is not ShardingStrategy.SAMPLE_ITEM) and (
            sample_size_key is not None
        ):
            warnings.warn(
                "The `sample_size_key` parameter is ignored when using sharding strategies "
                "other than SAMPLE_ITEM.",
                UserWarning,
            )

        if (sharding_strategy is ShardingStrategy.SAMPLE_ITEM) and (sample_size_key is None):
            raise ValueError(
                "The `sample_size_key` must be specified when using the `SAMPLE_ITEM` sharding "
                "strategy."
            )

        if (sharding_strategy is not ShardingStrategy.FILE_SIZE) and isinstance(
            max_shard_size, str
        ):
            raise ValueError(
                "The `max_shard_size` parameter must be an integer when not using the `FILE_SIZE` "
                "sharding strategy."
            )

        if isinstance(max_shard_size, str):
            # parse the size string to an integer
            max_shard_size = _parse_size(max_shard_size)

        if (sample_size_key is not None) and not isinstance(sample_size_key, FeatureKey):
            sample_size_key = FeatureKey(sample_size_key)

        self._is_multi_processed = is_multi_processed
        # sharding strategy
        self._sharding_strategy = sharding_strategy
        self._max_shard_size = max_shard_size
        self._sample_size_key = sample_size_key
        # shard state
        self._shard_id: None | int = None
        self._shard_size = 0
        self._shard_bytes = 0
        self._sample_size = 0
        # global shard state
        self._num_shards = _manager.Value("i32", 0) if is_multi_processed else 0
        self._lock = _manager.Lock() if is_multi_processed else None
        # shard creation
        self._initialize_shard = initialize_shard
        self._finalize_shard = finalize_shard

        self._logger = get_cls_logger(type(self))

    @property
    def is_active(self) -> bool:
        """Returns whether the sharding controller is active.

        Returns:
            bool: True if sharding is active, False otherwise.
        """
        return self._sharding_strategy is not ShardingStrategy.NONE

    def _next_shard_id(self) -> None:
        """Assign the next shard ID.

        If multi-processing is enabled, this is done with thread-safe increments
        using locks; otherwise, the counter is incremented directly.
        """

        assert self._shard_id is None

        if not self._is_multi_processed:
            self._shard_id = self._num_shards
            self._num_shards += 1

        else:
            with self._lock:
                self._shard_id = self._num_shards.get()
                self._num_shards.set(self._shard_id + 1)

    def _reset_state(self) -> None:
        """Reset the shard state variables for the next shard."""
        self._shard_id = None
        self._shard_size = 0
        self._shard_bytes = 0
        self._sample_size = 0

    def callback(self, batch: list[Sample]) -> Sample:
        """Process each batch before writing and check if a new shard is required.

        Args:
            batch (list[Sample]): The batch of samples to be written.

        Returns:
            [Sample]: The batch, unchanged.
        """

        if self._sharding_strategy is ShardingStrategy.SAMPLE_ITEM:
            # cache the size of the sample to be used later in the shard size update
            self._sample_size = sum(map(self._sample_size_key.index_example, batch))

        # check if shard is full
        if self._shard_size >= self._max_shard_size:
            # finalize current shard and initialize a new one
            self.finalize()
            self.initialize()

        return batch

    def update(self, num_bytes: int) -> None:
        """Update the shard size based on the number of bytes written or sample size.

        Args:
            num_bytes (int): The number of bytes written to the current shard.
        """

        # udpate shard size according to the strategy
        self._shard_bytes += num_bytes
        self._shard_size += (
            1
            if self._sharding_strategy is ShardingStrategy.SAMPLE_COUNT
            else self._sample_size
            if self._sharding_strategy is ShardingStrategy.SAMPLE_ITEM
            else num_bytes
            if self._sharding_strategy is ShardingStrategy.FILE_SIZE
            else 0
        )

    def initialize(self) -> None:
        """Initialize a new shard by assigning an ID and invoking the initialization logic."""
        # initialize new shard
        self._next_shard_id()
        self._initialize_shard(self._shard_id)
        self._logger.info(f"Initialized new shard with id {self._shard_id}.")

    def finalize(self) -> None:
        """Finalize the current shard, reset its state, and invoke the finalization logic."""
        if self._shard_id is not None:
            self._finalize_shard()
            self._logger.info(f"Finalized shard with id {self._shard_id}")
            self._reset_state()


class BaseDatasetWriter(ABC):
    """Base class for writing datasets to disk.

    This class provides a framework for saving a dataset to a specified directory.
    The folder structure and save format follows the Hugging Face :func:`save_to_disk`
    structure, ensuring compatibility with datasets saved using this format.

    Subclasses should implement methods for writing individual samples, and initializing
    and finalizing the write operation. The working directory is temporarily set to the
    save directory for writing operations.
    """

    def __init__(
        self,
        save_dir: str,
        overwrite: bool = False,
        num_proc: int = mp.cpu_count(),
        prefetch_factor: int = 8,
        write_batch_size: int = 32,
        tqdm_update_interval: float = 0.1,
        disable_tqdm: bool = False,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FILE_SIZE,
        max_shard_size: None | int | str = "5GB",
        sample_size_key: None | FeatureKey = None,
    ) -> None:
        """Initialize the :class:`BaseDatasetWriter`.

        Args:
            save_dir (str): Directory where the dataset will be saved.
            overwrite (bool): Whether to overwrite existing files in the save directory.
                Defaults to False.
            num_proc (int): Number of processes to use for dataset processing. Defaults
                to the number of CPU cores.
            prefetch_factor (int): Number of samples to prefetch for improved
                performance. Defaults to 8.
            write_batch_size (int): The number of samples to write in a single batch.
                Defaults to 32.
            tqdm_update_interval (float): The interval in seconds at which the tqdm
                progress bar updates. Default is 0.1.
            disable_tqdm (bool): Whether to disable the tqdm progress bar. Default is
                False, meaning the progress bar is enabled.
            sharding_strategy (ShardingStrategy): The strategy to use for sharding
                dataset samples. Defaults to :class:`FILE_SIZE`.
            max_shard_size (None | int | str): Maximum size for each shard according to the
                sharding strategy. If specified, the sharding strategy will consider this limit.
                Defaults to '5GB' matching the default sharding strategy.
            sample_size_key (str, optional): The key in the dataset sample to measure size if using
                the :class:`SAMPLE_ITEM` sharding strategy.
        """

        self.save_dir = save_dir
        self._overwrite = overwrite
        self._num_proc = num_proc
        self._prefetch = prefetch_factor
        self._write_batch_size = write_batch_size
        # tqdm setup
        self._tqdm_update_interval = tqdm_update_interval
        self._disable_tqdm = disable_tqdm
        # sharding
        self._sharding_strategy = sharding_strategy
        self._max_shard_size = max_shard_size
        self._sample_size_key = sample_size_key

        self._logger = get_cls_logger(type(self))

    @property
    def logger(self) -> Logger:
        return self._logger

    def _write_info(self, ds: IterableDataset) -> None:
        """Write dataset information to a JSON file in the save directory.

        Args:
            ds (IterableDataset): The dataset object containing metadata to be saved.
        """

        self._logger.info(f"Writing dataset info to {os.getcwd()}.")
        info = asdict(ds.info)

        with open(
            datasets.config.DATASET_INFO_FILENAME, "w", encoding="utf-8"
        ) as dataset_info_file:
            # Sort only the first level of keys, or we might shuffle fields of nested
            # features if we use sort_keys=True
            sorted_keys_dataset_info = {key: info[key] for key in sorted(info)}
            json.dump(sorted_keys_dataset_info, dataset_info_file, indent=2)

    def _write_state(self, ds: IterableDataset) -> None:
        """Write the state of the dataset to a JSON file in the save directory.

        Args:
            ds (IterableDataset): The dataset object containing state information to be saved.
        """

        self._logger.info(f"Writing dataset state to {os.getcwd()}.")

        keys = (
            "_fingerprint",
            "_format_columns",
            "_format_kwargs",
            "_format_type",
            "_output_all_columns",
        )
        # build state
        state = {key: getattr(ds, key, None) for key in keys}
        state["_format_kwargs"] = {}
        state["_split"] = str(ds.split) if ds.split is not None else ds.split
        state["_data_files"] = [{"filename": fname} for fname in os.listdir(".")]

        # write state to directory
        with open(datasets.config.DATASET_STATE_JSON_FILENAME, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)

    def _write_dataset(self, ds: IterableDataset | Dataset, save_dir: str) -> None:
        """Write the single dataset split to the specified directory.

        Args:
            ds (IterableDataset | Dataset): The dataset or iterable dataset to be written.
            save_dir (str): The directory where the split data will be saved.
        """

        self._logger.info(f"Writing dataset split {ds.split} to {os.getcwd()}.")

        if ds.features is None:
            warnings.warn(
                "The dataset features are not defined. Some dataset writers may require features "
                "to be specified.",
                UserWarning,
            )

        # create sharding controller
        sharding_controller = ShardingController(
            is_multi_processed=self._num_proc > 1,
            sharding_strategy=self._sharding_strategy,
            max_shard_size=self._max_shard_size,
            sample_size_key=self._sample_size_key,
            initialize_shard=partial(self.initialize_shard, info=ds.info),
            finalize_shard=partial(self.finalize_shard, info=ds.info),
        )

        # wrap write function in sharding callback if needed
        write_fn = (
            self.write_batch
            if not sharding_controller.is_active
            else compose(
                sharding_controller.update,
                self.write_batch,
                sharding_controller.callback,
            )
        )

        # wrap write function in batch buffer
        buffered_write_fn = BatchBuffer(batch_size=self._write_batch_size, apply_function=write_fn)

        os.makedirs(save_dir, exist_ok=True)
        # convert dataset to iterable dataset
        if isinstance(ds, Dataset):
            ds = ds.to_iterable_dataset(self._num_proc)

        with chdir(save_dir):
            # write dataset to directory
            consumer = DatasetConsumer(
                buffered_write_fn.add,
                num_proc=self._num_proc,
                prefetch_factor=self._prefetch,
                initialize=run_all(
                    sharding_controller.initialize,
                    partial(self.initialize, ds.info),
                ),
                finalize=run_all(sharding_controller.finalize, partial(self.finalize, ds.info)),
                tqdm_update_interval=self._tqdm_update_interval,
                disable_tqdm=self._disable_tqdm,
            )
            consumer.consume(ds)

            # write dataset info and state
            self._write_state(ds)
            self._write_info(ds)

    def write(self, ds: DatasetType) -> None:
        """Write the entire dataset or dataset dictionary to disk.

        Args:
            ds (DatasetType): The dataset or dataset dictionary to be written.
        """

        # check if save directory already exists
        if os.path.exists(self.save_dir):
            if not self._overwrite:
                raise FileExistsError(
                    f"Output path `{self.save_dir}` already exists. "
                    f"Set `overwrite=True` to overwrite."
                )
            else:
                # delete existing directory
                self._logger.info("Deleting existing directory: {self.save_dir}.")
                shutil.rmtree(self.save_dir)

        # create the save directory
        os.makedirs(self.save_dir, exist_ok=False)

        if isinstance(ds, (DatasetDict, IterableDatasetDict)):
            # write all splits
            for key, split in ds.items():
                self._write_dataset(split, os.path.join(self.save_dir, key))
            # write dataset splits json
            with open(
                os.path.join(self.save_dir, datasets.config.DATASETDICT_JSON_FILENAME), "w+"
            ) as f:
                f.write(json.dumps({"splits": list(ds.keys())}))

        else:
            # save dataset to directory
            self._write_dataset(ds, self.save_dir)

    @abstractmethod
    def write_batch(self, batch: list[Sample]) -> int:
        """Abstract method for writing a batch of samples.

        This method writes a batch of samples to the dataset shard and returns the
        number of bytes written.

        The working directory is temporarily set to the save directory during this method,
        any files created will be saved in the designated dataset directory.

        Args:
            batch (list[Sample]): The batch of samples to be written.

        Returns:
            int: The number of bytes written to the shard.
        """
        ...

    def initialize(self, info: DatasetInfo) -> None:
        """Initialize the global dataset write process.

        This method is responsible for any setup tasks that need to be performed once before
        writing begins for the dataset. This could include setting up metadata files, preparing
        the global output directory, or initializing any resources required for the write
        operation. The working directory is temporarily set to the global directory during
        this method.

        Args:
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        ...  # pragma: not covered

    def finalize(self, info: DatasetInfo) -> None:
        """Finalize the global dataset write process.

        This method is responsible for any cleanup tasks or final operations that should be
        performed after all shards of the dataset have been processed and written to disk.
        This could include writing final metadata files, closing any global resources, and
        ensuring that all data is properly stored. The working directory is temporarily set
        to the global save directory during this method.

        Args:
            info (DatasetInfo): Information about the dataset that was written, including metadata
                and configuration details.
        """
        ...  # pragma: not covered

    @abstractmethod
    def initialize_shard(self, shard_id: int, info: DatasetInfo) -> None:
        """Abstract method for initializing the write process for a new shard.

        Any setup tasks specific to writing a new shard, such as creating necessary files or folders
        for the shard, should take place here. The working directory is temporarily set to the save
        directory during this method.

        Args:
            shard_id (int): The id of the shard being initialized.
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        ...

    @abstractmethod
    def finalize_shard(self, info: DatasetInfo) -> None:
        """Abstract method for finalizing the write process for the current shard.

        This method should handle any cleanup or final write operations after the samples for the
        current shard have been processed. The working directory is temporarily set to the save
        directory during this method.

        Args:
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        ...

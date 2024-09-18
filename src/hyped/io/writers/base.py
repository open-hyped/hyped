"""Base Module for dynamic multiprocessing and dataset consumption.

This module provides high-level functionality for processing large datasets using a dynamic
multiprocessing system. It includes utilities for managing worker processes, tracking progress,
and implementing custom dataset writers.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import multiprocessing.connection  # noqa: F401
import os
import shutil
import threading
from abc import ABC, abstractmethod
from collections import Counter
from copy import copy
from dataclasses import asdict
from enum import Enum
from functools import partial
from itertools import count
from queue import Queue
from time import time
from typing import Any, Callable, Iterable, TypeAlias, TypeVar

import datasets
from datasets import Dataset, DatasetDict, DatasetInfo, IterableDataset, IterableDatasetDict
from datasets.iterable_dataset import (
    FilteredExamplesIterable,
    MappedExamplesIterable,
    TypedExamplesIterable,
    _BaseExamplesIterable,
)
from tqdm.auto import tqdm
from tqdm.std import EMA

from hyped.common._worker import reset_worker_info, set_worker_info
from hyped.common.logging import Logger, get_cls_logger
from hyped.common.typing import DatasetType, Rank, Sample
from hyped.common.utils import QueueIterator, StoppableIterator, chdir

IndexedSample: TypeAlias = tuple[str, Sample]


def do_nothing():
    """A no-operation function that returns nothing."""
    return


T = TypeVar("T")
U = TypeVar("U")


def _passthrough(x: T) -> T:
    """Identity function that returns the input unchanged."""
    return x


def _drop_key_and_apply(key_and_sample: tuple[str, Sample], fn: Callable[[Sample], U]) -> U:
    """Apply a function to the sample part of a key-value tuple.

    Args:
        key_and_sample (tuple[str, Sample]): A tuple where the second element is the sample.
        fn (Callable[[Sample], Any]): The function to apply to the sample.

    Returns:
        Any: The result of applying the function to the sample.
    """
    return fn(key_and_sample[1])


class WorkerRole(Enum):
    """Enumeration of different roles a worker can assume during multiprocessing.

    Workers can dynamically switch between these roles based on the current processing stage
    and system needs.
    """

    PROCESSOR = 0
    """Role where the worker processes a shard of data independently. 

    In this role, the worker is responsible for processing its assigned shard of the dataset
    without interacting with other workers. This occurs in Stage 1 where each worker processes
    a distinct shard.
    """

    PRODUCER = 1
    """Role where the worker produces data and adds it to a shared queue. 

    In this role, the worker reads data from a shard and places it into the queue for further
    processing by other workers. This occurs in Stage 2 when the system shifts to multi-worker
    processing of a single shard.
    """

    CONSUMER = 2
    """Role where the worker consumes data from a shared queue for processing. 

    In this role, the worker retrieves data from the queue (populated by a PRODUCER) and processes
    it. This role is also part of Stage 2, where multiple workers collaborate on processing data
    from a single shard.
    """


ContextTuple: TypeAlias = tuple[
    WorkerRole | None,
    Iterable[IndexedSample],
    Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]] | None,
    Callable[[IndexedSample], Any] | None,
    bool,
]
"""Type alias representing the context passed to workers.

This tuple defines the context used for setting up a worker's role, producer,
processing function, finalizer function, and a completion flag.

Elements:
    - WorkerRole | None: The role assigned to the worker (e.g., producer, processor),
      or None if the role remains unchanged.
    - Iterable[IndexedSample]: The producer for the worker, providing an iterable
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


class Worker(mp.Process):
    """A worker process for parallel data processing.

    This class extends :code:`mp.Process` to handle data processing in a separate
    process, managing context and processing stages for data.
    """

    def __init__(
        self,
        rank: Rank,
        num_workers: int,
        req_ctx_conn: mp.connection.Connection,
        tracker_conn: mp.connection.Connection,
        tracker_update_interval: float = 0.1,
        worker_init: Callable[[], Any] = do_nothing,
        worker_finalize: Callable[[], Any] = do_nothing,
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

        self._tracker_conn = tracker_conn
        self._tracker_update_interval = tracker_update_interval

        self._rank = rank
        self._num_workers = num_workers
        # context management connections
        self._req_ctx_conn = req_ctx_conn
        self._recv_ctx_conn, self._send_ctx_conn = mp.Pipe(duplex=False)
        self._recv_ctx_done = mp.Event()
        # worker initializer and finalizer
        self._worker_init = worker_init
        self._worker_finalize = worker_finalize
        # pipeline to be executed by the worker
        self._role: WorkerRole | None = None
        self._producer: Iterable[IndexedSample] | None = None
        self._processor: Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]] | None = None
        self._finalizer: Callable[[IndexedSample], Any] | None = None
        # set the event to avoid blocking the request of the first context
        self._recv_ctx_done.set()

        # create logger
        self._logger = get_cls_logger(type(self))

        # Log initialization details
        self._logger.debug(
            f"Created worker with rank {self._rank} of {self._num_workers} total workers."
        )

    def send_ctx(
        self,
        role: WorkerRole | None,
        producer: Iterable[IndexedSample] | None,
        processor: Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]] | None,
        finalizer: Callable[[IndexedSample], Any] | None,
        done: bool,
    ) -> None:
        """
        Send new processing context to the worker.

        Args:
            role (WorkerRole | None): The role of the worker.
            producer (Iterable[IndexedSample] | None): The producer iterable for generating samples.
            processor (Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]] | None): The
                processor function.
            finalizer (Callable[[IndexedSample], Any] | None): The finalizer function.
            done (bool): Whether the context change is final.
        """
        self._recv_ctx_done.clear()
        self._send_ctx_conn.send((role, producer, processor, finalizer, done))
        self._recv_ctx_done.wait()
        self._logger.debug("Sent new context to worker.")

    def _request_new_ctx(self) -> bool:
        """Request new processing context from the main process.

        Returns:
            bool: Whether the worker has been instructed to stop.
        """
        # only ask for a new context if currently no new context
        # is being received
        if self._recv_ctx_done.is_set():
            self._logger.debug("Requesting new context from main process.")
            # request new context from main process
            self._recv_ctx_done.clear()
            self._req_ctx_conn.send(self._rank)

        # wait for new context to be received
        return self._apply_ctx(self._recv_ctx())

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
        while not self._recv_ctx_conn.poll(timeout=1.0):
            self._logger.debug("Waiting for new context...")

        # receive context
        while self._recv_ctx_conn.poll():
            ctx = self._recv_ctx_conn.recv()

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
        # set receive process done event
        self._recv_ctx_done.set()

        self._logger.debug(
            f"Applied new context with role "
            f"`{self._role.name if self._role is not None else None}`."
        )

        return done

    def _report_progress(self, num_samples: int, producer_exhausted: bool, done: bool) -> None:
        """Send a progress update to the tracker.

        This function reports the progress of the current worker to the tracker by sending
        information about the worker's role, number of processed samples, producer status,
        and whether the worker has finished processing.

        Args:
            num_samples (int): The number of samples processed by the worker since the last update.
            producer_exhausted (bool): Whether the producer has exhausted all samples.
            done (bool): Whether the worker has finished processing.
        """
        self._tracker_conn.send((self._rank, self._role, num_samples, producer_exhausted, done))
        self._logger.debug("Reported progress to tracker.")

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
            self._logger.info("Initialization complete.")

            done = False
            # request a processing context
            while (not done) and (not self._request_new_ctx()):
                # report new context to tracker
                self._report_progress(0, False, False)
                # create producer iterator to avoid resetting when
                # new context is received during execution
                producer_iter = iter(self._producer)
                producer_exhausted = False

                # exhaust producer
                while not producer_exhausted:
                    num_samples = 0
                    last_update = time()

                    # create a stoppable producer that allows to dynamically
                    # interrupt the execution and apply the new context
                    stoppable_producer = StoppableIterator(producer_iter)

                    try:
                        self._logger.debug(f"Starting processing with role {self._role}.")

                        # apply the processor to the producer and apply the finalizer
                        samples_iter = self._processor(stoppable_producer)
                        work_iter = map(self._finalizer, samples_iter)

                        # main worker loop
                        for _ in work_iter:
                            num_samples += 1

                            # check if the worker was asked to apply a new context
                            if self._recv_ctx_conn.poll():
                                self._logger.debug("Detected context update request.")

                                # mark the worker as receiving a new context
                                self._recv_ctx_done.clear()

                                # receive and unpack context
                                ctx = self._recv_ctx()
                                _, prod, _, _, done = ctx

                                # the producer is not allowed to change
                                assert (
                                    prod is None
                                ), f"Unexpected producer received in worker with rank {self._rank}."

                                if done:
                                    # stop worker
                                    self._logger.info("Received 'done' signal, stopping.")
                                    raise StopIteration()

                                # stop the producer from generating further samples
                                # and exhaust the current samples generated by the producer
                                stoppable_producer.stop()
                                for _ in work_iter:
                                    num_samples += 1

                                # send progress to tracker
                                self._report_progress(num_samples, producer_exhausted, False)

                                # apply the new context
                                self._apply_ctx(ctx)
                                break

                            # send continuous updates to tracker
                            if (num_samples > 0) and (
                                time() - last_update > self._tracker_update_interval
                            ):
                                self._report_progress(num_samples, producer_exhausted, False)
                                num_samples = 0
                                last_update = time()

                        else:
                            # producer exhausted
                            self._logger.info("Finished processing current context.")
                            producer_exhausted = True

                    except StopIteration:
                        # catch stop execution error
                        producer_exhausted = True

                    except KeyboardInterrupt:  # pragma: not covered
                        self._logger.warning("Worker interrupted by user.")

                    except Exception as e:
                        # gracefully handle exceptions without stopping the worker
                        self._logger.error(
                            f"Unexpected error during processing: {str(e)}.", exc_info=True
                        )

                # send final progress update before new requesting context
                self._report_progress(num_samples, producer_exhausted, False)

        except KeyboardInterrupt:  # pragma: not covered
            self._logger.warning("Worker interrupted by user.")

        except Exception as e:
            # gracefully handle exception
            self._logger.error(f"Unexpected error during processing: {str(e)}.", exc_info=True)

        finally:
            try:
                # finalize worker
                self._worker_finalize()
                self._logger.info("Worker finalized successfully.")
            except Exception as e:
                self._logger.error(f"Error finalizing worker: {str(e)}.", exc_info=True)
            finally:
                # tell tracker that worker terminated
                self._report_progress(0, False, True)

        self._logger.info("Worker finished.")


class TqdmReporter(threading.Thread):
    """Tqdm Reporter Thread.

    A thread that reports progress using the :func:`tqdm` progress bar, based on the state of a
    :class:`ProgressTracker`.
    """

    def __init__(self, tracker: ProgressTracker, update_interval: float = 0.1) -> None:
        """Initializes the :class:`TqdmReporter` thread.

        Args:
            tracker (ProgressTracker): The :class:`ProgressTracker` instance to track progress.
            update_interval (float): The interval in seconds between progress updates. Defaults
                to 0.1.
        """
        super(TqdmReporter, self).__init__(daemon=True)
        self._tracker = tracker
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
        counts = Counter(self._tracker._roles)
        return (
            f"Workers {self._tracker.num_busy_workers}/{self._tracker._num_workers} "
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

        with tqdm(total=self._tracker._num_shards, desc=self._pbar_desc) as pbar:
            self._logger.debug(
                f"Initialized tqdm progress bar with {self._tracker._num_shards} total shards."
            )

            prev_total_samples = 0
            prev_update_time = pbar._time()

            def _iter():
                # iterate as long as the tracker is running
                while not self._tracker._done.wait(timeout=self._update_interval):
                    yield
                # do a final update after the tracker finished
                yield

            for _ in _iter():
                # get current state
                total_samples = self._tracker.total_consumed_samples
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
                        f"{self._tracker._queue.qsize()}q, "
                        f"{throughput:.02f}ex/s, "
                        f"{formatted_total_samples}ex"
                    ),
                    refresh=False,
                )
                # update values
                prev_total_samples = total_samples
                prev_update_time = update_time

                # update the progress bar
                pbar.update(self._tracker._finished_shards - pbar.n)

        self._logger.info("Thread finished.")


class ProgressTracker(threading.Thread):
    """Progress Tracker Thread

    A monitoring thread that tracks the state and progress of all workers in a centralized fashion.

    The progress tracker monitors the number of processed and consumed samples by each worker,
    updates the roles of the workers (processor, producer, consumer), and tracks the number of
    finished shards.
    """

    def __init__(self, num_shards: int, num_workers: int, queue: mp.Queue) -> None:
        """Initializes the ProgressTracker.

        Args:
            num_shards (int): The total number of shards to process.
            num_workers (int): The number of workers processing the data.
            queue (mp.Queue): Sample queue filled by producer in stage 2.
        """
        super(ProgressTracker, self).__init__(daemon=True)

        self._queue = queue
        # track shards
        self._num_shards = num_shards
        self._finished_shards = 0
        # track workers
        self._num_workers = num_workers
        self._stopped_workers = 0
        self._roles = [None] * num_workers
        # track samples generated by each worker in different roles
        self._num_samples = [
            {
                WorkerRole.PROCESSOR: 0,
                WorkerRole.PRODUCER: 0,
                WorkerRole.CONSUMER: 0,
            }
            for _ in range(num_workers)
        ]
        # connection to workers
        self._recv_prog_conn, self._send_prog_conn = mp.Pipe(duplex=False)
        # event set by worker indicating that the process has finished
        self._done = threading.Event()

        self._logger = get_cls_logger(type(self))

    @property
    def total_produced_samples(self) -> int:
        """Returns the total number of samples produced by all workers.

        This includes samples generated by workers in the :class:`WorkerRole.PROCESSOR` and
        :class:`WorkerRole.PRODUCER` roles.

        Returns:
            int: Total number of produced samples.
        """
        return sum(samples[WorkerRole.PROCESSOR] for samples in self._num_samples) + sum(
            samples[WorkerRole.PRODUCER] for samples in self._num_samples
        )

    @property
    def total_consumed_samples(self) -> int:
        """Returns the total number of samples consumed by all workers.

        This includes samples generated by workers in the :class:`WorkerRole.PROCESSOR` and
        :class:`WorkerRole.CONSUMER` roles.

        Returns:
            int: Total number of consumed samples.
        """
        return sum(samples[WorkerRole.PROCESSOR] for samples in self._num_samples) + sum(
            samples[WorkerRole.CONSUMER] for samples in self._num_samples
        )

    def recv_progress(self) -> tuple[Rank, WorkerRole, int, bool, bool]:
        """Receives progress updates from the workers through a multiprocessing pipe.

        Returns:
            tuple[Rank, WorkerRole, int, bool bool]: A tuple containing the worker
            rank, role, number of samples processed, whether the producer is exhausted,
            and whether the worker has terminated.
        """
        rank, role, num_samples, producer_exhausted, done = self._recv_prog_conn.recv()
        self._logger.debug(
            f"Received progress update: Rank {rank}, Role {role}, Samples {num_samples}, "
            f"Producer exhausted {producer_exhausted}, Done {done}"
        )
        return rank, role, num_samples, producer_exhausted, done

    @property
    def num_busy_workers(self) -> int:
        """Returns the number of workers that are currently busy.

        A worker is considered busy if its role is not None.

        Returns:
            int: Number of busy workers.
        """
        return sum(role is not None for role in self._roles)

    def run(self) -> None:
        """Run progress tracker thread.

        Starts the thread and updates the tqdm progress bar based on progress reports from workers.

        It polls the progress pipe for updates, processes the reports, and computes throughput
        statistics. The progress bar is updated with the number of finished shards and the
        processing throughput.

        The thread continues running until all workers have stopped.
        """
        self._logger.info("Thread started.")

        while self._stopped_workers < self._num_workers:
            if self._recv_prog_conn.poll(timeout=0.05):
                # receive progress update
                rank, role, num_samples, producer_exhausted, done = self.recv_progress()

                if role is not None:
                    # update worker role and number of processed samples
                    self._roles[rank] = role
                    self._num_samples[rank][role] += num_samples

                if producer_exhausted and (role in {WorkerRole.PROCESSOR, WorkerRole.PRODUCER}):
                    # update number of finished shards
                    self._finished_shards += 1
                    self._logger.debug(
                        f"Finished shards incremented. Total finished shards: "
                        f"{self._finished_shards}"
                    )

                if done:
                    # update number of stopped workers
                    self._stopped_workers += 1
                    self._roles[rank] = None
                    self._logger.debug(
                        f"Worker {rank} stopped. Total stopped workers: {self._stopped_workers}."
                    )

        # set done event
        self._done.set()
        self._logger.info("Thread finished.")


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
        worker_init: Callable[[], Any] = do_nothing,
        worker_finalize: Callable[[], Any] = do_nothing,
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

    def run(
        self,
        ds: IterableDataset,
        processor: Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]],
        finalizer: Callable[[IndexedSample], Any],
    ) -> None:
        """Execute data processing using the worker processes.

        Args:
            ds: (IterableDataset): The dataset to process.
            processor (Callable[[Iterable[IndexedSample]], Iterable[IndexedSample]]): The function
                to process each sample.
            finalizer (Callable[[IndexedSample], Any]): The function to finalize each
                sample.
        """

        self._logger.info("Starting data processing.")

        with mp.Manager() as manager:
            # prepare the dataset
            num_shards = ds.n_shards
            ds = ds._prepare_ex_iterable_for_iteration(batch_size=self._prefetch)
            self._logger.info("Number of shards: %d", num_shards)

            # create connection for workers to request new context
            req_ctx_conn, worker_req_ctx_conn = mp.Pipe(duplex=False)

            # create sample queue used in stage 2 but required for tracker
            queue = manager.Queue(maxsize=self._num_workers * self._prefetch)
            iterable = QueueIterator(queue, sentinel=None, timeout=None)

            # create the progress tracker
            tracker = ProgressTracker(num_shards, self._num_workers, queue)
            tracker.start()
            self._logger.info("Progress tracker started.")

            # create the reporter if asked for
            reporter: None | TqdmReporter
            if not self._disable_progress_bar:
                reporter = TqdmReporter(tracker, update_interval=self._progress_update_interval)
                reporter.start()
                self._logger.info("TqdmReporter started.")

            # create all workers
            workers = [
                Worker(
                    rank=rank,
                    num_workers=self._num_workers,
                    req_ctx_conn=worker_req_ctx_conn,
                    tracker_conn=tracker._send_prog_conn,
                    tracker_update_interval=self._progress_update_interval,
                    worker_init=self._worker_init,
                    worker_finalize=self._worker_finalize,
                )
                for rank in range(self._num_workers)
            ]
            self._logger.info(f"All {self._num_workers} workers created.")

            # start all workers
            for worker in workers:
                worker.start()

            self._logger.info("All workers started.")

            # keep track of all running workers
            running_worker_ranks: set[int] = set()

            self._logger.info("Starting Stage 1: Single-Shard Single-Worker")
            # stage 1: shards available
            # whenever a group is idleing give it a new shard to process
            for shard_id in range(num_shards):
                shard = ds.shard_data_sources(shard_id, num_shards)
                # wait for worker to request new context
                rank = req_ctx_conn.recv()
                worker = workers[rank]
                # send new context to worker
                worker.send_ctx(WorkerRole.PROCESSOR, shard, processor, finalizer, False)
                running_worker_ranks.add(rank)

                self._logger.info(f"Assigned shard {shard_id} to worker {rank}.")

            # Stage 2: all shards being processed
            self._logger.info("Starting Stage 2: Single-Shard Multiple-Workers")

            # keep track of producer and consumer workers
            producer_worker_rank: None | int = None
            consumer_worker_ranks: set[int] = set()

            while len(running_worker_ranks) > 0:
                # collect all idling workers
                idling_worker_ranks = {req_ctx_conn.recv()}
                while req_ctx_conn.poll(timeout=0.1):
                    idling_worker_ranks.add(req_ctx_conn.recv())

                self._logger.debug("Idling workers: %s", idling_worker_ranks)

                # update running ranks
                running_worker_ranks -= idling_worker_ranks

                # use idling ranks as consumers
                for rank in idling_worker_ranks:
                    workers[rank].send_ctx(
                        WorkerRole.CONSUMER, iterable, processor, finalizer, False
                    )
                    self._logger.info(f"Assigned worker {rank} as consumer.")

                # update consumer worker ranks
                consumer_worker_ranks.update(idling_worker_ranks)

                if len(running_worker_ranks) > 0:
                    if (producer_worker_rank is None) or (
                        producer_worker_rank in idling_worker_ranks
                    ):
                        # select a running rank as the producer
                        producer_worker_rank = next(iter(running_worker_ranks))
                        workers[producer_worker_rank].send_ctx(
                            WorkerRole.PRODUCER, None, _passthrough, queue.put, False
                        )
                        self._logger.info(f"Assigned worker {producer_worker_rank} as producer.")

                else:
                    # no worker running that could act as producer
                    producer_worker_rank = None

            self._logger.info("Sending sentinel signals to consumer workers.")
            # put sentinel signal to queue, one per consumer worker
            for _ in consumer_worker_ranks:
                queue.put(None)

            self._logger.info("Stopping all workers.")
            # send stop signal to all workers
            for _ in range(self._num_workers):
                rank = req_ctx_conn.recv()
                workers[rank].send_ctx(None, None, None, None, True)

            self._logger.info("Waiting for all workers to join.")
            # wait for all workers to join
            for worker in workers:
                worker.join()

            self._logger.info("Waiting for tracker and reporter to join.")
            # wait for tracker and reporter to join
            tracker.join()
            if not self._disable_progress_bar:
                reporter.join()

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

        # prepare the function to apply
        fn = partial(_drop_key_and_apply, fn=fn)

        # create and start the tracker thread
        tracker = ProgressTracker(num_shards, 1, Queue())
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
                shard = map(fn, shard)

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
        initialize: Callable[[], Any] = do_nothing,
        finalize: Callable[[], Any] = do_nothing,
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

    def _prepare_dataset(
        self, ds: IterableDataset
    ) -> tuple[IterableDataset, Callable[[Iterable[Sample]], Iterable[Sample]]]:
        """Prepare the dataset for processing by separating processing steps.

        Args:
            ds (IterableDataset): The dataset to prepare.

        Returns:
            tuple[IterableDataset, Callable[[Iterable[Sample]], Iterable[Sample]]]:
            A tuple containing the processed dataset and a function to run the pipeline.
        """

        if not isinstance(
            ds._ex_iterable,
            (
                MappedExamplesIterable,
                FilteredExamplesIterable,
                TypedExamplesIterable,
            ),
        ):
            return ds, _passthrough

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

        # create the source dataaset that excludes the
        # pipeline processing steps
        src_ds = IterableDataset(ex_iterable=pipeline.src_iterable)

        return src_ds, pipeline.copy()

    def consume(self, ds: IterableDataset) -> None:
        """Process the dataset.

        Args:
            ds (IterableDataset): The dataset to process.
        """

        if self._num_proc > 1:
            self._logger.info("Running in multi-process mode.")
            # prepare the dataset and function to apply
            src_ds, processor = self._prepare_dataset(ds)
            finalizer = partial(_drop_key_and_apply, fn=self._fn)

            # create the multiprocessing runner and run it
            runner = DynamicMultiprocessingRunner(
                num_workers=self._num_proc,
                prefetch_factor=self._prefetch,
                worker_init=self._initialize,
                worker_finalize=self._finalize,
                progress_update_interval=self._tqdm_update_interval,
                disable_progress_bar=self._disable_tqdm,
            )
            runner.run(src_ds, processor, finalizer)

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
        tqdm_update_interval: float = 0.1,
        disable_tqdm: bool = False,
    ) -> None:
        """Initialize the :class:`BaseDatasetWriter`.

        Args:
            save_dir (str): Directory where the dataset will be saved.
            overwrite (bool, optional): Whether to overwrite existing files in the save directory.
                Defaults to False.
            num_proc (int, optional): Number of processes to use for dataset processing. Defaults
                to the number of CPU cores.
            prefetch_factor (int, optional): Number of samples to prefetch for improved
                performance. Defaults to 8.
            tqdm_update_interval (float, optional): The interval in seconds at which the tqdm
                progress bar updates. Default is 0.1.
            disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Default is
                False, meaning the progress bar is enabled.
        """
        self.save_dir = save_dir
        self._overwrite = overwrite
        self._num_proc = num_proc
        self._prefetch = prefetch_factor
        self._tqdm_update_interval = tqdm_update_interval
        self._disable_tqdm = disable_tqdm

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

        os.makedirs(save_dir, exist_ok=True)
        # convert dataset to iterable dataset
        if isinstance(ds, Dataset):
            ds = ds.to_iterable_dataset(self._num_proc)

        with chdir(save_dir):
            # write dataset to directory
            consumer = DatasetConsumer(
                self.write_sample,
                num_proc=self._num_proc,
                prefetch_factor=self._prefetch,
                initialize=partial(self.initialize, ds.info),
                finalize=partial(self.finalize, ds.info),
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
    def write_sample(self, sample: Sample) -> None:
        """Abstract method for writing an individual sample.

        The working directory is temporarily set to the save directory during this method,
        so any files created will be saved in the designated dataset directory.

        Args:
            sample (Sample): The sample to be written.
        """
        ...

    @abstractmethod
    def initialize(self, info: DatasetInfo) -> None:
        """Abstract method for initializing the write process.

        Any setup tasks, such as creating necessary files or folders, should take place here.
        The working directory is temporarily set to the save directory during this method.

        Args:
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        ...

    @abstractmethod
    def finalize(self, info: DatasetInfo) -> None:
        """Abstract method for finalizing the write process.

        This method should handle any cleanup or final write operations after all samples
        have been processed. The working directory is temporarily set to the save directory
        during this method.

        Args:
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        ...

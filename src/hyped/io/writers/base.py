"""

TODO:

Error handling:
- Error Handling in Workers: Each worker should catch exceptions during task processing and skip to
  the next sample instead of crashing.
- Error Logging: Workers will log errors, including details like the task, error message, and
  traceback.
- Shared Error Log: Use a thread-safe structure (e.g., Queue) to collect errors from all workers
  and pass them back to the main process.
- User-Accessible Error Log: After processing, the consumer will provide access to the error log
  via a method (e.g., get_errors()) for user inspection.
- No Retries: Tasks will not be retried upon failure; they will simply be logged and skipped.

Error log content:
- Worker rank: Identifies which worker process encountered the error.
- Sample index: Indicates which sample in the dataset caused the error.
- Error message: A description of the error that occurred.
- Error type: The specific type of error (e.g., ValueError, TypeError).
- Traceback: A detailed stack trace for debugging purposes.

Worker & Group Monitoring:
- Monitor Idle Time of worker
- Monitor Idle Time of consumer process in group
- Monitor Average Idle Time of worker process in group

DONE:

Separate Dataset from Transformation:
- datasets.iterable_dataset._BaseExamplesIterable wraps dataset (nested)
- wrap source dataset at lowest level in thread-safe iterable
  - manages a queue which some prefetch size that the object always fills up with source samples
  - iterable iterates over the queue
  - maybe include batching logic for reduced communication overhead
- exchange source dataset with iterable dataset build from thread-safe iterable

Worker Groups & Processing Strategies:
- If a worker group consists of only one worker, the worker uses the shard process stategy
- If a worker group consists of more then one worker, the group uses the distributed process stategy
- Worker groups are dynamically managed by the main process based on ressource usage

Dynamic Group Management:
- Worker Groups are build based on ressource usage and remaining shards to process
- Initial Worker Groups:
  - Number of Shards >= Number of Workers: N groups of size 1
  - Number of Shards < Number of Workers:  1 group of size N
- Dynamic Group Updates:
  - 
  
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
from time import time
from typing import Any, Callable, Iterable, TypeAlias

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
from hyped.common.typing import DatasetType, Rank, Sample
from hyped.common.utils import (
    QueueIterator,
    StoppableIterator,
    chdir,
    dict_of_lists_to_list_of_dicts,
)

IndexedSamplesIterable: TypeAlias = Iterable[tuple[str, Sample]]


def do_nothing():
    """A no-operation function that returns nothing."""
    return


def _passthrough(x: Any) -> Any:
    """Identity function that returns the input unchanged."""
    return x


def _drop_key_and_apply(key_and_sample: tuple[str, Sample], fn: Callable[[Sample], Any]) -> Any:
    """Apply a function to the sample part of a key-value tuple.

    Args:
        key_and_sample (tuple[str, Sample]): A tuple where the second element is the sample.
        fn (Callable[[Sample], Any]): The function to apply to the sample.

    Returns:
        Any: The result of applying the function to the sample.
    """
    return fn(key_and_sample[1])


class WorkerRole(Enum):
    PROCESSOR = 0
    PRODUCER = 1
    CONSUMER = 2


class Worker(mp.Process):
    """A worker process for parallel data processing.

    This class extends :code:`mp.Process` to handle data processing in a separate
    process, managing context and processing stages for data.
    """

    def __init__(
        self,
        rank: Rank,
        num_workers: int,
        dataset_info: DatasetInfo,
        tracker_conn: mp.connection.Connection,
        req_ctx_conn: mp.connection.Connection,
        worker_init: Callable[[], Any] = do_nothing,
        worker_finalize: Callable[[], Any] = do_nothing,
    ) -> None:
        """Initialize a worker process for parallel data processing.

        Args:
            rank (Rank): The rank or identifier for the worker, typically representing
                the worker's position in a set of workers.
            num_workers (int): The total number of workers involved in the process.
            req_ctx_conn (mp.connection.Connection): A connection object to request
                new context updates from the main process.
        """
        super(Worker, self).__init__(daemon=True)

        self._tracker_conn = tracker_conn

        self._rank = rank
        self._num_workers = num_workers
        self._dataset_info = dataset_info
        # context management connections
        self._req_ctx_conn = req_ctx_conn
        self._recv_ctx_conn, self._send_ctx_conn = mp.Pipe(duplex=False)
        # worker initializer and finalizer
        self._worker_init = worker_init
        self._worker_finalize = worker_finalize
        # pipeline to be executed by the worker
        self._role: WorkerRole | None = None
        self._producer: IndexedSamplesIterable | None = None
        self._processor: Callable[[IndexedSamplesIterable], IndexedSamplesIterable] | None = None
        self._finalizer: Callable[[IndexedSamplesIterable], Any] | None = None

    def send_ctx(
        self,
        role: WorkerRole | None,
        producer: IndexedSamplesIterable | None,
        processor: Callable[[IndexedSamplesIterable], IndexedSamplesIterable] | None,
        finalizer: Callable[[IndexedSamplesIterable], Any] | None,
        done: bool,
    ) -> None:
        """
        Send new processing context to the worker.

        Args:
            role (WorkerRole | None): The role of the worker.
            producer (IndexedSamplesIterable | None): The producer iterable for generating samples.
            processor (Callable[[IndexedSamplesIterable], IndexedSamplesIterable] | None): The
                processor function.
            finalizer (Callable[[IndexedSamplesIterable], Any] | None): The finalizer function.
            done (bool): Whether the context change is final.
        """
        self._send_ctx_conn.send((role, producer, processor, finalizer, done))

    def _request_new_ctx(self) -> bool:
        """Request new processing context from the main process.

        Returns:
            bool: Whether the worker has been instructed to stop.
        """
        # request new context from main process
        self._req_ctx_conn.send(self._rank)
        # wait for new context to be received
        done, new_ctx = self._recv_ctx(keep_producer=False)
        assert new_ctx or done

        return done

    def _check_ctx(self) -> tuple[bool, bool]:
        """Check if a new context is available.

        Returns:
            tuple[bool, bool]: A tuple where the first element indicates if the worker should stop,
            and the second element indicates if new context was set.
        """

        # check if the worker was asked to apply a new context
        if self._recv_ctx_conn.poll():
            return self._recv_ctx(keep_producer=True)

        return False, False

    def _recv_ctx(self, keep_producer: bool) -> tuple[bool, bool]:
        """Receive and update the processing context.

        This method receives a new processing context from the worker's connection,
        updating the worker's pipeline with the new producer, processor, and finalizer
        functions. If :code:`keep_producer` is :code:`True`, the current producer is
        preserved, and only the processing stages are updated.

        Args:
            keep_producer (bool): Whether to keep the current producer. If :code:`True`,
                the producer will not be updated from the received context.

        Returns:
            tuple[bool, bool]: A tuple where the first element indicates if the worker
                should stop, and the second element indicates if any new context has been
                applied .

        Raises:
            AssertionError: If :code:`keep_producer` is :code:`True` but a new producer was
                reveived.
        """
        # receive context
        role, prod, proc, fn, done = self._recv_ctx_conn.recv()
        # keep the producer and only change the processing stages of the pipeline
        if keep_producer:
            assert prod is None, "Unexpected producer received while keeping the current producer."
        # update context
        self._role = role if role is not None else self._role
        self._producer = prod if prod is not None else self._producer
        self._processor = proc if proc is not None else self._processor
        self._finalizer = fn if fn is not None else self._finalizer
        # return done
        return done, (prod is not None) or (proc is not None) or (fn is not None)

    def _send_prog_to_tracker(
        self, role: WorkerRole | None, num_samples: int, producer_exhausted: bool, done: bool
    ) -> None:
        self._tracker_conn.send((self._rank, role, num_samples, producer_exhausted, done))

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
            dataset_info=self._dataset_info,
        )

        try:
            # initialize worker
            self._worker_init()

            # request a processing context
            while not self._request_new_ctx():
                # create producer iterator to avoid resetting when
                # new context is received during execution
                producer_iter = iter(self._producer)
                producer_exhausted = False

                # exhaust producer
                while not producer_exhausted:
                    num_samples = 0
                    last_update = time()

                    current_role = self._role
                    # create a stoppable producer that allows to dynamically
                    # interrupt the execution and apply the new context
                    controllable_producer = StoppableIterator(producer_iter)
                    # apply the processor to the producer and apply the finalizer
                    samples_iter = self._processor(controllable_producer)
                    work_iter = map(self._finalizer, samples_iter)

                    for _ in work_iter:
                        num_samples += 1

                        # check for a new context
                        done, new_ctx = self._check_ctx()

                        # stop worker
                        if done:
                            # send progress to tracker
                            self._send_prog_to_tracker(
                                current_role, num_samples, producer_exhausted, True
                            )

                            try:
                                # finalize worker
                                self._worker_finalize()
                            except Exception:  # pragma: not covered
                                ...
                            return

                        # apply new context
                        if new_ctx:
                            # stop the producer from generating further samples
                            # and exhaust the current samples generated by the producer
                            controllable_producer.stop()
                            for _ in work_iter:
                                num_samples += 1
                            # send progress to tracker
                            self._send_prog_to_tracker(
                                current_role, num_samples, producer_exhausted, False
                            )
                            # apply the new context
                            break

                        # send continuous updates to tracker
                        if (num_samples > 0) and (time() - last_update > 0.1):
                            self._send_prog_to_tracker(
                                current_role, num_samples, producer_exhausted, False
                            )
                            num_samples = 0
                            last_update = time()

                    else:
                        # producer exhausted
                        producer_exhausted = True

                # send final progress update before new requesting context
                self._send_prog_to_tracker(current_role, num_samples, producer_exhausted, False)

        except KeyboardInterrupt:  # pragma: not covered
            ...

        finally:
            self._send_prog_to_tracker(None, 0, False, True)
            try:
                # finalize worker
                self._worker_finalize()
            except Exception:  # pragma: not covered
                pass


class ProgressTracker(threading.Thread):
    """Progress Tracker Thread

    A thread that tracks the progress of multiprocessing workers, and updates a tqdm progress bar.

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
    def total_consumed_samples(self):
        """Returns the total number of samples consumed by all workers.

        This includes samples generated by workers in the :class:`WorkerRole.PROCESSOR` and
        :class:`WorkerRole.CONSUMER` roles.

        Returns:
            int: Total number of consumed samples.
        """
        return sum(samples[WorkerRole.PROCESSOR] for samples in self._num_samples) + sum(
            samples[WorkerRole.CONSUMER] for samples in self._num_samples
        )

    def recv_progress(self):
        """Receives progress updates from the workers through a multiprocessing pipe.

        Returns:
            tuple: A tuple containing the worker rank, role, number of samples processed,
            whether the producer is exhausted, and whether the worker has terminated.
        """
        rank, role, num_samples, producer_exhausted, done = self._recv_prog_conn.recv()
        return rank, role, num_samples, producer_exhausted, done

    @property
    def busy_workers(self) -> int:
        """Returns the number of workers that are currently busy.

        A worker is considered busy if its role is not None.

        Returns:
            int: Number of busy workers.
        """
        return sum(role is not None for role in self._roles)

    @property
    def _pbar_desc(self) -> str:
        """
        Returns a formatted description of the progress bar, including the current number
        of busy workers and their roles (processor, producer, consumer).

        Returns:
            str: A string describing the progress bar status.
        """
        counts = Counter(self._roles)
        return (
            f"Workers {self.busy_workers}/{self._num_workers} "
            f"(S={counts[WorkerRole.PROCESSOR]}, "
            f"P={counts[WorkerRole.PRODUCER]}, "
            f"C={counts[WorkerRole.CONSUMER]})"
        )

    def run(self):
        """Run progress tracker thread.

        Starts the thread and updates the tqdm progress bar based on progress reports from workers.

        It polls the progress pipe for updates, processes the reports, and computes throughput
        statistics. The progress bar is updated with the number of finished shards and the
        processing throughput.

        The thread continues running until all workers have stopped.
        """

        with tqdm(total=self._num_shards, desc=self._pbar_desc) as pbar:
            ema_dn = EMA(smoothing=0.3)
            ema_dt = EMA(smoothing=0.3)

            prev_total_samples = 0
            prev_update_time = pbar._time()

            while self._stopped_workers < self._num_workers:
                if self._recv_prog_conn.poll(timeout=0.05):
                    # receive progress update
                    rank, role, num_samples, producer_exhausted, done = self.recv_progress()

                    if role is not None:
                        # update worker role and number of processed samples
                        self._roles[rank] = role
                        self._num_samples[rank][role] += num_samples

                    if role in {WorkerRole.PROCESSOR, WorkerRole.PRODUCER}:
                        # update number of finished shards
                        self._finished_shards += int(producer_exhausted)

                    if done:
                        # update number of stopped workers
                        self._stopped_workers += 1
                        self._roles[rank] = None

                    total_samples = self.total_consumed_samples
                    update_time = pbar._time()

                    # exhaust progress updates before rendering
                    if ((update_time - prev_update_time) < 0.05) and self._recv_prog_conn.poll(
                        timeout=0.05
                    ):
                        continue

                    # compute sample throughput
                    dn = total_samples - prev_total_samples
                    dt = update_time - prev_update_time
                    throughput = ema_dn(dn) / max(ema_dt(dt), 1e-5)

                    # format total samples
                    formatting_string = "%d" if total_samples < 10**6 else "%.2e"
                    formatted_total_samples = formatting_string % total_samples
                    # update progress bar
                    pbar.set_description(self._pbar_desc, refresh=False)
                    pbar.set_postfix_str(
                        (
                            f"{self._queue.qsize()}q, "
                            f"{throughput:.02f}ex/s, "
                            f"{formatted_total_samples}ex"
                        ),
                        refresh=False,
                    )
                    # update values
                    prev_total_samples = total_samples
                    prev_update_time = update_time

                    # update the progress bar
                    pbar.update(self._finished_shards - pbar.n)

                else:
                    # refresh the progress bar
                    pbar.refresh()


class DynamicMultiprocessingRunner(object):
    """Manages and runs a set of worker processes to handle parallel data processing.

    This class coordinates multiple worker processes to process data in parallel,
    dynamically assigning tasks and handling context changes.

    The processing is carried out in two distinct stages:

    - **Stage 1:** Single-Shard Single-Worker
      Each dataset shard is assigned to a single worker process. This ensures that each worker
      handles only one shard at a time. The system tracks which workers are actively processing
      data.

      This stage has minimal communication overhead while fully utilizing all workers.

    - **Stage 2:** Single-Shard Multiple-Workers
      Begins after all shards have been assigned in Stage 1. In this stage, a single shard is
      processed by multiple workers. The workers are dynamically assigned as producers or consumers
      based on their status:
        - **Producers**: Add data to a queue.
        - **Consumers**: Process data from the queue.
      This approach maximizes the utilization of workers and maintains a continuous data flow until
      all shards are processed.

    By transitioning to Stage 2, the system ensures efficient and parallel processing of data,
    optimizing performance and resource usage.
    """

    def __init__(
        self,
        num_workers: int,
        prefetch_factor: int = 8,
        worker_init: Callable[[], Any] = do_nothing,
        worker_finalize: Callable[[], Any] = do_nothing,
    ) -> None:
        """Initialize the multiprocessing runner.

        Args:
            num_workers (int): The number of worker processes to create for parallel data
                processing.
            prefetch_factor (int, optional): The number of items that should be prefetched
                in Stage 2 when using a queue. Default is 8.
        """

        self._num_workers = num_workers
        self._prefetch = prefetch_factor

        self._worker_init = worker_init
        self._worker_finalize = worker_finalize

    def run(
        self,
        dataset_info: DatasetInfo,
        ds: IterableDataset,
        processor: Callable[[IndexedSamplesIterable], IndexedSamplesIterable],
        finalizer: Callable[[IndexedSamplesIterable], Any],
    ) -> None:
        """Execute data processing using the worker processes.

        Args:
            ds (IterableDataset): The dataset to process.
            processor (Callable[[IndexedSamplesIterable], IndexedSamplesIterable]): The function to
                process each sample.
            finalizer (Callable[[IndexedSamplesIterable], Any]): The function to finalize each
                sample.
        """

        with mp.Manager() as manager:
            # prepare the dataset
            num_shards = ds.n_shards
            ds = ds._prepare_ex_iterable_for_iteration()

            # create connection for workers to request new context
            req_ctx_conn, worker_req_ctx_conn = mp.Pipe(duplex=False)

            # create sample queue used in stage 2 but required for tracker
            queue = manager.Queue(maxsize=self._num_workers * self._prefetch)
            iterable = QueueIterator(queue, sentinel=None, timeout=None)

            # create the progress tracker thread
            tracker = ProgressTracker(num_shards, self._num_workers, queue)

            # create all workers
            workers = [
                Worker(
                    rank,
                    self._num_workers,
                    dataset_info,
                    tracker._send_prog_conn,
                    worker_req_ctx_conn,
                    self._worker_init,
                    self._worker_finalize,
                )
                for rank in range(self._num_workers)
            ]

            # start the tracker thread
            tracker.start()
            # start all workers
            for worker in workers:
                worker.start()

            # keep track of all running workers
            running_worker_ranks: set[int] = set()

            # stage 1: shards available
            # whenever a group is idleing give it a new shard to process
            for shard_id in range(num_shards):
                shard = ds.shard_data_sources(shard_id, num_shards)
                shard._event = manager.Event()
                # wait for worker to request new context
                rank = req_ctx_conn.recv()
                worker = workers[rank]
                # send new context to worker
                worker.send_ctx(WorkerRole.PROCESSOR, shard, processor, finalizer, False)
                running_worker_ranks.add(rank)

            # Stage 2: all shards being processed

            # keep track of producer and consumer workers
            producer_worker_rank: None | int = None
            consumer_worker_ranks: set[int] = set()

            while len(running_worker_ranks) > 0:
                # collect all idling workers
                idling_worker_ranks = {req_ctx_conn.recv()}
                while req_ctx_conn.poll():
                    idling_worker_ranks.add(req_ctx_conn.recv())

                # update running ranks
                running_worker_ranks -= idling_worker_ranks

                if len(running_worker_ranks) > 0:
                    if (producer_worker_rank is None) or (
                        producer_worker_rank in idling_worker_ranks
                    ):
                        # select a running rank as the producer
                        producer_worker_rank = next(iter(running_worker_ranks))
                        workers[producer_worker_rank].send_ctx(
                            WorkerRole.PRODUCER, None, _passthrough, queue.put, False
                        )

                else:
                    # no worker running that could act as producer
                    producer_worker_rank = None

                # use idling ranks as consumers
                for rank in idling_worker_ranks:
                    workers[rank].send_ctx(
                        WorkerRole.CONSUMER, iterable, processor, finalizer, False
                    )

                # update consumer worker ranks
                consumer_worker_ranks.update(idling_worker_ranks)

            # put sentinel signal to queue
            for _ in consumer_worker_ranks:
                queue.put(None)

            # send stop signal to all workers
            for _ in range(self._num_workers):
                rank = req_ctx_conn.recv()
                workers[rank].send_ctx(None, None, None, None, True)

            # wait for all workers to join
            for worker in workers:
                worker.join()
            # wait for tracker to join
            tracker.join()


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

    def __call__(self, ex_iterable):
        """Run the pipeline on the given example iterable.

        Args:
            ex_iterable: The example iterable to be processed by the pipeline.

        Returns:
            Iterable: Processed samples from the pipeline.
        """

        pipeline = self.copy()
        pipeline[0].ex_iterable = ex_iterable

        yield from pipeline[-1]


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
    ) -> None:
        """Initialize the dataset consumer.

        Args:
            fn (Callable[[Sample], Any]): The function to be applied to each sample in the dataset.
            num_proc (int, optional): The number of processes to use for parallel processing.
                Defaults to the number of cpus, meaning single-threaded execution.
        """
        self._fn = fn
        self._num_proc = num_proc
        self._prefetch = prefetch_factor

        self._initialize = initialize
        self._finalize = finalize

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

        # create the source dataaset that excludes the
        # pipeline processing steps
        src_ds = IterableDataset(ex_iterable=pipeline.src_iterable)

        return src_ds, pipeline.copy()

    def consume(self, ds: IterableDataset) -> None:
        """Process the dataset using worker processes.

        Args:
            ds (IterableDataset): The dataset to process.
        """

        if self._num_proc > 1:
            # prepare the dataset and function to apply
            src_ds, processor = self._prepare_dataset(ds)
            finalizer = partial(_drop_key_and_apply, fn=self._fn)

            # create the multiprocessing runner and run it
            runner = DynamicMultiprocessingRunner(
                num_workers=self._num_proc,
                prefetch_factor=self._prefetch,
                worker_init=self._initialize,
                worker_finalize=self._finalize,
            )
            runner.run(ds.info, src_ds, processor, finalizer)

        else:
            # set the worker info
            set_worker_info(rank=0, num_workers=1, seed=None, dataset_info=ds.info)

            try:
                # initialize the consumer
                self._initialize()
                # consume dataset
                for batch in ds.iter(batch_size=self._prefetch):
                    for sample in dict_of_lists_to_list_of_dicts(batch):
                        self._fn(sample)

            except KeyboardInterrupt:  # pragma: not covered
                raise

            finally:
                try:
                    # finalize the consumer
                    self._finalize()
                except Exception:  # pragma: not covered
                    ...

                # reset the worker info
                reset_worker_info()


class BaseDatasetWriter(ABC):
    """Base class for writing datasets to disk.

    This class provides a framework for saving a dataset to a specified directory.
    The folder structure and save format follow the Hugging Face :func:`save_to_disk`
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
    ) -> None:
        """Initialize the BaseDataWriter.

        Args:
            save_dir (str): Directory where the dataset will be saved.
            overwrite (bool, optional): Whether to overwrite existing files in the save directory.
                Defaults to False.
            num_proc (int, optional): Number of processes to use for dataset processing. Defaults
                to the number of CPU cores.
            prefetch_factor (int, optional): Number of samples to prefetch for improved
                performance. Defaults to 8.
        """
        self.save_dir = save_dir
        self.overwrite = overwrite
        self.num_proc = num_proc
        self.prefetch = prefetch_factor

    def _write_info(self, ds: IterableDataset) -> None:
        """Write dataset information to a JSON file in the save directory.

        Args:
            ds (IterableDataset): The dataset object containing metadata to be saved.
        """

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

    def _write_split(self, ds: IterableDataset | Dataset, save_dir: str) -> None:
        """Write the dataset split to the specified directory.

        Args:
            ds (IterableDataset | Dataset): The dataset or iterable dataset to be written.
            save_dir (str): The directory where the split data will be saved.
        """

        os.makedirs(save_dir, exist_ok=True)
        # convert dataset to iterable dataset
        if isinstance(ds, Dataset):
            ds = ds.to_iterable_dataset(self.num_proc)

        with chdir(save_dir):
            # write dataset to directory
            consumer = DatasetConsumer(
                self.write_sample,
                num_proc=self.num_proc,
                prefetch_factor=self.prefetch,
                initialize=self.initialize,
                finalize=self.finalize,
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
            if not self.overwrite:
                raise FileExistsError(
                    f"Output path `{self.save_dir}` already exists. "
                    f"Set `overwrite=True` to overwrite."
                )
            else:
                # delete existing directory
                shutil.rmtree(self.save_dir)

        # create the save directory
        os.makedirs(self.save_dir, exist_ok=False)

        if isinstance(ds, (DatasetDict, IterableDatasetDict)):
            # write all splits
            for key, split in ds.items():
                self._write_split(split, os.path.join(self.save_dir, key))
            # write dataset splits json
            with open(
                os.path.join(self.save_dir, datasets.config.DATASETDICT_JSON_FILENAME), "w+"
            ) as f:
                f.write(json.dumps({"splits": list(ds.keys())}))

        else:
            # save dataset to directory
            self._write_split(ds, self.save_dir)

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
    def initialize(self) -> None:
        """Abstract method for initializing the write process.

        Any setup tasks, such as creating necessary files or folders, should take place here.
        The working directory is temporarily set to the save directory during this method.
        """
        ...

    @abstractmethod
    def finalize(self) -> None:
        """Abstract method for finalizing the write process.

        This method should handle any cleanup or final write operations after all samples
        have been processed. The working directory is temporarily set to the save directory
        during this method.
        """
        ...

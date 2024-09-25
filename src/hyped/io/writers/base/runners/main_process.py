"""Main Process Runner Module.

This module provides the :class:`MainProcessRunner` class, which implements a data processing 
pipeline that runs entirely in the main thread. It processes data in batches, applies a 
user-defined function to each sample, and reports progress through an optional progress 
bar. The runner manages initialization and finalization tasks and handles progress reporting 
through callbacks.
"""

from queue import Queue
from time import time
from typing import Any, Callable

from datasets import IterableDataset

from hyped.common._worker import reset_worker_info, set_worker_info
from hyped.common.iterators import TimedIterator, ith_entries
from hyped.common.logging import get_logger
from hyped.common.typing import Sample

from ..callbacks.base import CallbackManager
from ..monitor import ProgressMonitor, ProgressReport, TimeReport
from .base import BaseRunner, WorkerRole

logger = get_logger(__name__)


class MainProcessRunner(BaseRunner):
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
        progress_report_interval: float,
        callback: CallbackManager,
    ) -> None:
        """Initialize the MainProcessRunner.

        Args:
            batch_size (int): The number of samples to process in each batch.
            initialize (Callable[[], Any]): Function to initialize the processing environment.
            finalize (Callable[[], Any]): Function to finalize the processing environment.
        """

        self._batch_size = batch_size
        self._initialize = initialize
        self._finalize = finalize
        self._report_interval = progress_report_interval
        self._callback = callback

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
        logger.info("Preparing to run data processing on dataset.")

        # set the worker info
        set_worker_info(rank=0, num_workers=1, seed=None)

        # prepare the dataset
        ds = ds._prepare_ex_iterable_for_iteration(batch_size=self._batch_size)
        logger.info(f"Dataset prepared with {ds.n_shards} shards.")

        # create the progress monitor
        monitor = ProgressMonitor(ds.n_shards, 1, Queue(), 0)

        try:
            # call start callback
            self._callback.on_start(monitor, ds)

            # initialize the consumer
            self._initialize()
            logger.info("Initialization complete.")

            monitor._mark_worker_ready(0)
            monitor._mark_worker_idling(0)

            num_samples = 0
            last_report = time()

            for shard_id in range(ds.n_shards):
                try:
                    monitor._mark_shard_in_progress(0, shard_id)
                    monitor._mark_worker_busy(0, WorkerRole.PROCESSOR)
                    self._callback.on_shard_in_progress(monitor, shard_id)

                    # get the dataset shard and time it
                    shard = ds.shard_data_sources(shard_id, ds.n_shards)
                    timed_shard = TimedIterator(iter(shard), smoothing=0.1)

                    # apply the function
                    work_iter = map(fn, ith_entries(timed_shard, i=1))
                    timed_work_iter = TimedIterator(iter(work_iter), smoothing=0.1)

                    # main worker loop
                    for _ in timed_work_iter:
                        num_samples += 1

                        now = time()
                        if now - last_report > self._report_interval:
                            report = ProgressReport(
                                timestamp=now,
                                num_samples=num_samples,
                                elapsed_time=now - last_report,
                                average_time=TimeReport(
                                    producer=timed_shard.average_time(),
                                    processor=timed_shard.average_time(),
                                    finalizer=timed_work_iter.average_time(),
                                ),
                                total_time=TimeReport(
                                    producer=timed_shard.total_time(),
                                    processor=timed_shard.total_time(),
                                    finalizer=timed_work_iter.total_time(),
                                ),
                            )
                            monitor._report_progress(0, report)
                            # reset
                            num_samples, last_report = 0, now

                    logger.info(f"Finished processing shard {shard_id}.")

                    monitor._mark_worker_completed(0)

                except KeyboardInterrupt:
                    # handle exception
                    monitor._mark_worker_canceled(0)
                    self._callback.on_shard_canceled(monitor, shard_id)
                    raise

                except Exception as e:
                    # handle exception
                    monitor._mark_worker_canceled(0)
                    self._callback.on_shard_canceled(monitor, shard_id)
                    # log
                    logger.error(f"Unexpected error during processing: {str(e)}.", exc_info=True)

                else:
                    # call shard complete when no error was detected
                    self._callback.on_shard_completed(monitor, shard_id)

        except KeyboardInterrupt:  # pragma: not covered
            logger.warning("Processing interrupted by user.")
            monitor._mark_worker_canceled(0)
            raise

        finally:
            # report worker done
            logger.info("Processing completed. Finalizing worker.")

            if num_samples > 0:
                now = time()
                report = ProgressReport(
                    timestamp=now,
                    num_samples=num_samples,
                    elapsed_time=now - last_report,
                    average_time=TimeReport(producer=0, processor=0, finalizer=0),
                    total_time=TimeReport(producer=0, processor=0, finalizer=0),
                )
                monitor._report_progress(0, report)

            try:
                # finalize the consumer
                self._finalize()
                logger.info("Worker finalized successfully.")
            except Exception as e:  # pragma: not covered
                logger.error(f"Error during worker finalization: {e}", exc_info=True)

            # stop worker
            monitor._mark_worker_idling(0)
            monitor._mark_worker_done(0)
            # reset the worker info
            reset_worker_info()
            # stopping
            monitor._mark_as_stopping()
            self._callback.on_stopping(monitor)
            # done
            monitor._mark_as_done()
            self._callback.on_done(monitor)

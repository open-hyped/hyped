import multiprocessing as mp
import os
from time import sleep
from unittest.mock import MagicMock, call, patch

import datasets
import pytest
from datasets import Dataset, IterableDatasetDict
from datasets.iterable_dataset import MappedExamplesIterable, TypedExamplesIterable
from sharedmock.mock import SharedMock

from hyped.io.writers.base import (
    BaseDatasetWriter,
    DatasetConsumer,
    DynamicMultiprocessingRunner,
    Worker,
    WorkerRole,
    _passthrough,
)


@pytest.fixture
def mock_connections():
    """Fixture for mock connections used in Worker."""
    return mp.Pipe(duplex=False), mp.Pipe(duplex=False)


@pytest.fixture
def worker(mock_connections):
    """Fixture to create a Worker instance."""
    ((req_ctx_conn, worker_req_ctx_conn), (tracker_conn, worker_tracker_conn)) = mock_connections

    return Worker(
        rank=0,
        num_workers=1,
        req_ctx_conn=worker_req_ctx_conn,
        tracker_conn=worker_tracker_conn,
    )


class TestWorker:
    def test_request_new_ctx(self, worker, mock_connections):
        req_ctx_conn, worker_req_ctx_conn = mock_connections[0]

        role = WorkerRole.PROCESSOR
        producer = "PRODUCER"
        processor = "PROCESSOR"
        finalizer = "FINALIZER"

        worker._recv_ctx_done = MagicMock()
        # send new context before worker request to avoid deadlock
        worker.send_ctx(role, producer, processor, finalizer, False)
        worker._request_new_ctx()

        # check if worker asked for new context
        assert req_ctx_conn.recv() == worker._rank

        # check worker context
        assert worker._role == role
        assert worker._producer == producer
        assert worker._processor == processor
        assert worker._finalizer == finalizer

    @patch("hyped.io.writers.base.set_worker_info")
    def test_run(self, mock_set_worker_info, worker):
        processor_marker = MagicMock()

        worker._tracker_update_interval = 0.0
        # set up worker
        worker._worker_init = MagicMock()
        worker._worker_finalize = MagicMock()
        worker._producer = [MagicMock(), MagicMock(), MagicMock()]
        worker._processor = lambda x: map(processor_marker, x)
        worker._finalizer = MagicMock()
        # mock request new context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)

        # run worker
        worker.run()

        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        worker._finalizer.assert_has_calls(
            [call(processor_marker(x)) for x in worker._producer], any_order=True
        )
        worker._worker_init.assert_called_once()
        worker._worker_finalize.assert_called_once()

    @patch("hyped.io.writers.base.set_worker_info")
    def test_run_with_ctx_update(self, mock_set_worker_info, worker):
        processor_marker = MagicMock()
        # set up worker
        worker._producer = [MagicMock(), MagicMock(), MagicMock()]
        worker._processor = lambda x: map(processor_marker, x)
        worker._finalizer = MagicMock()
        # mock request new and check context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)

        worker._recv_ctx_conn = MagicMock()
        worker._recv_ctx_conn.poll = MagicMock(return_value=True)

        worker._recv_ctx = MagicMock(return_value=(None, None, None, None, False))

        worker.run()

        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        assert len(worker._recv_ctx.mock_calls) == 3
        worker._finalizer.assert_has_calls(
            [call(processor_marker(x)) for x in worker._producer], any_order=True
        )

    @patch("hyped.io.writers.base.set_worker_info")
    def test_run_with_abort(self, mock_set_worker_info, worker):
        processor_marker = MagicMock()
        # set up worker
        worker._producer = [MagicMock(), MagicMock(), MagicMock()]
        worker._processor = lambda x: map(processor_marker, x)
        worker._finalizer = MagicMock()
        # mock request new and check context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)

        worker._recv_ctx_conn = MagicMock()
        worker._recv_ctx_conn.poll = MagicMock(return_value=True)

        worker._recv_ctx = MagicMock(return_value=(None, None, None, None, True))

        worker.run()

        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        assert len(worker._recv_ctx.mock_calls) == 1
        worker._finalizer.assert_has_calls(
            [call(processor_marker(worker._producer[0]))], any_order=True
        )

    @patch("hyped.io.writers.base.set_worker_info")
    def test_error_logging(self, mock_set_worker_info, worker):
        # set up worker
        worker._producer = [MagicMock(), MagicMock(), MagicMock()]
        worker._processor = _passthrough
        worker._finalizer = MagicMock(side_effect=RuntimeError)
        # mock request new and check context
        worker._logger = MagicMock()
        worker._logger.error = MagicMock()
        # mock request new context and run worker
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        worker.run()
        # make sure errors were logged
        assert len(worker._logger.error.mock_calls) == 3

        worker._logger.error.reset_mock()
        worker._worker_init = MagicMock(side_effect=RuntimeError)
        # mock request new context and run worker
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        worker.run()
        # make sure errors were logged
        worker._logger.error.assert_called_once()

        worker._logger.error.reset_mock()
        worker._worker_finalize = MagicMock(side_effect=RuntimeError)
        # mock request new context and run worker
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        worker.run()
        # make sure errors were logged
        assert len(worker._logger.error.mock_calls) == 2


class TestDynamicMultiprocessingRunner:
    @pytest.mark.parametrize("num_shards", [1, 2, 3])
    @pytest.mark.parametrize("num_samples", [20])
    def test_run(self, num_shards, num_samples):
        # create mock dataset
        samples = {"obj": [i for i in range(num_samples)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(num_shards)
        # create mock processor and finalizer
        processor = _passthrough
        finalizer = SharedMock()
        # run dynamic multiprocessing runner
        runner = DynamicMultiprocessingRunner(
            num_workers=2, disable_progress_bar=True, progress_update_interval=0.0
        )
        runner.run(ds, processor, finalizer)
        # make sure all samples have been processed
        finalizer.assert_has_calls(
            [call((0, {"obj": i})) for i in range(num_samples)], same_order=False
        )


def _double_fn(x):
    sleep(0.1)
    return {"obj": x["obj"] * 2}


class TestDatasetConsumer:
    @pytest.mark.parametrize("num_proc", [1, 2, 3])
    @pytest.mark.parametrize("num_samples", [20])
    def test_consume(self, num_proc, num_samples):
        # create mock dataset
        samples = {"obj": [i for i in range(num_samples)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(4)

        # create mock function
        fn = SharedMock()

        # create dataset consumer and consumer dataset
        consumer = DatasetConsumer(fn=fn, num_proc=num_proc)
        consumer.consume(ds)

        # make sure all samples have been processed
        fn.assert_has_calls([call({"obj": i}) for i in range(num_samples)], same_order=False)

    @pytest.mark.parametrize("num_proc", [1, 2, 3])
    @pytest.mark.parametrize("num_samples", [20])
    def test_consume_with_pipeline(self, num_proc, num_samples):
        # create mock dataset
        samples = {"obj": [i for i in range(num_samples)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(4)

        # apply map function
        ds = ds.map(_double_fn)

        # create mock function
        fn = SharedMock()

        # create dataset consumer and consumer dataset
        consumer = DatasetConsumer(fn=fn, num_proc=num_proc)
        consumer.consume(ds)

        # make sure all samples have been processed
        fn.assert_has_calls([call({"obj": i * 2}) for i in range(num_samples)], same_order=False)

    def test_prepare_dataset(self):
        ds = Dataset.from_dict({"obj": [0]})
        ds = ds.to_iterable_dataset(1)
        # apply map function
        mapped_ds = ds.map(_double_fn)

        # call prepare dataset
        consumer = DatasetConsumer(fn=MagicMock(), num_proc=2)
        src_ds, pipeline = consumer._prepare_dataset(mapped_ds)

        # check output
        assert isinstance(src_ds._ex_iterable, type(ds._ex_iterable))
        assert isinstance(pipeline[0], TypedExamplesIterable)
        assert isinstance(pipeline[1], MappedExamplesIterable)


class TestBaseDatasetWriter:
    def test_write_split(self, tmp_path):
        ds = Dataset.from_dict({"obj": [0]})
        ds = ds.to_iterable_dataset(1)

        class MockDatasetWriter(BaseDatasetWriter):
            initialize = MagicMock()
            write_sample = MagicMock()
            finalize = MagicMock()

        with (
            patch("hyped.io.writers.base.DatasetConsumer") as mock,
            patch("hyped.io.writers.base.partial") as partial_mock,
        ):
            writer = MockDatasetWriter(save_dir=tmp_path, overwrite=True)
            writer._write_dataset(ds, save_dir=tmp_path)

            # make sure the consumer is created and called correctly
            mock.assert_called_once_with(
                writer.write_sample,
                num_proc=writer._num_proc,
                prefetch_factor=writer._prefetch,
                initialize=partial_mock(writer.initialize, ds.info),
                finalize=partial_mock(writer.finalize, ds.info),
                tqdm_update_interval=writer._tqdm_update_interval,
                disable_tqdm=writer._disable_tqdm,
            )
            mock().consume.assert_called_once_with(ds)

            # check the output directory
            files = os.listdir(tmp_path)
            assert len(files)
            assert datasets.config.DATASET_STATE_JSON_FILENAME in files
            assert datasets.config.DATASET_INFO_FILENAME in files

    @pytest.mark.parametrize("path_exists", [True, False])
    def test_write_dataset(self, path_exists, tmp_path):
        tmp_path = tmp_path if path_exists else os.path.join(tmp_path, "data")

        ds = Dataset.from_dict({"obj": [0]})
        ds = ds.to_iterable_dataset(1)
        ds._format_kwargs = {"key": 0}

        class MockDatasetWriter(BaseDatasetWriter):
            initialize = MagicMock()
            write_sample = MagicMock()
            finalize = MagicMock()

        with patch("hyped.io.writers.base.BaseDatasetWriter._write_dataset") as write_split_mock:
            writer = MockDatasetWriter(save_dir=tmp_path, overwrite=path_exists)
            writer.write(ds)

            write_split_mock.assert_called_once_with(ds, tmp_path)

    @pytest.mark.parametrize("path_exists", [True, False])
    def test_write_dataset_dict(self, path_exists, tmp_path):
        tmp_path = tmp_path if path_exists else os.path.join(tmp_path, "data")

        ds = Dataset.from_dict({"obj": [0]})
        ds = ds.to_iterable_dataset(1)
        ds = IterableDatasetDict({"train": ds, "test": ds})

        class MockDatasetWriter(BaseDatasetWriter):
            initialize = MagicMock()
            write_sample = MagicMock()
            finalize = MagicMock()

        with patch("hyped.io.writers.base.BaseDatasetWriter._write_dataset") as write_split_mock:
            writer = MockDatasetWriter(save_dir=tmp_path, overwrite=path_exists)
            writer.write(ds)

            write_split_mock.assert_has_calls(
                [call(split, os.path.join(tmp_path, key)) for key, split in ds.items()],
                any_order=True,
            )

        # check if dataset dict json exists in output directory
        assert datasets.config.DATASETDICT_JSON_FILENAME in os.listdir(tmp_path)

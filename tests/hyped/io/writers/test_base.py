import multiprocessing as mp
import os
from time import sleep
from unittest.mock import MagicMock, call, patch

import datasets
import pytest
from datasets import Dataset, IterableDatasetDict
from sharedmock.mock import SharedMock

from hyped.io.writers.base import (
    BaseDatasetWriter,
    DatasetConsumer,
    DynamicMultiprocessingRunner,
    Serializer,
    ShardingController,
    ShardingStrategy,
    Worker,
    WorkerRole,
    _parse_size,
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

        # mock context conn receiver to avoid deadlock
        parent_ctx_conn_recv = worker._parent_ctx_conn.recv
        worker._parent_ctx_conn.recv = MagicMock()

        # send new context before worker request to avoid deadlock in worker
        worker.send_ctx(role, producer, processor, finalizer, False)
        worker._parent_ctx_conn.recv.assert_called_once()

        # request new context
        worker._request_new_ctx()

        # check if worker asked for new context and accepted the send context
        assert req_ctx_conn.recv() == worker._rank
        assert parent_ctx_conn_recv() is True

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

        worker._child_ctx_conn = MagicMock()
        worker._child_ctx_conn.poll = MagicMock(return_value=True)

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

        worker._child_ctx_conn = MagicMock()
        worker._child_ctx_conn.poll = MagicMock(return_value=True)

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
        worker._processor = lambda x: x
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


class TestSerializer:
    def test_serialization_deserialization(self):
        serializer = Serializer(batch_size=5)
        samples = [{"key": i} for i in range(32)]

        # Serialize and deserialize the samples
        serialized = list(serializer.serialize(samples))
        deserialized = list(serializer.deserialize(serialized))

        # Check that the deserialized output matches the original samples
        assert deserialized == samples, "Deserialized output does not match the original samples"


class TestDynamicMultiprocessingRunner:
    @pytest.mark.parametrize("num_shards", [1, 2, 3])
    @pytest.mark.parametrize("num_samples", [20])
    def test_run(self, num_shards, num_samples):
        # create mock dataset
        samples = {"obj": [i for i in range(num_samples)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(num_shards)
        # create mock processor and finalizer
        fn = SharedMock()
        # run dynamic multiprocessing runner
        runner = DynamicMultiprocessingRunner(
            num_workers=2, disable_progress_bar=True, progress_update_interval=0.0
        )
        runner.run(ds, fn)
        # make sure all samples have been processed
        fn.assert_has_calls([call({"obj": i}) for i in range(num_samples)], same_order=False)

    @pytest.mark.parametrize("nested", [0, 1, 2])
    def test_prepare_dataset(self, nested):
        ds = Dataset.from_dict({"obj": [0]})
        ds = ds.to_iterable_dataset(1)
        # apply map function
        mapped_ds = ds
        for _ in range(nested):
            mapped_ds = mapped_ds.map(_double_fn)

        # call prepare dataset
        runner = DynamicMultiprocessingRunner(num_workers=2)
        src_ex_it, pipeline = runner._prepare_dataset(mapped_ds)

        # check output
        assert isinstance(src_ex_it, type(ds._ex_iterable))

        # test pipeline output
        expected = list(pipeline(src_ex_it))
        actual = list(ds)
        assert actual == expected


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


class TestParseSize:
    def test_parse_size_valid_units(self):
        assert _parse_size("5GB") == 5 * 1024**3
        assert _parse_size("200MB") == 200 * 1024**2
        assert _parse_size("1.5KB") == int(1.5 * 1024)
        assert _parse_size("2TB") == 2 * 1024**4
        assert _parse_size("300B") == 300

    def test_parse_size_with_spaces(self):
        assert _parse_size(" 5 GB ") == 5 * 1024**3
        assert _parse_size(" 200 MB") == 200 * 1024**2
        assert _parse_size("1.5 KB") == int(1.5 * 1024)

    def test_parse_size_invalid_units(self):
        with pytest.raises(ValueError):
            _parse_size("5XYZ")
        with pytest.raises(ValueError):
            _parse_size("NotASize")

    def test_parse_size_invalid_value(self):
        with pytest.raises(ValueError):
            _parse_size("abcMB")
        with pytest.raises(ValueError):
            _parse_size("5.5.5GB")

    def test_parse_size_edge_cases(self):
        assert _parse_size("0KB") == 0
        assert _parse_size("0B") == 0

        with pytest.raises(ValueError):
            _parse_size("")

        with pytest.raises(ValueError):
            _parse_size("   ")


class TestShardingController:
    def test_initialization_valid_parameters(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        controller = ShardingController(
            is_multi_processed=True,
            sharding_strategy=ShardingStrategy.FILE_SIZE,
            max_shard_size=1024,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
        )

        assert controller._is_multi_processed is True
        assert controller._sharding_strategy == ShardingStrategy.FILE_SIZE
        assert controller._max_shard_size == 1024
        assert controller._sample_size_key is None

    def test_initialization_invalid_sample_size_key(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        with pytest.raises(ValueError):
            ShardingController(
                is_multi_processed=True,
                sharding_strategy=ShardingStrategy.SAMPLE_ITEM,
                max_shard_size=1024,
                sample_size_key=None,
                initialize_shard=initialize_shard,
                finalize_shard=finalize_shard,
            )

    def test_initialization_invalid_max_shard_size_for_non_file_size(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        with pytest.raises(ValueError):
            ShardingController(
                is_multi_processed=True,
                sharding_strategy=ShardingStrategy.SAMPLE_ITEM,
                max_shard_size="5GB",
                sample_size_key="key",
                initialize_shard=initialize_shard,
                finalize_shard=finalize_shard,
            )

    def test_warning_on_sample_size_key_for_non_sample_item(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        with pytest.warns(UserWarning):
            ShardingController(
                is_multi_processed=True,
                sharding_strategy=ShardingStrategy.FILE_SIZE,
                max_shard_size=1024,
                sample_size_key="key",
                initialize_shard=initialize_shard,
                finalize_shard=finalize_shard,
            )

    @pytest.mark.parametrize("is_multi_processed", [True, False])
    def test_initialize_and_finalize(self, is_multi_processed):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        controller = ShardingController(
            is_multi_processed=is_multi_processed,
            sharding_strategy=ShardingStrategy.FILE_SIZE,
            max_shard_size=1024,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
        )

        for i in range(3):
            initialize_shard.reset_mock()
            finalize_shard.reset_mock()

            controller.initialize()
            initialize_shard.assert_called_once_with(i)

            controller.finalize()
            finalize_shard.assert_called_once()

        # cannot initialize new shard without finalize in between
        controller.initialize()
        with pytest.raises(AssertionError):
            controller.initialize()
        # finalize
        controller.finalize()

        # shard finalizer is not called when no shard is initialized
        finalize_shard.reset_mock()
        controller.finalize()
        assert not finalize_shard.called

    def test_sample_count_strategy(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        max_shard_size = 8
        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.SAMPLE_COUNT,
            max_shard_size=max_shard_size,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
        )
        # initialize the controller
        controller.initialize()
        # reset the mocks
        initialize_shard.reset_mock()
        finalize_shard.reset_mock()

        for _ in range(max_shard_size):
            controller.callback({})
            controller.update(42)
            # no new shard needed to be generated yet
            assert not finalize_shard.called
            assert not initialize_shard.called

        # this update should kick of a new shard
        controller.callback({})
        controller.update(42)

        finalize_shard.assert_called_once()
        initialize_shard.assert_called_once()

    def test_sample_item_strategy(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        key = MagicMock()
        key.index_example = MagicMock(return_value=42)

        max_shard_size = 153
        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.SAMPLE_ITEM,
            max_shard_size=max_shard_size,
            sample_size_key="mock",
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
        )
        controller._sample_size_key = key
        # initialize the controller
        controller.initialize()
        # reset the mocks
        initialize_shard.reset_mock()
        finalize_shard.reset_mock()

        for _ in range(max_shard_size // 42 + 1):
            controller.callback({})
            controller.update(1)
            # no new shard needed to be generated yet
            assert not finalize_shard.called
            assert not initialize_shard.called

        # this update should kick of a new shard
        controller.callback({})
        controller.update(1)

        finalize_shard.assert_called_once()
        initialize_shard.assert_called_once()

    def test_file_size_strategy(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        max_shard_size = 153
        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.FILE_SIZE,
            max_shard_size=max_shard_size,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
        )
        # initialize the controller
        controller.initialize()
        # reset the mocks
        initialize_shard.reset_mock()
        finalize_shard.reset_mock()

        for _ in range(max_shard_size // 42 + 1):
            controller.callback({})
            controller.update(42)
            # no new shard needed to be generated yet
            assert not finalize_shard.called
            assert not initialize_shard.called

        # this update should kick of a new shard
        controller.callback({})
        controller.update(42)

        finalize_shard.assert_called_once()
        initialize_shard.assert_called_once()


class TestBaseDatasetWriter:
    @pytest.mark.parametrize("with_sharding", [True, False])
    def test_write_split(self, tmp_path, with_sharding):
        ds = Dataset.from_dict({"obj": [0]})
        ds = ds.to_iterable_dataset(1)

        class MockDatasetWriter(BaseDatasetWriter):
            write_sample = MagicMock()
            initialize = MagicMock()
            finalize = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        with (
            patch("hyped.io.writers.base.DatasetConsumer") as consumer_mock,
            patch("hyped.io.writers.base.ShardingController") as sharding_mock,
        ):
            sharding_mock.is_active = with_sharding

            writer = MockDatasetWriter(save_dir=tmp_path, overwrite=True)
            writer._write_dataset(ds, save_dir=tmp_path)

            # check sharding strategy
            sharding_mock.assert_called_once()

            # check dataset consumer
            consumer_mock.assert_called_once()
            consumer_mock().consume.assert_called_once_with(ds)
            # get arguments to consumer initializer
            fn = consumer_mock.mock_calls[0].args[0]
            init = consumer_mock.mock_calls[0].kwargs["initialize"]
            finalize = consumer_mock.mock_calls[0].kwargs["finalize"]

            mock_sample = MagicMock()
            # apply function
            fn(mock_sample)
            # make sure that the callback is called first, then the write sample
            # and finally the update function
            if with_sharding:
                sharding_mock().callback.assert_called_once_with(mock_sample)
                MockDatasetWriter.write_sample.assert_called_once_with(sharding_mock().callback())
                sharding_mock().update(MockDatasetWriter.write_sample())
            else:
                MockDatasetWriter.write_sample.assert_called_once_with(sharding_mock().callback())

            # make sure the sharding initializers is called
            init()
            sharding_mock().initialize.assert_called_once()
            MockDatasetWriter.initialize.assert_called_once()

            # make sure the sharding finalizers is called
            finalize()
            sharding_mock().finalize.assert_called_once()
            MockDatasetWriter.finalize.assert_called_once()

            # check the output directory
            assert set(os.listdir(tmp_path)) == {
                datasets.config.DATASET_STATE_JSON_FILENAME,
                datasets.config.DATASET_INFO_FILENAME,
            }

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
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

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
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        with patch("hyped.io.writers.base.BaseDatasetWriter._write_dataset") as write_split_mock:
            writer = MockDatasetWriter(save_dir=tmp_path, overwrite=path_exists)
            writer.write(ds)

            write_split_mock.assert_has_calls(
                [call(split, os.path.join(tmp_path, key)) for key, split in ds.items()],
                any_order=True,
            )

        # check if dataset dict json exists in output directory
        assert datasets.config.DATASETDICT_JSON_FILENAME in os.listdir(tmp_path)

import json
import multiprocessing as mp
from unittest.mock import MagicMock, call, patch

import pytest
from datasets import Dataset
from sharedmock.mock import SharedMock

from hyped.io.writers.base.callbacks.base import CallbackManager
from hyped.io.writers.base.runners.base import WorkerRole
from hyped.io.writers.base.runners.multi_process_runner import (
    DynamicMultiprocessingRunner,
    MessageType,
    Serializer,
    Worker,
)


class TestWorker:
    @pytest.fixture
    def worker_init(self):
        return MagicMock()

    @pytest.fixture
    def worker_finalizer(self):
        return MagicMock()

    @pytest.fixture
    def msg_pipe(self):
        return mp.Pipe(duplex=False)

    @pytest.fixture
    def worker(self, msg_pipe, worker_init, worker_finalizer):
        _, send_msg_conn = msg_pipe

        return Worker(
            rank=0,
            num_workers=1,
            send_msg_conn=send_msg_conn,
            progress_report_interval=0.0,
            worker_init=worker_init,
            worker_finalize=worker_finalizer,
        )

    def test_request_new_ctx(self, msg_pipe, worker):
        recv_msg, _ = msg_pipe

        role = WorkerRole.PRODUCER
        producer = "PRODUCER"
        processor = "PROCESSOR"
        finalizer = "FINALIZER"

        # mock context conn receiver to avoid deadlock
        worker._parent_ctx_conn.recv = MagicMock()
        # send new context before worker request to avoid deadlock in worker
        worker.send_ctx(role, producer, processor, finalizer, False)
        worker._parent_ctx_conn.recv.assert_called_once()

        # request new context
        worker._request_new_ctx()

        # make sure the worker send a request context message
        assert recv_msg.poll(timeout=0.1)
        msg = json.loads(recv_msg.recv_bytes().decode("utf-8"))
        assert msg["rank"] == worker._rank
        assert msg["type"] == MessageType.CTX_REQUEST.value

        # check if new context was applied
        assert worker._role == role
        assert worker._producer == producer
        assert worker._processor == processor
        assert worker._finalizer == finalizer

    @patch("hyped.io.writers.base.runners.multi_process_runner.set_worker_info")
    def test_run(self, mock_set_worker_info, worker):
        processor_marker = MagicMock()

        # set up worker
        worker._role = WorkerRole.PROCESSOR
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

    @patch("hyped.io.writers.base.runners.multi_process_runner.set_worker_info")
    def test_run_with_ctx_update(self, mock_set_worker_info, worker):
        processor_marker = MagicMock()
        # set up worker
        worker._role = WorkerRole.PROCESSOR
        worker._producer = [MagicMock(), MagicMock(), MagicMock()]
        worker._processor = lambda x: map(processor_marker, x)
        worker._finalizer = MagicMock()
        # mock request new and check context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)

        worker._child_ctx_conn = MagicMock()
        worker._child_ctx_conn.poll = MagicMock(return_value=True)

        worker._recv_ctx = MagicMock(return_value=(WorkerRole.PROCESSOR, None, None, None, False))

        worker.run()

        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        assert len(worker._recv_ctx.mock_calls) == 3
        worker._finalizer.assert_has_calls(
            [call(processor_marker(x)) for x in worker._producer], any_order=True
        )

    @patch("hyped.io.writers.base.runners.multi_process_runner.set_worker_info")
    def test_run_with_abort(self, mock_set_worker_info, worker):
        processor_marker = MagicMock()
        # set up worker
        worker._role = WorkerRole.PROCESSOR
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

    @patch("hyped.io.writers.base.runners.multi_process_runner.set_worker_info")
    def test_error_logging(self, mock_set_worker_info, worker):
        # set up worker
        worker._role = WorkerRole.PROCESSOR
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


def _double_fn(x):
    return {"obj": x["obj"] * 2}


class TestDynamicMultiprocessingRunner:
    @pytest.fixture
    def ds(self):
        # create mock dataset
        samples = {"obj": [i for i in range(20)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(3)
        return ds

    @pytest.fixture
    def runner(self):
        # run dynamic multiprocessing runner
        return DynamicMultiprocessingRunner(
            num_workers=2,
            prefetch_factor=8,
            worker_init=SharedMock(),
            worker_finalize=SharedMock(),
            progress_report_interval=0.0,
            callback=CallbackManager([]),
        )

    @pytest.mark.parametrize("num_shards", [1, 2, 3])
    def test_run(self, num_shards, runner):
        # create mock dataset
        samples = {"obj": [i for i in range(20)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(num_shards)
        # create mock processor and finalizer
        fn = SharedMock()
        runner.run(ds, fn)
        # make sure all samples have been processed
        fn.assert_has_calls([call(sample) for sample in ds], same_order=False)

    def test_run_with_keyboard_interrupt(self, runner, ds):
        def raise_exc(sample):
            raise KeyboardInterrupt()

        runner.run(ds, raise_exc)

    @pytest.mark.parametrize("nested", [0, 1, 2])
    def test_prepare_dataset(self, nested, runner, ds):
        # apply map function
        mapped_ds = ds
        for _ in range(nested):
            mapped_ds = mapped_ds.map(_double_fn)

        src_ex_it, pipeline = runner._prepare_dataset(mapped_ds)

        # check output
        assert isinstance(src_ex_it, type(ds._ex_iterable))

        # test pipeline output
        expected = list(pipeline(src_ex_it))
        actual = list(mapped_ds)
        assert actual == expected

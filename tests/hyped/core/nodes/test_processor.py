from unittest.mock import call

import pytest
from datasets import Features, Value

from hyped.core.nodes.processor import IOContext
from tests.hyped.core.mock import MockProcessor


class TestBaseDataProcessor:
    def test_properties(self):
        # create mock processor
        proc = MockProcessor()
        # check config and input keys property
        assert isinstance(proc.config, MockProcessor.Config)
        assert proc.required_input_keys == {"a", "b"}

    @pytest.mark.asyncio
    async def test_batch_process(self):
        # create mock instance
        proc = MockProcessor()
        # create dummy inputs
        rank = 0
        index = list(range(10))
        batch = {"a": index, "b": index}
        io_ctx = IOContext(
            node_id=-1,
            inputs=Features({"a": Value("int32"), "b": Value("int32")}),
            outputs=Features({"y": Value("int64")}),
        )
        # run batch process
        out_batch = await proc.batch_process(batch, index, rank, io_ctx)

        # check output
        assert out_batch == {"y": [0] * 10}
        # make sure the process function was called for each input sample
        calls = [call({"a": i, "b": i}, i, rank, io_ctx) for i in index]
        proc.process.assert_has_calls(calls, any_order=True)

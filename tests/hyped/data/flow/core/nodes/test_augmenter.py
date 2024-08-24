from itertools import chain
from unittest.mock import call

import pytest
from datasets import Features, Value

from hyped.data.flow.core.nodes.processor import IOContext
from tests.hyped.data.flow.core.mock import MockAugmenter


class TestDataAugmenter:
    @pytest.mark.asyncio
    async def test_batch_process(self):
        # create the mock instance
        augmenter = MockAugmenter()
        # create dummy inputs
        rank = 0
        index = list(range(10))
        batch = {"x": index}
        io_ctx = IOContext(
            node_id=-1,
            inputs=Features({"x": Value("int32")}),
            outputs=Features({"y": Value("int32")}),
        )
        # run batch process
        out_batch, trace_index = await augmenter.batch_process(
            batch, index, rank, io_ctx
        )
        # check output
        assert out_batch == {"y": [0] * 10 * 2}
        assert trace_index == list(chain.from_iterable(zip(index, index)))
        # make sure the process function was called for each input sample
        calls = [call({"x": i}, i, rank, io_ctx) for i in index]
        augmenter.process.assert_has_calls(calls, any_order=True)

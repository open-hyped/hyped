from unittest.mock import MagicMock, call

import pytest
from datasets import Features, Value

from hyped.core.nodes.processor import IOContext
from tests.hyped.core.mock import MockAugmenter


class TestDataAugmenter:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("k", range(0, 5))
    async def test_batch_process(self, k):
        # create the mock instance
        augmenter = MockAugmenter()
        augmenter.process = MagicMock(return_value=[{"y": i} for i in range(k)])

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
        out_batch, trace_index = await augmenter.batch_process(batch, index, rank, io_ctx)
        # check output
        assert out_batch == {"y": [i % k for i in range(10 * k)]}
        assert all(i == j // k for j, i in enumerate(trace_index))
        # make sure the process function was called for each input sample
        calls = [call({"x": i}, i, rank, io_ctx) for i in index]
        augmenter.process.assert_has_calls(calls, any_order=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("k", range(0, 5))
    async def test_batch_process_with_async(self, k):
        class AsyncMockAugmenter(MockAugmenter):
            async def process(self, *args, **kwargs):
                for i in range(k):
                    yield {"y": i}

        # create the mock instance
        augmenter = AsyncMockAugmenter()
        augmenter.process = MagicMock(wraps=augmenter.process)

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
        out_batch, trace_index = await augmenter.batch_process(batch, index, rank, io_ctx)
        # check output
        assert out_batch == {"y": [i % k for i in range(10 * k)]}
        assert all(i == j // k for j, i in enumerate(trace_index))
        # make sure the process function was called for each input sample
        calls = [call({"x": i}, i, rank, io_ctx) for i in index]
        augmenter.process.assert_has_calls(calls, any_order=True)

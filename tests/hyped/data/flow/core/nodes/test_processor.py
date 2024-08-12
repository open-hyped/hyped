from unittest.mock import MagicMock, call

import pytest
from datasets import Features, Value
from typing_extensions import Annotated

from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import (
    CheckFeatureEquals,
    InputRefs,
    InputRefsContainer,
)
from hyped.data.flow.core.refs.outputs import OutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class MockOutputRefs(OutputRefs):
    out: Annotated[FeatureRef, OutputFeature(Value("int32"))]


class MockInputRefs(InputRefs):
    x: Annotated[FeatureRef, CheckFeatureEquals(Value("int32"))]


class MockProcessorConfig(BaseDataProcessorConfig):
    c: float = 0.0


class MockProcessor(
    BaseDataProcessor[MockProcessorConfig, MockInputRefs, MockOutputRefs]
):
    process = MagicMock(return_value={"out": 0})


class TestBaseDataProcessor:
    def test_init(self):
        a = MockProcessor()
        b = MockProcessor(MockProcessorConfig())
        # test default values
        assert a.config.c == b.config.c == 0.0

        a = MockProcessor(c=1.0)
        b = MockProcessor(MockProcessorConfig(c=1.0))
        # test setting value
        assert a.config.c == b.config.c == 1.0

    def test_properties(self):
        # create mock processor
        proc = MockProcessor()
        # check config and input keys property
        assert isinstance(proc.config, MockProcessorConfig)
        assert proc.required_input_keys == {"x"}

    @pytest.mark.asyncio
    async def test_batch_process(self):
        # create mock instance
        proc = MockProcessor()
        # create dummy inputs
        rank = 0
        index = list(range(10))
        batch = {"x": index}
        io_ctx = IOContext(
            node_id=-1,
            inputs=Features({"x": Value("int32")}),
            outputs=Features({"out": Value("int32")}),
        )
        # run batch process
        out_batch = await proc.batch_process(batch, index, rank, io_ctx)
        # check output
        assert out_batch == {"out": [0] * 10}
        # make sure the process function was called for each input sample
        calls = [call({"x": i}, i, rank, io_ctx) for i in index]
        proc.process.assert_has_calls(calls, any_order=True)

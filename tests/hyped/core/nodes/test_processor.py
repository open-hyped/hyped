from inspect import Parameter, Signature, _ParameterKind
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest

from hyped.core.nodes.base import ProcessMode, RunContext
from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from hyped.core.typing import Bool, Int


class MockConfig(BaseDataProcessorConfig):
    @classmethod
    @property
    def type_id(self) -> str:
        return str(uuid4())


class TestBaseDataProcessor:
    def test_signature(self) -> None:
        class MockDataProcessor(BaseDataProcessor[MockConfig]):
            def process(self, ctx: RunContext, x: Bool, y: Int) -> Bool:
                ...

        processor: MockDataProcessor = MockDataProcessor()
        assert processor.signature == Signature(
            parameters=[
                Parameter(name="x", kind=_ParameterKind.POSITIONAL_OR_KEYWORD, annotation=Bool),
                Parameter(name="y", kind=_ParameterKind.POSITIONAL_OR_KEYWORD, annotation=Int),
            ],
            return_annotation=Bool,
        )

    def test_init_subclass(self) -> None:
        # valid definition
        class MockDataProcessor(BaseDataProcessor[MockConfig]):
            def process(self, ctx: RunContext, x: Bool, y: Int) -> Bool:
                ...

        # make sure the process mode default is set
        ProcessMode.from_decorated_fn(MockDataProcessor.process)

        with pytest.raises(TypeError):
            # no process method specified
            class MockDataProcessor(BaseDataProcessor[MockConfig]):
                ...

        with pytest.raises(TypeError):
            # missing context input
            class MockDataProcessor(BaseDataProcessor[MockConfig]):
                def process(self, x: Bool, y: Int) -> Bool:
                    ...

    @pytest.mark.asyncio
    async def test_run(self) -> None:
        class MockDataProcessor(BaseDataProcessor[MockConfig]):
            def process(self, ctx: RunContext, a: Bool) -> Bool:
                ...

        with patch("hyped.core.nodes.processor.ProcessMode") as mock_process_mode:
            # set up process mode mock
            mock_mode = MagicMock(spec=ProcessMode)
            mock_process_mode.from_decorated_fn.return_value = mock_mode
            # set up process mode prepare mock
            prepared_inputs = [(MagicMock(), {"x": MagicMock()}), (MagicMock(), {"x": MagicMock()})]
            mock_mode.prepare.return_value = prepared_inputs
            # set up process mode finalize mock, needs to make sure to
            # consume the outputs iterable in order to execute all calls
            # to the process function
            mock_mode.finalize.side_effect = lambda _, outputs: list(outputs)

            # create a processor and mock the process method
            processor: MockDataProcessor = MockDataProcessor()
            processor.process = MagicMock()
            # create mock context and input arrays
            mock_context = MagicMock()
            mock_arrays = {"x": MagicMock()}
            # run the processor
            await processor.run(mock_context, mock_arrays)

            # make sure the arrays where prepared by the
            mock_mode.prepare.assert_called_once_with(mock_context, **mock_arrays)
            processor.process.assert_has_calls(
                [call(ctx, **inputs) for ctx, inputs in prepared_inputs]
            )
            mock_mode.finalize.assert_called_once_with(mock_context, ANY)

            # reset mock processor mode
            mock_mode.reset_mock()

            # use async process function instead
            processor.process = AsyncMock()
            processor._is_process_async = True
            # run the processor
            await processor.run(mock_context, mock_arrays)
            # make sure the arrays where prepared by the
            mock_mode.prepare.assert_called_once_with(mock_context, **mock_arrays)
            processor.process.assert_has_calls(
                [call(ctx, **inputs) for ctx, inputs in prepared_inputs]
            )
            mock_mode.finalize.assert_called_once_with(mock_context, ANY)

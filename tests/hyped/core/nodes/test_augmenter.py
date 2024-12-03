from inspect import Parameter, Signature, _ParameterKind
from itertools import chain
from typing import AsyncIterable, Iterable
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest

from hyped.core.nodes.augmenter import BaseDataAugmenter, BaseDataAugmenterConfig
from hyped.core.nodes.base import ProcessMode, RunContext, process_mode
from hyped.core.typing import Bool, Int, TraceIndexList


class MockConfig(BaseDataAugmenterConfig):
    @classmethod
    @property
    def type_id(self) -> str:
        return str(uuid4())


class TestBaseDataProcessor:
    def test_signature(self) -> None:
        expected_signature = Signature(
            parameters=[
                Parameter(name="x", kind=_ParameterKind.POSITIONAL_OR_KEYWORD, annotation=Bool),
                Parameter(name="y", kind=_ParameterKind.POSITIONAL_OR_KEYWORD, annotation=Int),
            ],
            return_annotation=Bool,
        )

        class MockDataProcessor(BaseDataAugmenter[MockConfig]):
            def process(self, ctx: RunContext, x: Bool, y: Int) -> Iterable[Bool]:
                ...

        processor: MockDataProcessor = MockDataProcessor()
        assert processor.signature == expected_signature

        class MockDataProcessor(BaseDataAugmenter[MockConfig]):
            async def process(self, ctx: RunContext, x: Bool, y: Int) -> AsyncIterable[Bool]:
                ...

        processor: MockDataProcessor = MockDataProcessor()
        assert processor.signature == expected_signature

        class MockDataProcessor(BaseDataAugmenter[MockConfig]):
            @process_mode(batched=True, backend="python")
            async def process(
                self, ctx: RunContext, x: Bool, y: Int
            ) -> tuple[Bool, TraceIndexList]:
                ...

        processor: MockDataProcessor = MockDataProcessor()
        assert processor.signature == expected_signature

    def test_init_subclass(self) -> None:
        # valid definition
        class MockDataProcessor(BaseDataAugmenter[MockConfig]):
            def process(self, ctx: RunContext, x: Bool, y: Int) -> Bool:
                ...

        # make sure the process mode default is set
        ProcessMode.from_decorated_fn(MockDataProcessor.process)

        with pytest.raises(TypeError):
            # no process method specified
            class MockDataProcessor(BaseDataAugmenter[MockConfig]):
                ...

        with pytest.raises(TypeError):
            # missing context input
            class MockDataProcessor(BaseDataAugmenter[MockConfig]):
                def process(self, x: Bool, y: Int) -> Bool:
                    ...

        with pytest.raises(TypeError):
            # wrong signature for batched process function
            class MockDataProcessor(BaseDataAugmenter[MockConfig]):
                @process_mode(batched=True, backend="python")
                def process(self, ctx: RunContext, x: Bool, y: Int) -> Bool:
                    ...

    @pytest.mark.asyncio
    async def test_run_batched(self) -> None:
        class MockDataProcessor(BaseDataAugmenter[MockConfig]):
            def process(self, ctx: RunContext, a: Bool) -> Iterable[Bool]:
                ...

        with patch("hyped.core.nodes.augmenter.ProcessMode") as mock_process_mode:
            # set up process mode mock
            mock_mode = MagicMock(spec=ProcessMode, batched=True)
            mock_process_mode.from_decorated_fn.return_value = mock_mode

            # set up process mode prepare mock
            prepared_inputs = [(MagicMock(), {"x": MagicMock()}), (MagicMock(), {"x": MagicMock()})]
            mock_mode.prepare.return_value = prepared_inputs

            mock_output = MagicMock()
            mock_trace_index = MagicMock()
            # create mock context and input arrays
            mock_context = MagicMock()
            mock_arrays = {"x": MagicMock()}

            # create a processor and mock the process method
            processor: MockDataProcessor = MockDataProcessor()
            processor.process = MagicMock(return_value=(mock_output, mock_trace_index))
            # run the processor
            await processor.run(mock_context, mock_arrays)
            # make sure the arrays where prepared by the
            mock_mode.prepare.assert_called_once_with(mock_context, **mock_arrays)
            processor.process.assert_has_calls(
                [call(ctx, **inputs) for ctx, inputs in prepared_inputs]
            )
            mock_mode.finalize.assert_called_once_with(
                mock_context, (mock_output,) * len(prepared_inputs)
            )

    @pytest.mark.asyncio
    async def test_run_batched_async(self) -> None:
        class MockDataProcessor(BaseDataAugmenter[MockConfig]):
            def process(self, ctx: RunContext, a: Bool) -> Iterable[Bool]:
                ...

        with patch("hyped.core.nodes.augmenter.ProcessMode") as mock_process_mode:
            # set up process mode mock
            mock_mode = MagicMock(spec=ProcessMode, batched=True)
            mock_process_mode.from_decorated_fn.return_value = mock_mode

            # set up process mode prepare mock
            prepared_inputs = [(MagicMock(), {"x": MagicMock()}), (MagicMock(), {"x": MagicMock()})]
            mock_mode.prepare.return_value = prepared_inputs

            mock_output = MagicMock()
            mock_trace_index = MagicMock()
            # create mock context and input arrays
            mock_context = MagicMock()
            mock_arrays = {"x": MagicMock()}

            # create a processor and mock the async process method
            processor: MockDataProcessor = MockDataProcessor()
            processor.process = AsyncMock(return_value=(mock_output, mock_trace_index))
            processor._is_process_async = True
            # run the processor
            await processor.run(mock_context, mock_arrays)
            # make sure the arrays where prepared by the
            mock_mode.prepare.assert_called_once_with(mock_context, **mock_arrays)
            processor.process.assert_has_calls(
                [call(ctx, **inputs) for ctx, inputs in prepared_inputs]
            )
            mock_mode.finalize.assert_called_once_with(
                mock_context, (mock_output,) * len(prepared_inputs)
            )

    @pytest.mark.asyncio
    async def test_run_non_batched(self) -> None:
        class MockDataProcessor(BaseDataAugmenter[MockConfig]):
            def process(self, ctx: RunContext, a: Bool) -> Iterable[Bool]:
                ...

        with patch("hyped.core.nodes.augmenter.ProcessMode") as mock_process_mode, patch(
            "hyped.core.nodes.augmenter.chain",
            MagicMock(
                side_effect=lambda *x: list(chain(*x)),
                from_iterable=lambda x: list(chain.from_iterable(x)),
            ),
        ):
            # set up process mode mock
            mock_mode = MagicMock(spec=ProcessMode, batched=False)
            mock_process_mode.from_decorated_fn.return_value = mock_mode

            # set up process mode prepare mock
            prepared_inputs = [(MagicMock(), {"x": MagicMock()}), (MagicMock(), {"x": MagicMock()})]
            mock_mode.prepare.return_value = prepared_inputs

            mock_output = MagicMock()
            # create mock context and input arrays
            mock_context = MagicMock()
            mock_arrays = {"x": MagicMock()}

            # create a processor and mock the process method
            processor: MockDataProcessor = MockDataProcessor()
            processor.process = MagicMock(return_value=[mock_output, mock_output])
            # run the processor
            await processor.run(mock_context, mock_arrays)
            # make sure the arrays where prepared by the
            mock_mode.prepare.assert_called_once_with(mock_context, **mock_arrays)
            processor.process.assert_has_calls(
                [call(ctx, **inputs) for ctx, inputs in prepared_inputs]
            )
            mock_mode.finalize.assert_called_once_with(
                mock_context,
                [mock_output] * len(prepared_inputs) * len(processor.process.return_value),
            )

    @pytest.mark.asyncio
    async def test_run_non_batched_async(self) -> None:
        class MockDataProcessor(BaseDataAugmenter[MockConfig]):
            def process(self, ctx: RunContext, a: Bool) -> Iterable[Bool]:
                ...

        with patch("hyped.core.nodes.augmenter.ProcessMode") as mock_process_mode, patch(
            "hyped.core.nodes.augmenter.chain",
            MagicMock(
                side_effect=lambda *x: list(chain(*x)),
                from_iterable=lambda x: list(chain.from_iterable(x)),
            ),
        ):
            # set up process mode mock
            mock_mode = MagicMock(spec=ProcessMode, batched=False)
            mock_process_mode.from_decorated_fn.return_value = mock_mode

            # set up process mode prepare mock
            prepared_inputs = [(MagicMock(), {"x": MagicMock()}), (MagicMock(), {"x": MagicMock()})]
            mock_mode.prepare.return_value = prepared_inputs

            mock_output = MagicMock()
            # create mock context and input arrays
            mock_context = MagicMock()
            mock_arrays = {"x": MagicMock()}

            async def async_process(*args, **kwargs) -> AsyncIterable:
                yield mock_output
                yield mock_output

            # create a processor and mock the async process method
            processor: MockDataProcessor = MockDataProcessor()
            processor.process = MagicMock(side_effect=async_process)
            processor._is_process_async = True
            # run the processor
            await processor.run(mock_context, mock_arrays)
            # make sure the arrays where prepared by the
            mock_mode.prepare.assert_called_once_with(mock_context, **mock_arrays)
            processor.process.assert_has_calls(
                [call(ctx, **inputs) for ctx, inputs in prepared_inputs]
            )
            mock_mode.finalize.assert_called_once_with(
                mock_context, [mock_output] * len(prepared_inputs) * 2
            )

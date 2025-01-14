from inspect import Parameter, Signature
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from hyped.core.builder import DataFlowGraphBuilder
from hyped.core.features.dtypes import Int32Type, MappingType
from hyped.core.features.features import Feature
from hyped.core.features.reference import ConcreteReference
from hyped.core.flow import DataFlow
from hyped.core.nodes.base import BaseNode, BaseNodeConfig, ProcessMode, RunContext, process_mode


class TestProcessMode:
    def test_decorator_assignment(self) -> None:
        @process_mode(batched=True, backend="python")
        def fn():
            pass

        mode = ProcessMode(batched=True, backend="python")
        assert ProcessMode.from_decorated_fn(fn) == mode

        def invalid_fn():
            pass

        with pytest.raises(RuntimeError):
            ProcessMode.from_decorated_fn(invalid_fn)

    def test_set_default(self) -> None:
        # create default mode
        mode = ProcessMode(batched=False, backend="arrow")

        @mode.set_default
        def fn():
            pass

        # set default to function without specified process mode
        mode.set_default(fn)
        assert ProcessMode.from_decorated_fn(fn) == mode

        # create function object with specified process mode
        @process_mode(batched=True, backend="python")
        def decorated_fn():
            pass

        # set deafult should not overwrite the process mode
        mode.set_default(decorated_fn)
        assert ProcessMode.from_decorated_fn(decorated_fn) != mode

    def test_registries(self) -> None:
        # create mock objects
        mode = MagicMock(spec=ProcessMode)
        to_arrow_converter = MagicMock()
        from_arrow_converter = MagicMock()
        # test register to arrow converter
        ProcessMode.register_to_arrow_converter(mode)(to_arrow_converter)
        assert to_arrow_converter == ProcessMode.to_arrow_converters[mode]
        # test register from arrow converter
        ProcessMode.register_from_arrow_converter(mode)(from_arrow_converter)
        assert from_arrow_converter == ProcessMode.from_arrow_converters[mode]

    def test_prepare_batched(self) -> None:
        # create a dummy process mode
        mode = ProcessMode(batched=True, backend="dummy")
        # register a dummy from arrow converter
        from_arrow_converter = MagicMock()
        ProcessMode.register_from_arrow_converter(mode)(from_arrow_converter)
        assert from_arrow_converter == ProcessMode.from_arrow_converters[mode]

        # create a run context
        ctx = RunContext(
            session=MagicMock(),
            node_id=MagicMock(),
            index=[MagicMock(), MagicMock(), MagicMock()],
            rank=MagicMock(),
            input_dtype=MagicMock(),
            output_dtype=MagicMock(),
        )

        # create mock keyword arguments and prepare them
        inputs = {"a": MagicMock(), "b": MagicMock()}
        prepare_output = list(mode.prepare(ctx, **inputs))
        assert len(prepare_output) == 1
        prepared_ctx, prepared_inputs = prepare_output[0]

        # context should stay unchanged in batched mode
        assert prepared_ctx == ctx
        # inputs should be passed through the converter
        from_arrow_converter.assert_called_once_with(ctx.input_dtype.arrow_schema, **inputs)
        assert from_arrow_converter.return_value == prepared_inputs

    def test_prepare_non_batched(self) -> None:
        # create a dummy process mode
        mode = ProcessMode(batched=False, backend="dummy")
        # register a dummy from arrow converter
        from_arrow_converter = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
        ProcessMode.register_from_arrow_converter(mode)(from_arrow_converter)
        assert from_arrow_converter == ProcessMode.from_arrow_converters[mode]

        # create a run context
        ctx = RunContext(
            session=MagicMock(),
            node_id=MagicMock(),
            index=[MagicMock(), MagicMock(), MagicMock()],
            rank=MagicMock(),
            input_dtype=MagicMock(),
            output_dtype=MagicMock(),
        )

        # create mock keyword arguments and prepare them
        inputs = {"a": MagicMock(), "b": MagicMock()}
        prepare_output = list(mode.prepare(ctx, **inputs))
        assert len(prepare_output) == len(ctx.index) == 3
        # make sure converter was called correctly
        from_arrow_converter.assert_called_once_with(ctx.input_dtype.arrow_schema, **inputs)

        for i, (prepared_ctx, prepared_inputs) in enumerate(prepare_output):
            # check prepared context
            assert prepared_ctx.index == ctx.index[i]
            assert prepared_ctx.node_id == ctx.node_id
            assert prepared_ctx.rank == ctx.rank
            assert prepared_ctx.input_dtype == ctx.input_dtype
            assert prepared_ctx.output_dtype == ctx.output_dtype
            # check prepared input sample
            assert prepared_inputs == from_arrow_converter.return_value[i]

    def test_finalize(self) -> None:
        # create a dummy process mode
        mode = ProcessMode(batched=False, backend="dummy")
        # register a dummy from arrow converter
        to_arrow_converter = MagicMock()
        ProcessMode.register_to_arrow_converter(mode)(to_arrow_converter)
        assert to_arrow_converter == ProcessMode.to_arrow_converters[mode]

        # create mock context and outputs
        ctx = MagicMock()
        outputs = MagicMock()

        # finalize mock outputs
        finalized_outputs = mode.finalize(ctx, outputs)
        # check finalized outputs
        to_arrow_converter.assert_called_once_with(ctx.output_dtype.arrow_type, outputs)
        assert finalized_outputs == to_arrow_converter.return_value

    @pytest.mark.parametrize(
        "mode",
        [
            ProcessMode(batched=True, backend="arrow"),
            ProcessMode(batched=True, backend="python"),
            ProcessMode(batched=False, backend="python"),
        ],
    )
    def test_registered_converters(self, mode: ProcessMode) -> None:
        # create a runcontext
        ctx = RunContext(
            session=MagicMock(),
            node_id=0,
            index=[0, 1, 2, 3, 4],
            rank=0,
            input_dtype=MappingType.construct({"field": Int32Type}),
            output_dtype=Int32Type,
        )
        # create an array
        arr = pa.array([0, 1, 2, 3, 4], type=Int32Type.arrow_type)
        # prepare and finalize the array
        inputs = [x["field"] for _, x in mode.prepare(ctx, field=arr)]
        output = mode.finalize(ctx, inputs)
        # composing prepare and finalize should be the identity function
        assert output.to_pylist() == arr.to_pylist()


class MockConfig(BaseNodeConfig):
    pass


class MockNode(BaseNode[MockConfig]):
    @property
    def signature(self):
        return Signature(
            parameters=[
                Parameter(name="x", kind=Parameter.POSITIONAL_OR_KEYWORD),
                Parameter(name="y", kind=Parameter.POSITIONAL_OR_KEYWORD),
            ],
        )


class TestBaseNode:
    @patch("hyped.core.nodes.base.build_feature_from_reference", MagicMock())
    def test_call(self) -> None:
        # create mock flow and builder
        builder = MagicMock(spec=DataFlowGraphBuilder)
        flow = MagicMock(spec=DataFlow, _builder=builder)
        # create mock inputs
        x = MagicMock(spec=Feature, ref=MagicMock(spec=ConcreteReference, _builder=flow._builder))
        y = MagicMock(spec=Feature, ref=MagicMock(spec=ConcreteReference, _builder=flow._builder))
        # call the node with only positional arguments
        node = MockNode()
        node.call(flow, x, y)
        # make sure the compute node was added correctly
        builder.compute.assert_called_once_with(node, {"x": x.ref, "y": y.ref})

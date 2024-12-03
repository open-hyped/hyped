from unittest.mock import ANY, MagicMock, patch

import pyarrow as pa
import pytest

from hyped.core.features.features import Feature
from hyped.core.features.reference import Reference
from hyped.core.features.types import Int32Type, MappingType
from hyped.core.flow import DataFlow
from hyped.core.graph import DataFlowGraph
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
            node_id=MagicMock(),
            index=[MagicMock(), MagicMock(), MagicMock()],
            rank=MagicMock(),
            input_type=MagicMock(),
            output_type=MagicMock(),
        )

        # create mock keyword arguments and prepare them
        inputs = {"a": MagicMock(), "b": MagicMock()}
        prepare_output = list(mode.prepare(ctx, **inputs))
        assert len(prepare_output) == 1
        prepared_ctx, prepared_inputs = prepare_output[0]

        # context should stay unchanged in batched mode
        assert prepared_ctx == ctx
        # inputs should be passed through the converter
        from_arrow_converter.assert_called_once_with(ctx.input_type.arrow_schema, **inputs)
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
            node_id=MagicMock(),
            index=[MagicMock(), MagicMock(), MagicMock()],
            rank=MagicMock(),
            input_type=MagicMock(),
            output_type=MagicMock(),
        )

        # create mock keyword arguments and prepare them
        inputs = {"a": MagicMock(), "b": MagicMock()}
        prepare_output = list(mode.prepare(ctx, **inputs))
        assert len(prepare_output) == len(ctx.index) == 3
        # make sure converter was called correctly
        from_arrow_converter.assert_called_once_with(ctx.input_type.arrow_schema, **inputs)

        for i, (prepared_ctx, prepared_inputs) in enumerate(prepare_output):
            # check prepared context
            assert prepared_ctx.index == ctx.index[i]
            assert prepared_ctx.node_id == ctx.node_id
            assert prepared_ctx.rank == ctx.rank
            assert prepared_ctx.input_type == ctx.input_type
            assert prepared_ctx.output_type == ctx.output_type
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
        to_arrow_converter.assert_called_once_with(ctx.output_type.arrow_type, outputs)
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
            node_id=0,
            index=[0, 1, 2, 3, 4],
            rank=0,
            input_type=MappingType.construct({"field": Int32Type}),
            output_type=Int32Type,
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
        return None


class TestBaseNode:
    @patch("hyped.core.nodes.base.FeatureEngine")
    def test_call(self, mock_feature_engine: MagicMock) -> None:
        obj = MagicMock()
        obj_dtype = MagicMock()
        mock_feature_engine.return_value.get_references_and_objects.return_value = (
            MagicMock(),
            {"obj": obj},
            {"obj": obj_dtype},
        )

        # create mock graph and inputs
        flow = MagicMock(spec=DataFlow, _graph=MagicMock(spec=DataFlowGraph))
        x = MagicMock(spec=Feature, ref=MagicMock(spec=Reference, _graph=flow._graph))
        y = MagicMock(spec=Feature, ref=MagicMock(spec=Reference, _graph=flow._graph))

        # call the node with only positional arguments
        node = MockNode()
        node.call(flow, x, y)
        # check calls to feature engine
        mock_feature_engine.return_value.validate_signature.assert_called_once()
        mock_feature_engine.return_value.validate_arguments.assert_called_once_with(x, y)
        # make sure objects were collected as expected
        flow._graph.add_collect_node_with_constants.assert_called_with(obj, obj_dtype)
        # make sure the node was added to the graph
        flow._graph.add_compute_node.assert_called_once_with(node, ANY)

        # reset mocks
        mock_feature_engine.reset_mock()
        flow.reset_mock()
        # call the node and provide flow as keyword argument
        node = MockNode()
        node.call(x, y=y, flow=flow)
        # check calls to feature engine
        mock_feature_engine.return_value.validate_signature.assert_called_once()
        mock_feature_engine.return_value.validate_arguments.assert_called_once_with(x, y=y)
        # make sure objects were collected as expected
        flow._graph.add_collect_node_with_constants.assert_called_with(obj, obj_dtype)
        # make sure the node was added to the graph
        flow._graph.add_compute_node.assert_called_once_with(node, ANY)

        # reset mocks
        mock_feature_engine.reset_mock()
        flow.reset_mock()
        # call the node and provide flow by reference
        node = MockNode()
        node.call(x, y)
        # check calls to feature engine
        mock_feature_engine.return_value.validate_signature.assert_called_once()
        mock_feature_engine.return_value.validate_arguments.assert_called_once_with(x, y)
        # make sure objects were collected as expected
        flow._graph.add_collect_node_with_constants.assert_called_with(obj, obj_dtype)
        # make sure the node was added to the graph
        flow._graph.add_compute_node.assert_called_once_with(node, ANY)

        with pytest.raises(RuntimeError):
            # cannot infer data flow graph
            MockNode().call()

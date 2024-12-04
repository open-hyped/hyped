from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from hyped.core.features.types import Int32Type, Int64Type, MappingType, SequenceType, Type
from hyped.core.nodes.base import RunContext
from hyped.core.nodes.collect import CollectNode
from hyped.core.utils import NestedType


class TestCollectNode:
    @pytest.mark.parametrize(
        "collect, input_types, input_objects, expected_dtype, expected_object, expected_required_casts",
        [
            # collect simple feature
            ("x", {"x": Int32Type}, {"x": 0}, Int32Type, 0, {}),
            # collect sequence type
            (
                ["x", "y", "z"],
                {"x": Int32Type, "y": Int32Type, "z": Int32Type},
                {"x": 0, "y": 1, "z": 2},
                SequenceType(Int32Type, length=3),
                [0, 1, 2],
                {},
            ),
            (
                ["x", "y", "z"],
                {"x": Int32Type, "y": Int64Type, "z": Int32Type},
                {"x": 0, "y": 1, "z": 2},
                SequenceType(Int64Type, length=3),
                [0, 1, 2],
                {"x": Int64Type, "z": Int64Type},
            ),
            # collect mapping type
            (
                {"a": "x", "b": "y"},
                {"x": Int32Type, "y": Int32Type},
                {"x": 0, "y": 1},
                MappingType.construct({"a": Int32Type, "b": Int32Type}),
                {"a": 0, "b": 1},
                {},
            ),
            # nested mapping and sequence types
            (
                {"a": ["x", "x"], "b": "y"},
                {"x": Int32Type, "y": Int32Type},
                {"x": 0, "y": 1},
                MappingType.construct({"a": SequenceType(Int32Type, length=2), "b": Int32Type}),
                {"a": [0, 0], "b": 1},
                {},
            ),
            # nested sequence of mappings with casting required
            (
                [{"b": "x"}, {"b": "y"}],
                {"x": Int32Type, "y": Int64Type},
                {"x": 0, "y": 1},
                SequenceType(MappingType.construct({"b": Int64Type}), 2),
                [{"b": 0}, {"b": 1}],
                {"x": Int64Type},
            ),
            # nested sequence of sequences with casting required
            (
                [["x", "x"], ["y", "y"]],
                {"x": Int32Type, "y": Int64Type},
                {"x": 0, "y": 1},
                SequenceType(SequenceType(Int64Type, 2), 2),
                [[0, 0], [1, 1]],
                {"x": Int64Type},
            ),
        ],
    )
    def test_collect_node(
        self,
        collect: NestedType[str],
        input_types: dict[str, Type],
        input_objects: dict[str, Any],
        expected_dtype: Type,
        expected_object: pa.Array,
        expected_required_casts: dict[str, Type],
    ) -> None:
        # create a mock graph
        graph = MagicMock()
        graph.get_dtype_from_reference.side_effect = lambda x: x
        # create the collect node instance
        node = CollectNode(lookup=collect)
        # build the output type and check it
        dtype, required_casts = node.build_output_type(graph, input_types)
        assert dtype == expected_dtype
        assert required_casts == expected_required_casts

        # build a run context with the expected input and output types
        ctx = RunContext(
            node_id=0,
            index=[0],
            rank=0,
            input_type=MappingType.construct(input_types | required_casts),
            output_type=dtype,
        )

        # convert the input objects to pyarrow arrays
        input_arrays = {
            k: pa.array([v], type=ctx.input_type[k].arrow_type) for k, v in input_objects.items()
        }

        # apply the collect operation on the arrays
        object = node.collect(ctx, input_arrays)

        # check the collected objects matches the expectation
        assert object.to_pylist()[0] == expected_object

    def test_empty_sequences_not_supported(self) -> None:
        node = CollectNode(lookup=[])
        with pytest.raises(RuntimeError):
            dtype = node.build_output_type(MagicMock(), {})

from unittest.mock import ANY, MagicMock, patch

from hyped.core.features.features import Feature
from hyped.core.graph import DataFlowGraph
from hyped.core.typing import Int, cast


def test_cast() -> None:
    # no-op for non-feature inputs
    assert cast(int, 8) == 8

    # create a mock graph with the required functions
    graph = MagicMock(
        spec=DataFlowGraph, add_cast_node=MagicMock(), get_feature_from_reference=MagicMock()
    )
    # create a mock feature to cast
    feature = MagicMock(spec=Feature, ref=MagicMock(_graph=graph))

    # patch the function that retrieves the concrete data type to cast to
    with patch("hyped.core.typing.build_feature_from_annotation"):
        # cast the feature to an integer and check the output and internals
        assert cast(Int, feature) == graph.get_feature_from_reference.return_value
        graph.add_cast_node.assert_called_once_with(feature.ref, ANY)

from unittest.mock import ANY, MagicMock, patch

from hyped.core.builder import DataFlowGraphBuilder
from hyped.core.features.features import Feature
from hyped.core.features.reference import ConcreteReference
from hyped.core.ops.cast import cast
from hyped.core.typing import Int


def test_cast() -> None:
    # no-op for non-feature inputs
    assert cast(int, 8) == 8

    # create a mock graph with the required functions
    builder = MagicMock(spec=DataFlowGraphBuilder, cast_node=MagicMock())
    # create a mock feature to cast
    feature = MagicMock(spec=Feature, ref=MagicMock(spec=ConcreteReference, _builder=builder))

    # patch the function that retrieves the concrete data type to cast to
    with (
        patch("hyped.core.ops.cast.build_feature_from_annotation"),
        patch("hyped.core.ops.cast.build_feature_from_reference") as mock_build_feature_from_ref,
    ):
        # cast the feature to an integer and check the output and internals
        assert cast(Int, feature) == mock_build_feature_from_ref.return_value
        builder.cast_node.assert_called_once_with(feature.ref, ANY)

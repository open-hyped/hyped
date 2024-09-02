from unittest.mock import patch

import pytest
from pydantic import BaseModel

from hyped.core.refs.ref import FeatureRef


@pytest.fixture(autouse=True)
def patch_feature_ref_eq_op():
    # patches the feature ref type to use the standard comparator
    # operators instead of the overwritten ones
    # this is required for the mock.assert_called_with checks which
    # make use of the == opeartor to compare the objects

    class PatchedFeatureRef(FeatureRef):
        __eq__ = BaseModel.__eq__
        __ne__ = BaseModel.__ne__

    with (
        patch("hyped.core.refs.ref.FeatureRef", PatchedFeatureRef),
        patch("hyped.core.graph.FeatureRef", PatchedFeatureRef),
        patch("hyped.core.flow.FeatureRef", PatchedFeatureRef),
    ):
        yield

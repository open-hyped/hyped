from unittest.mock import MagicMock, call

import datasets

from hyped.core.flow import DataFlow
from hyped.nodes.decorators import as_processor


def test_as_processor():
    # create a mock object to track the function calls
    mock = MagicMock()

    @as_processor
    def mock_func(x: int) -> int:
        mock(x)
        return x * 2

    #
    features = datasets.Features({"x": datasets.Value("int32")})
    ds = datasets.Dataset.from_dict({"x": range(10)}, features=features)

    flow = DataFlow(ds.features)
    out = mock_func(x=flow.src_features.x)
    out_ds, _ = flow.apply(ds, collect={"out": out})

    mock.assert_has_calls(calls=[call(i) for i in range(10)], any_order=True)

    assert all(2 * i == j for i, j in zip(ds["x"], out_ds["out"]))

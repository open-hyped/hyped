import datasets
import pytest

import hyped


@pytest.fixture
def ds() -> datasets.Dataset:
    data = {"x": [0, 1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5, 6]}
    features = {"x": datasets.Value("int32"), "y": datasets.Value("int32")}
    return datasets.Dataset.from_dict(data, features=datasets.Features(features))


@pytest.mark.integration_test
def test_simple_aggregation(ds: datasets.Dataset) -> None:
    flow = hyped.DataFlow(ds.features)
    _, aggregates = flow.apply(ds, collect=flow.source, aggregate={"sum": flow.source["x"].sum()})
    # check output
    assert aggregates["sum"] == 15


@pytest.mark.integration_test
def test_complex_aggregation(ds: datasets.Dataset) -> None:
    flow = hyped.DataFlow(ds.features)
    # compute the complex aggregated value
    out = (flow.source["x"].sum() + flow.source["y"].mean()) * 4
    _, aggregates = flow.apply(ds, collect=flow.source, aggregate={"out": out})
    # check output
    assert aggregates["out"] == 18.5 * 4

import datasets
import pytest

import hyped


@pytest.fixture
def ds() -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {"x": [[0, 1, 2, 3, 4, 5]], "y": [[1, 2, 3, 4, 5, 6]], "v": [True]},
        features=datasets.Features(
            {
                "x": datasets.Sequence(datasets.Value("int32"), 6),
                "y": datasets.Sequence(datasets.Value("int32"), 6),
                "v": datasets.Value("bool"),
            }
        ),
    )


@pytest.mark.integration_test
def test_foreach(ds: datasets.Dataset) -> None:
    # create the data flow
    flow = hyped.DataFlow(ds.features)
    out = flow.source["x"].foreach(lambda x: x * 2)
    # apply the flow
    out_ds = flow.apply(ds, collect={"out": out})
    # check output data
    assert out_ds.to_dict()["out"] == [[0, 2, 4, 6, 8, 10]]


@pytest.mark.integration_test
def test_complex_foreach_with_zip(ds: datasets.Dataset) -> None:
    # create the data flow
    flow = hyped.DataFlow(ds.features)
    # zip sequences and apply foreach on zipped elements
    zipped = hyped.ops.zip_(flow.source["x"], flow.source["y"])
    out = zipped.foreach(lambda xy: xy[0] * xy[1] * 2)
    # apply the flow
    out_ds = flow.apply(ds, collect={"out": out})
    # check output data
    assert out_ds.to_dict()["out"] == [[0, 4, 12, 24, 40, 60]]


@pytest.mark.integration_test
def test_foreach_with_augmenter(ds: datasets.Dataset) -> None:
    # create the data flow
    flow = hyped.DataFlow(ds.features)
    outA = flow.source["x"].foreach(lambda x: x * 2)
    outB = flow.source["x"].filter(flow.source["v"])
    # apply the flow
    out_ds = flow.apply(ds, collect={"outA": outA, "outB": outB})
    # check output data
    assert out_ds.to_dict()["outA"] == [[0, 2, 4, 6, 8, 10]]
    assert out_ds.to_dict()["outB"] == [[0, 1, 2, 3, 4, 5]]


@pytest.mark.integration_test
def test_foreach_on_empty_sequence() -> None:
    # create dataset containing empty sequences
    ds = datasets.Dataset.from_dict(
        {"x": [[0, 1], [], [2, 3]]},
        features=datasets.Features(
            {
                "x": datasets.Sequence(datasets.Value("int32")),
            }
        ),
    )
    # create the data flow
    flow = hyped.DataFlow(ds.features)
    outA = flow.source["x"].foreach(lambda x: x * 2)
    # apply the flow
    out_ds = flow.apply(ds, collect={"out": outA})
    # check output data
    assert out_ds.to_dict()["out"] == [[0, 2], [], [4, 6]]

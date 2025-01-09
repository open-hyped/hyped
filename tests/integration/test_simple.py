import datasets
import pytest

from hyped import DataFlow


@pytest.mark.integration_test
def test_simple_math_ops() -> None:
    ds = datasets.Dataset.from_dict(
        {"x": [1, 2, 3, 4, 5, 6, 7, 8, 9], "y": [9, 8, 7, 6, 5, 4, 3, 2, 1]}
    )

    flow = DataFlow(ds.features)
    ds = flow.apply(
        ds,
        collect={
            "add": flow.source["x"] + flow.source["y"],
            "sub": flow.source["x"] - flow.source["y"],
            "mul": flow.source["x"] * flow.source["y"],
            "truediv": flow.source["x"] / flow.source["y"],
            "floordiv": flow.source["x"] // flow.source["y"],
        },
    )

    assert ds["add"] == [1 + 9, 2 + 8, 3 + 7, 4 + 6, 5 + 5, 6 + 4, 7 + 3, 8 + 2, 9 + 1]
    assert ds["sub"] == [1 - 9, 2 - 8, 3 - 7, 4 - 6, 5 - 5, 6 - 4, 7 - 3, 8 - 2, 9 - 1]
    assert ds["mul"] == [1 * 9, 2 * 8, 3 * 7, 4 * 6, 5 * 5, 6 * 4, 7 * 3, 8 * 2, 9 * 1]
    assert ds["truediv"] == [1 / 9, 2 / 8, 3 / 7, 4 / 6, 5 / 5, 6 / 4, 7 / 3, 8 / 2, 9 / 1]
    assert ds["floordiv"] == [
        1 // 9,
        2 // 8,
        3 // 7,
        4 // 6,
        5 // 5,
        6 // 4,
        7 // 3,
        8 // 2,
        9 // 1,
    ]


@pytest.mark.integration_test
def test_simple_sequence_ops() -> None:
    ds = datasets.Dataset.from_dict(
        {
            "x": [
                [1, 2, 3, 4],
                [3, 4, 5, 6],
                [5, 6, 7, 8],
            ],
        }
    )

    flow = DataFlow(ds.features)
    ds = flow.apply(
        ds,
        collect={
            "len": flow.source["x"].length(),
            "min": flow.source["x"].min(),
            "max": flow.source["x"].max(),
            "sum": flow.source["x"].sum(),
        },
    )

    assert ds["len"] == [4, 4, 4]
    assert ds["min"] == [1, 3, 5]
    assert ds["max"] == [4, 6, 8]
    assert ds["sum"] == [10, 18, 26]

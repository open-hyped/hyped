import pyarrow as pa
import pytest

from hyped.core.nodes.const import ConstNode


@pytest.mark.parametrize(
    "value",
    [
        pa.array([True], type=pa.bool_()),
        pa.array([16], type=pa.int32()),
        pa.array([{"field": 16}], type=pa.struct((("field", pa.int32()),))),
    ],
)
def test_const_node_serialization(value: pa.Array) -> None:
    node = ConstNode(value=value)

    config = ConstNode.Config.model_validate(node.config.model_dump())
    reconstructed_node = ConstNode(config=config)

    assert node.config.value.to_pylist() == reconstructed_node.config.value.to_pylist()

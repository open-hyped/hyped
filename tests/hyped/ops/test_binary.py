from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Value

from hyped import ops
from hyped.core.flow import DataFlow
from hyped.core.refs.ref import FeatureRef
from hyped.ops.utils import _handle_constant_inputs_for_binary_op


def test_binary_op_constant_inputs_handler():
    mock_flow = MagicMock()
    mock_binary_op = MagicMock()
    # wrap mock binary operator
    wrapped_binary_op = _handle_constant_inputs_for_binary_op(mock_binary_op)
    # create a feature reference instance
    ref = FeatureRef(node_id_="", key_=tuple(), flow_=mock_flow, feature_=Value("int32"))

    # expected error on only constant inputs
    with pytest.raises(RuntimeError):
        wrapped_binary_op(0, 0)

    # called with only references
    wrapped_binary_op(ref, ref)
    mock_binary_op.assert_called_with(ref, ref)

    with patch("hyped.ops.utils.Const") as mock_const:
        # first constant then reference
        wrapped_binary_op(0, ref)
        mock_const.assert_called_with(value=0)
        mock_const(value=0).call.assert_called_with(mock_flow)
        mock_binary_op.assert_called_with(mock_const(value=0).call(mock_flow).value, ref)
        # first reference then constant
        wrapped_binary_op(ref, 1)
        mock_const.assert_called_with(value=1)
        mock_const(value=1).call.assert_called_with(mock_flow)
        mock_binary_op.assert_called_with(ref, mock_const(value=1).call(mock_flow).value)


@pytest.mark.parametrize(
    "op, proc_type, dtype",
    [
        (
            ops.add,
            "hyped.ops.binary.ops.Add",
            "int32",
        ),
        (
            ops.sub,
            "hyped.ops.binary.ops.Sub",
            "int32",
        ),
        (
            ops.mul,
            "hyped.ops.binary.ops.Mul",
            "int32",
        ),
        (
            ops.pow_,
            "hyped.ops.binary.ops.Pow",
            "int32",
        ),
        (
            ops.mod,
            "hyped.ops.binary.ops.Mod",
            "int32",
        ),
        (
            ops.truediv,
            "hyped.ops.binary.ops.TrueDiv",
            "int32",
        ),
        (
            ops.floordiv,
            "hyped.ops.binary.ops.FloorDiv",
            "int32",
        ),
        (
            ops.eq,
            "hyped.ops.binary.ops.Equals",
            "int32",
        ),
        (
            ops.ne,
            "hyped.ops.binary.ops.NotEquals",
            "int32",
        ),
        (
            ops.lt,
            "hyped.ops.binary.ops.LessThan",
            "int32",
        ),
        (
            ops.le,
            "hyped.ops.binary.ops.LessThanOrEqual",
            "int32",
        ),
        (
            ops.gt,
            "hyped.ops.binary.ops.GreaterThan",
            "int32",
        ),
        (
            ops.ge,
            "hyped.ops.binary.ops.GreaterThanOrEqual",
            "int32",
        ),
        (
            ops.and_,
            "hyped.ops.binary.ops.LogicalAnd",
            "bool",
        ),
        (
            ops.or_,
            "hyped.ops.binary.ops.LogicalOr",
            "bool",
        ),
        (
            ops.xor_,
            "hyped.ops.binary.ops.LogicalXOr",
            "bool",
        ),
    ],
)
def test_binary_op(op, proc_type, dtype):
    flow = DataFlow(
        Features(
            {
                "a": Value(dtype),
                "b": Value(dtype),
            }
        )
    )

    with patch(proc_type) as mock:
        # run operator
        op(flow.src_features.a, flow.src_features.b)
        # make sure the operator was called correctly
        mock().call.assert_called_once_with(a=flow.src_features.a, b=flow.src_features.b)

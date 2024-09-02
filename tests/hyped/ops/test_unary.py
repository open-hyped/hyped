from unittest.mock import patch

import pytest
from datasets import Features, Sequence, Value

from hyped import ops
from hyped.core.flow import DataFlow


@pytest.mark.parametrize(
    "op, agg_type",
    [
        (ops.sum_, "hyped.ops.unary.agg_ops.SumAggregator"),
        (ops.mean, "hyped.ops.unary.agg_ops.MeanAggregator"),
    ],
)
def test_simple_aggregators(op, agg_type):
    flow = DataFlow(Features({"a": Value("int32")}))

    with patch(agg_type) as mock:
        # run operator
        op(flow.src_features.a)
        # make sure the processor was called with the correct inputs
        mock().call.assert_called_once_with(x=flow.src_features.a)


@pytest.mark.parametrize(
    "op, agg_type",
    [
        (ops.sum_, "hyped.ops.unary.proc_ops.SequenceSum"),
        (ops.mean, "hyped.ops.unary.proc_ops.SequenceMean"),
    ],
)
def test_sequence_aggregators(op, agg_type):
    flow = DataFlow(Features({"a": Sequence(Value("int32"))}))

    with patch(agg_type) as mock:
        # run operator
        op(flow.src_features.a)
        # make sure the processor was called with the correct inputs
        mock().call.assert_called_once_with(a=flow.src_features.a)


@pytest.mark.parametrize(
    "op, proc_type, dtype",
    [
        (ops.neg, "hyped.ops.unary.proc_ops.Neg", "int32"),
        (ops.abs_, "hyped.ops.unary.proc_ops.Abs", "int32"),
        (ops.invert, "hyped.ops.unary.proc_ops.Invert", "int32"),
    ],
)
def test_unary_op(op, proc_type, dtype):
    flow = DataFlow(
        Features(
            {
                "a": Value(dtype),
            }
        )
    )

    with patch(proc_type) as mock:
        # run operator
        op(flow.src_features.a)
        # make sure the operator was called correctly
        mock().call.assert_called_once_with(
            a=flow.src_features.a,
        )

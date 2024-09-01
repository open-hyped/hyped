from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Sequence, Value
from pydantic import BaseModel

from hyped import ops
from hyped.core.flow import DataFlow
from hyped.core.graph import DataFlowGraph
from hyped.core.refs.ref import FeatureRef
from hyped.processors._ops.collect import CollectFeatures


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


def test_binary_op_constant_inputs_handler():
    mock_flow = MagicMock()
    mock_binary_op = MagicMock()
    # wrap mock binary operator
    wrapped_binary_op = ops._handle_constant_inputs_for_binary_op(
        mock_binary_op
    )
    # create a feature reference instance
    ref = FeatureRef(
        node_id_="", key_=tuple(), flow_=mock_flow, feature_=Value("int32")
    )

    # expected error on only constant inputs
    with pytest.raises(RuntimeError):
        wrapped_binary_op(0, 0)

    # called with only references
    wrapped_binary_op(ref, ref)
    mock_binary_op.assert_called_with(ref, ref)

    with patch("hyped.ops.Const") as mock_const:
        # first constant then reference
        wrapped_binary_op(0, ref)
        mock_const.assert_called_with(value=0)
        mock_const(value=0).call.assert_called_with(mock_flow)
        mock_binary_op.assert_called_with(
            mock_const(value=0).call(mock_flow).value, ref
        )
        # first reference then constant
        wrapped_binary_op(ref, 1)
        mock_const.assert_called_with(value=1)
        mock_const(value=1).call.assert_called_with(mock_flow)
        mock_binary_op.assert_called_with(
            ref, mock_const(value=1).call(mock_flow).value
        )


def test_collect():
    flow = DataFlow(
        Features(
            {
                "x": Value("string"),
                "y": Value("string"),
            }
        )
    )

    out = ops.collect([flow.src_features.x, flow.src_features.y])
    # make sure the output feature is correct
    assert out.feature_ == Sequence(Value("string"), length=2)
    # make sure the node has been added
    assert out.node_id_ in flow._graph
    assert isinstance(
        flow._graph.nodes[out.node_id_][DataFlowGraph.NodeAttribute.NODE_OBJ],
        CollectFeatures,
    )
    # check the connections
    assert flow._graph.has_edge(flow.src_features.x.node_id_, out.node_id_)
    assert flow._graph.has_edge(flow.src_features.y.node_id_, out.node_id_)


@pytest.mark.parametrize(
    "op, agg_type",
    [
        (ops.sum_, "hyped.ops.SumAggregator"),
        (ops.mean, "hyped.ops.MeanAggregator"),
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
        (ops.sum_, "hyped.ops.SequenceSum"),
        (ops.mean, "hyped.ops.SequenceMean"),
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
        (
            ops.add,
            "hyped.ops.binary.Add",
            "int32",
        ),
        (
            ops.sub,
            "hyped.ops.binary.Sub",
            "int32",
        ),
        (
            ops.mul,
            "hyped.ops.binary.Mul",
            "int32",
        ),
        (
            ops.pow,
            "hyped.ops.binary.Pow",
            "int32",
        ),
        (
            ops.mod,
            "hyped.ops.binary.Mod",
            "int32",
        ),
        (
            ops.truediv,
            "hyped.ops.binary.TrueDiv",
            "int32",
        ),
        (
            ops.floordiv,
            "hyped.ops.binary.FloorDiv",
            "int32",
        ),
        (
            ops.eq,
            "hyped.ops.binary.Equals",
            "int32",
        ),
        (
            ops.ne,
            "hyped.ops.binary.NotEquals",
            "int32",
        ),
        (
            ops.lt,
            "hyped.ops.binary.LessThan",
            "int32",
        ),
        (
            ops.le,
            "hyped.ops.binary.LessThanOrEqual",
            "int32",
        ),
        (
            ops.gt,
            "hyped.ops.binary.GreaterThan",
            "int32",
        ),
        (
            ops.ge,
            "hyped.ops.binary.GreaterThanOrEqual",
            "int32",
        ),
        (
            ops.and_,
            "hyped.ops.binary.LogicalAnd",
            "bool",
        ),
        (
            ops.or_,
            "hyped.ops.binary.LogicalOr",
            "bool",
        ),
        (
            ops.xor_,
            "hyped.ops.binary.LogicalXOr",
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
        mock().call.assert_called_once_with(
            a=flow.src_features.a, b=flow.src_features.b
        )


@pytest.mark.parametrize(
    "op, proc_type, dtype",
    [
        (ops.neg, "hyped.ops.unary.Neg", "int32"),
        (ops.abs_, "hyped.ops.unary.Abs", "int32"),
        (ops.invert, "hyped.ops.unary.Invert", "int32"),
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


def test_len_op():
    flow = DataFlow(
        Features(
            {
                "constant_seq": Sequence(Value("int32"), length=5),
                "dynamic_seq": Sequence(Value("int32")),
                "str": Value("string"),
                "inv": Value("int32"),
            }
        )
    )

    with patch("hyped.ops.SequenceLength") as mock:
        # test constant length sequence
        out = flow.src_features.constant_seq.length_()
        # make sure the processor was not called and check the output
        assert not mock().call.called
        assert out == 5

    with patch("hyped.ops.SequenceLength") as mock:
        # test dynamic length sequence
        out = flow.src_features.dynamic_seq.length_()
        # make sure processor was called correctly
        mock().call.assert_called_once_with(a=flow.src_features.dynamic_seq)

    with pytest.raises(NotImplementedError):
        # TODO: string features are not supported yet
        flow.src_features.str.length_()

    with pytest.raises(TypeError):
        # test with invalid feature
        flow.src_features.inv.length_()


@pytest.mark.parametrize(
    "op, proc_type",
    [(ops.chain, "hyped.ops.SequenceChain")],
)
def test_chain_op(op, proc_type):
    flow = DataFlow(
        Features(
            {
                "seqA": Sequence(Value("int32")),
                "seqB": Sequence(Value("int32")),
                "seqC": Sequence(Value("int32")),
                "strA": Value("string"),
                "strB": Value("string"),
                "inv": Value("int32"),
            }
        )
    )

    with (
        patch(proc_type) as proc_mock,
        patch("hyped.ops.collect") as collect_mock,
    ):
        op(
            flow.src_features.seqA,
            flow.src_features.seqB,
            flow.src_features.seqC,
        )
        # make sure processor was called correctly
        collect_mock.assert_called_once_with(
            {
                "0": flow.src_features.seqA,
                "1": flow.src_features.seqB,
                "2": flow.src_features.seqC,
            }
        )

        # and the processor is called on the collected features
        proc_mock().call.assert_called_once_with(sequences=collect_mock())


def test_sequence_access():
    flow = DataFlow(
        Features(
            {
                "seq": Sequence(Value("int32")),
                "idx": Value("int32"),
                "val": Value("int32"),
            }
        )
    )

    with patch("hyped.ops.SequenceGetItem") as mock:
        ops.get_item(flow.src_features.seq, flow.src_features.idx)
        # make sure processor was called correctly
        mock().call.assert_called_once_with(
            sequence=flow.src_features.seq, index=flow.src_features.idx
        )

    with patch("hyped.ops.SequenceSetItem") as mock:
        ops.set_item(
            flow.src_features.seq, flow.src_features.idx, flow.src_features.val
        )
        # make sure processor was called correctly
        mock().call.assert_called_once_with(
            sequence=flow.src_features.seq,
            index=flow.src_features.idx,
            value=flow.src_features.val,
        )


def test_compress():
    # Set up the data flow with features
    flow = DataFlow(
        Features(
            {
                "values": Sequence(Value("int32")),
                "mask": Sequence(Value("bool")),
            }
        )
    )

    # Mock the BooleanIndexing processor
    with patch("hyped.ops.BooleanIndexing") as mock:
        # Call the compress function
        ops.compress(flow.src_features.values, flow.src_features.mask)

        # Verify that the processor was called correctly
        mock().call.assert_called_once_with(
            values=flow.src_features.values, mask=flow.src_features.mask
        )


@pytest.mark.parametrize(
    "op, seq_proc_type",
    [
        (
            FeatureRef.contains_,
            "hyped.ops.SequenceContains",
        ),
        (
            ops.count_of,
            "hyped.ops.SequenceCountOf",
        ),
        (
            ops.index_of,
            "hyped.ops.SequenceIndexOf",
        ),
    ],
)
def test_value_lookup_op(op, seq_proc_type):
    flow = DataFlow(
        Features(
            {
                "seq": Sequence(Value("string")),
                "str": Value("string"),
                "val": Value("string"),
                "inv": Value("int32"),
            }
        )
    )

    with patch(seq_proc_type) as mock:
        # run operator
        op(flow.src_features.seq, flow.src_features.val)
        # make sure the operator was called correctly
        mock().call.assert_called_once_with(
            sequence=flow.src_features.seq,
            value=flow.src_features.val,
        )

    with pytest.raises(NotImplementedError):
        # TODO: string features are not supported yet
        op(flow.src_features.str, flow.src_features.val)

    with pytest.raises(TypeError):
        # test with invalid feature
        op(flow.src_features.inv, flow.src_features.val)


@pytest.mark.parametrize(
    "op, proc_type",
    [(ops.zip_, "hyped.ops.SequenceZip")],
)
def test_multi_sequence_op(op, proc_type):
    flow = DataFlow(
        Features(
            {
                "a": Sequence(Value("string")),
                "b": Sequence(Value("string")),
                "c": Sequence(Value("string")),
            }
        )
    )

    with (
        patch(proc_type) as proc_mock,
        patch("hyped.ops.collect") as collect_mock,
    ):
        # call the operator
        op(
            flow.src_features.a,
            flow.src_features.b,
            flow.src_features.c,
        )
        # make sure the features are collected before
        collect_mock.assert_called_once_with(
            {
                "0": flow.src_features.a,
                "1": flow.src_features.b,
                "2": flow.src_features.c,
            }
        )
        # and the processor is called on the collected features
        proc_mock().call.assert_called_once_with(sequences=collect_mock())


def test_prfs():
    flow = DataFlow(
        Features(
            {
                "y_true": Sequence(Value("bool"), length=3),
                "y_pred": Sequence(Value("bool"), length=3),
            }
        )
    )

    with (
        patch("hyped.ops.MultiLabelConfusionMatrix") as mcm_mock,
        patch("hyped.ops.PrecisionRecallFScoreSupport") as prfs_mock,
    ):
        # call the operator
        ops.precision_recall_fscore_support(
            y_true=flow.src_features.y_true,
            y_pred=flow.src_features.y_pred,
            labels=[0, 1, 2],
            beta=1.0,
            average="micro",
        )
        # assert processors were called
        mcm_mock().call.assert_called_once_with(
            y_true=flow.src_features.y_true,
            y_pred=flow.src_features.y_pred,
        )
        prfs_mock().call.assert_called_once_with(
            confusion_matrix=mcm_mock().call().confusion_matrix
        )

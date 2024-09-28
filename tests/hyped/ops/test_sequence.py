from unittest.mock import patch

import pytest
from datasets import Features, Sequence, Value

from hyped import ops
from hyped.core.flow import DataFlow
from hyped.core.refs.ref import FeatureRef


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

    with patch("hyped.ops.sequence.ops.SequenceLength") as mock:
        # test constant length sequence
        out = flow.src_features.constant_seq.length_()
        # make sure the processor was not called and check the output
        assert not mock().call.called
        assert out == 5

    with patch("hyped.ops.sequence.ops.SequenceLength") as mock:
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
    [(ops.chain, "hyped.ops.sequence.ops.SequenceChain")],
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
        patch("hyped.ops.sequence.collect") as collect_mock,
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

    with patch("hyped.ops.sequence.ops.SequenceGetItem") as mock:
        ops.get_item(flow.src_features.seq, flow.src_features.idx)
        # make sure processor was called correctly
        mock().call.assert_called_once_with(
            sequence=flow.src_features.seq, index=flow.src_features.idx
        )

    with patch("hyped.ops.sequence.ops.SequenceSetItem") as mock:
        ops.set_item(flow.src_features.seq, flow.src_features.idx, flow.src_features.val)
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
    with patch("hyped.ops.sequence.ops.BooleanIndexing") as mock:
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
            "hyped.ops.sequence.ops.SequenceContains",
        ),
        (
            ops.count_of,
            "hyped.ops.sequence.ops.SequenceCountOf",
        ),
        (
            ops.index_of,
            "hyped.ops.sequence.ops.SequenceIndexOf",
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
    [(ops.zip_, "hyped.ops.sequence.ops.SequenceZip")],
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
        patch("hyped.ops.sequence.collect") as collect_mock,
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


def test_chunk_op():
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
        patch("hyped.ops.sequence.ops.SequenceChunk") as proc_mock,
        patch("hyped.ops.sequence.collect") as collect_mock,
    ):
        # call the operator
        ops.chunk(
            [
                flow.src_features.a,
                flow.src_features.b,
                flow.src_features.c,
            ],
            chunk_size=3,
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
        proc_mock().call.assert_called_once_with(collect_mock())

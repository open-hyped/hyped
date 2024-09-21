from datasets import Features, Sequence, Value

from hyped.nodes._ops.sequence import access
from tests.hyped.nodes.base import BaseDataProcessorTest


class TestSequenceGetItem(BaseDataProcessorTest):
    processor_type = access.SequenceGetItem
    processor_config = access.SequenceGetItemConfig()

    input_features = Features({"sequence": Sequence(Value("int32")), "index": Value("int32")})
    input_data = {
        "sequence": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "index": [0, 1, 2],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 5, 9]}


class TestSequenceGetItem_MultiIndex(BaseDataProcessorTest):
    processor_type = access.SequenceGetItem
    processor_config = access.SequenceGetItemConfig()

    input_features = Features(
        {
            "sequence": Sequence(Value("int32")),
            "index": Sequence(Value("int32")),
        }
    )
    input_data = {
        "sequence": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "index": [[0], [0, 1], [0, 1, 2]],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[1], [4, 5], [7, 8, 9]]}


class TestSequenceGetItem_MultiIndex_FixedLength(BaseDataProcessorTest):
    processor_type = access.SequenceGetItem
    processor_config = access.SequenceGetItemConfig()

    input_features = Features(
        {
            "sequence": Sequence(Value("int32")),
            "index": Sequence(Value("int32"), length=2),
        }
    )
    input_data = {
        "sequence": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "index": [[0, 1], [1, 2], [0, 2]],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Sequence(Value("int32"), length=2)})
    expected_output_data = {"result": [[1, 2], [5, 6], [7, 9]]}


class TestSequenceSetItem(BaseDataProcessorTest):
    processor_type = access.SequenceSetItem
    processor_config = access.SequenceSetItemConfig()

    input_features = Features(
        {
            "sequence": Sequence(Value("int32")),
            "index": Value("int32"),
            "value": Value("int32"),
        }
    )
    input_data = {
        "sequence": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "index": [0, 1, 2],
        "value": [-1, -2, -3],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[-1, 2, 3], [4, -2, 6], [7, 8, -3]]}


class TestSequenceSetItem_MultiIndex(BaseDataProcessorTest):
    processor_type = access.SequenceSetItem
    processor_config = access.SequenceSetItemConfig()

    input_features = Features(
        {
            "sequence": Sequence(Value("int32")),
            "index": Sequence(Value("int32")),
            "value": Sequence(Value("int32")),
        }
    )
    input_data = {
        "sequence": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "index": [[0], [0, 1], [0, 1, 2]],
        "value": [[-1], [-1, -2], [-1, -2, -3]],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[-1, 2, 3], [-1, -2, 6], [-1, -2, -3]]}


class TestSequenceSetItem_MultiIndex_FixedLength(BaseDataProcessorTest):
    processor_type = access.SequenceSetItem
    processor_config = access.SequenceSetItemConfig()

    input_features = Features(
        {
            "sequence": Sequence(Value("int32")),
            "index": Sequence(Value("int32"), length=2),
            "value": Sequence(Value("int32"), length=2),
        }
    )
    input_data = {
        "sequence": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "index": [[0, 1], [1, 2], [0, 2]],
        "value": [[-1, -2], [-1, -2], [-1, -2]],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[-1, -2, 3], [4, -1, -2], [-1, 8, -2]]}


class TestBooleanIndexing(BaseDataProcessorTest):
    processor_type = access.BooleanIndexing
    processor_config = access.BooleanIndexing.Config()

    input_features = Features({"values": Sequence(Value("int32")), "mask": Sequence(Value("bool"))})
    input_data = {
        "values": [[1, 2, 3, 4], [10, 20, 30]],
        "mask": [[True, False, True, False], [False, True, True]],
    }

    expected_output_features = Features({"indexed_values": Sequence(Value("int32"), length=-1)})
    expected_output_data = {"indexed_values": [[1, 3], [20, 30]]}


class TestBooleanIndexingWithEmptySequences(BaseDataProcessorTest):
    processor_type = access.BooleanIndexing
    processor_config = access.BooleanIndexing.Config()

    input_features = Features({"values": Sequence(Value("int32")), "mask": Sequence(Value("bool"))})
    input_data = {
        "values": [[], [1, 2, 3]],
        "mask": [[], [False, False, False]],
    }

    expected_output_features = Features({"indexed_values": Sequence(Value("int32"), length=-1)})
    expected_output_data = {"indexed_values": [[], []]}


class TestBooleanIndexingWithStrings(BaseDataProcessorTest):
    processor_type = access.BooleanIndexing
    processor_config = access.BooleanIndexing.Config()

    input_features = Features(
        {"values": Sequence(Value("string")), "mask": Sequence(Value("bool"))}
    )
    input_data = {
        "values": [["apple", "banana", "cherry"], ["dog", "cat", "mouse"]],
        "mask": [[True, False, True], [True, True, False]],
    }

    expected_output_features = Features({"indexed_values": Sequence(Value("string"), length=-1)})
    expected_output_data = {"indexed_values": [["apple", "cherry"], ["dog", "cat"]]}


class TestBooleanIndexingWithFixedLengthSequences(BaseDataProcessorTest):
    processor_type = access.BooleanIndexing
    processor_config = access.BooleanIndexing.Config()

    input_features = Features(
        {
            "values": Sequence(Value("float32"), length=3),
            "mask": Sequence(Value("bool"), length=3),
        }
    )
    input_data = {
        "values": [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]],
        "mask": [
            [True, False, True],
            [False, True, False],
            [True, True, True],
        ],
    }

    expected_output_features = Features({"indexed_values": Sequence(Value("float32"), length=-1)})
    expected_output_data = {"indexed_values": [[1.1, 3.3], [5.5], [7.7, 8.8, 9.9]]}


class TestBooleanIndexingRaisesErrorOnMismatchedSequenceLengths(BaseDataProcessorTest):
    processor_type = access.BooleanIndexing
    processor_config = access.BooleanIndexing.Config()

    input_features = Features(
        {
            "values": Sequence(Value("float32"), length=5),
            "mask": Sequence(Value("bool"), length=3),
        }
    )

    expected_input_verification_error = RuntimeError

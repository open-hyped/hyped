from datasets import Features, Sequence, Value

from hyped.data.flow.processors.ops.sequence import item
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestSequenceGetItem(BaseDataProcessorTest):
    processor_type = item.SequenceGetItem
    processor_config = item.SequenceGetItemConfig()

    input_features = Features({"sequence": Sequence(Value("int32")), "index": Value("int32")})
    input_data = {
        "sequence": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "index": [0, 1, 2],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"gathered": Value("int32")})
    expected_output_data = {"gathered": [1, 5, 9]}


class TestSequenceGetItem_MultiIndex(BaseDataProcessorTest):
    processor_type = item.SequenceGetItem
    processor_config = item.SequenceGetItemConfig()

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

    expected_output_features = Features({"gathered": Sequence(Value("int32"))})
    expected_output_data = {"gathered": [[1], [4, 5], [7, 8, 9]]}


class TestSequenceGetItem_MultiIndex_FixedLength(BaseDataProcessorTest):
    processor_type = item.SequenceGetItem
    processor_config = item.SequenceGetItemConfig()

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

    expected_output_features = Features({"gathered": Sequence(Value("int32"), length=2)})
    expected_output_data = {"gathered": [[1, 2], [5, 6], [7, 9]]}


class TestSequenceSetItem(BaseDataProcessorTest):
    processor_type = item.SequenceSetItem
    processor_config = item.SequenceSetItemConfig()

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
    processor_type = item.SequenceSetItem
    processor_config = item.SequenceSetItemConfig()

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
    processor_type = item.SequenceSetItem
    processor_config = item.SequenceSetItemConfig()

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

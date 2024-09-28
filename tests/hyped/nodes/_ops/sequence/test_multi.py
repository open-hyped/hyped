from datasets import Features, Sequence, Value

from hyped.nodes._ops.sequence import multi
from tests.hyped.nodes.base import BaseDataProcessorTest


class TestSequenceChainInvalidTypes(BaseDataProcessorTest):
    processor_type = multi.SequenceChain
    processor_config = multi.SequenceChainConfig()
    input_features = Features(
        {
            "sequences": {
                "0": Sequence(Value("string")),
                "1": Sequence(Value("int32")),
            }
        }
    )
    expected_input_verification_error = RuntimeError


class TestSequenceChainNonConsecutiveIntegerMultiIndex(BaseDataProcessorTest):
    processor_type = multi.SequenceChain
    processor_config = multi.SequenceChainConfig()
    input_features = Features(
        {
            "sequences": {
                "1": Sequence(Value("string")),
                "3": Sequence(Value("string")),
            }
        }
    )
    expected_input_verification_error = RuntimeError


class TestSequenceChainNonIntegerMultiIndex(BaseDataProcessorTest):
    processor_type = multi.SequenceChain
    processor_config = multi.SequenceChainConfig()
    input_features = Features(
        {
            "sequences": {
                "a": Sequence(Value("string")),
                "b": Sequence(Value("string")),
            }
        }
    )
    expected_input_verification_error = RuntimeError


class TestSequenceChain(BaseDataProcessorTest):
    processor_type = multi.SequenceChain
    processor_config = multi.SequenceChainConfig()

    input_features = Features(
        {
            "sequences": {
                "0": Sequence(Value("int32")),
                "1": Sequence(Value("int32")),
            }
        }
    )
    input_data = {
        "sequences": [
            {
                "0": [1, 2, 3],
                "1": [4, 5, 6],
            }
        ]
    }
    input_index = [0]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[1, 2, 3, 4, 5, 6]]}


class TestSequenceChain_FixedLength(BaseDataProcessorTest):
    processor_type = multi.SequenceChain
    processor_config = multi.SequenceChainConfig()

    input_features = Features(
        {
            "sequences": {
                "0": Sequence(Value("int32"), length=3),
                "1": Sequence(Value("int32"), length=3),
            }
        }
    )
    input_data = {
        "sequences": [
            {
                "0": [1, 2, 3],
                "1": [4, 5, 6],
            }
        ]
    }
    input_index = [0]

    expected_output_features = Features({"result": Sequence(Value("int32"), length=6)})
    expected_output_data = {"result": [[1, 2, 3, 4, 5, 6]]}


class TestSequenceZip(BaseDataProcessorTest):
    processor_type = multi.SequenceZip
    processor_config = multi.SequenceZipConfig()

    input_features = Features(
        {
            "sequences": {
                "0": Sequence(Value("int32"), length=4),
                "1": Sequence(Value("int32"), length=4),
            }
        }
    )
    input_data = {
        "sequences": [
            {
                "0": [0, 1, 2, 3],
                "1": [4, 5, 6, 7],
            }
        ]
    }
    input_index = [0]

    expected_output_features = Features(
        {"result": Sequence(Sequence(Value("int32"), length=2), length=4)}
    )
    expected_output_data = {"result": [[(0, 4), (1, 5), (2, 6), (3, 7)]]}

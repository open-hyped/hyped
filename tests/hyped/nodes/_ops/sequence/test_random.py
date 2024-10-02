from unittest.mock import patch

import pytest
from datasets import Features, Sequence, Value

from hyped.nodes._ops.sequence.random import SequenceChoice, SequenceChoiceConfig
from tests.hyped.nodes.base import BaseDataProcessorTest


@pytest.fixture(autouse=True)
def patch_random_choice():
    with patch("hyped.nodes._ops.sequence.random.choice", lambda x: x[0]):
        yield


class TestSequenceRandomChoice(BaseDataProcessorTest):
    # processor type
    processor_type = SequenceChoice
    # processor config
    processor_config = SequenceChoiceConfig()
    # input
    input_features = Features({"sequence": Sequence(Value("int64"))})
    input_data = {
        "sequence": [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
    }
    input_index = [0, 1, 2]

    # expected output data when patching random.choice
    expected_output_data = {
        "result": [1, 5, 9]  # These will be selected by the patched random.choice
    }

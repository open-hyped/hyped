from datasets import Features, Sequence, Value

from hyped.processors.spans.overlaps import ResolveOverlaps, ResolveOverlapsConfig
from tests.hyped.processors.base import BaseDataProcessorTest


class TestResolveOverlaps(BaseDataProcessorTest):
    processor_type = ResolveOverlaps
    processor_config = ResolveOverlapsConfig()

    input_features = Features({"spans": Sequence(Sequence(Value("int32"), length=2))})
    input_data = {"spans": [[], [(1, 5), (7, 12)], [(1, 5), (5, 12)], [(1, 5), (3, 9)]]}
    input_index = [0, 1, 2, 3]

    expected_output_features = Features(
        {
            "spans": Sequence(Sequence(Value("int32"), length=2)),
            "mask": Sequence(Value("bool")),
        }
    )
    expected_output_data = {
        "spans": [[], [(1, 5), (7, 12)], [(1, 5), (5, 12)], [(3, 9)]],
        "mask": [[], [True, True], [True, True], [False, True]],
    }

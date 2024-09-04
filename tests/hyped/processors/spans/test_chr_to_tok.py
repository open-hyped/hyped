from datasets import Features, Sequence, Value

from hyped.processors.spans.chr_to_tok import ChrToTokSpans, ChrToTokSpansConfig
from tests.hyped.processors.base import BaseDataProcessorTest


class TestChrToTokSpans(BaseDataProcessorTest):
    # processor
    processor_type = ChrToTokSpans
    processor_config = ChrToTokSpansConfig()
    # input specification
    input_features = Features(
        {
            "chr_spans": Sequence(Sequence(Value("int32"), length=2)),
            "query_spans": Sequence(Sequence(Value("int32"), length=2)),
        }
    )
    input_data = {
        "chr_spans": [
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
        ],
        "query_spans": [
            [],
            [(0, 9)],
            [(0, 9), (14, 18)],
            [(6, 9), (14, 18)],
            [(0, 13), (14, 18), (18, 25)],
        ],
    }
    input_index = [0, 1, 2, 3, 4]
    # expected output specification
    expected_output_feature = Features({"tok_spans": Sequence(Sequence(Value("int32"), length=2))})
    expected_output_data = {
        "tok_spans": [
            [],
            [(0, 2)],
            [(0, 2), (3, 4)],
            [(1, 2), (3, 4)],
            [(0, 3), (3, 4), (4, 5)],
        ]
    }


class TestChrToTokSpans_Masked(BaseDataProcessorTest):
    # processor
    processor_type = ChrToTokSpans
    processor_config = ChrToTokSpansConfig()
    # input specification
    input_features = Features(
        {
            "chr_spans": Sequence(Sequence(Value("int32"), length=2)),
            "query_spans": Sequence(Sequence(Value("int32"), length=2)),
            "special_tokens_mask": Sequence(Value("int32")),
        }
    )
    input_data = {
        "chr_spans": [
            [(0, 0), (0, 5), (6, 9), (10, 13), (14, 18), (18, 25), (0, 0)],
            [(0, 0), (0, 5), (6, 9), (10, 13), (14, 18), (18, 25), (0, 0)],
            [(0, 0), (0, 5), (6, 9), (10, 13), (14, 18), (18, 25), (0, 0)],
            [(0, 0), (0, 5), (6, 9), (10, 13), (14, 18), (18, 25), (0, 0)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25), (0, 0)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25), (0, 0)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25), (0, 0)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25), (0, 0)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
        ],
        "query_spans": [
            [],
            [(0, 9)],
            [(0, 9), (14, 18)],
            [(6, 9), (14, 18)],
            [],
            [(0, 9)],
            [(0, 9), (14, 18)],
            [(6, 9), (14, 18)],
            [],
            [(0, 9)],
            [(0, 9), (14, 18)],
            [(6, 9), (14, 18)],
        ],
        "special_tokens_mask": [
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    }
    input_index = list(range(12))
    # expected output specification
    expected_output_feature = Features({"tok_spans": Sequence(Sequence(Value("int32"), length=2))})
    expected_output_data = {
        "tok_spans": [
            [],
            [(1, 3)],
            [(1, 3), (4, 5)],
            [(2, 3), (4, 5)],
            [],
            [(0, 2)],
            [(0, 2), (3, 4)],
            [(1, 2), (3, 4)],
            [],
            [(0, 2)],
            [(0, 2), (3, 4)],
            [(1, 2), (3, 4)],
        ]
    }


class TestChrToTokSpans_ExcludePartials(BaseDataProcessorTest):
    # processor
    processor_type = ChrToTokSpans
    processor_config = ChrToTokSpansConfig(
        include_partial_start=False,
        include_partial_end=False,
    )
    # input specification
    input_features = Features(
        {
            "chr_spans": Sequence(Sequence(Value("int32"), length=2)),
            "query_spans": Sequence(Sequence(Value("int32"), length=2)),
        }
    )
    input_data = {
        "chr_spans": [
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
        ],
        "query_spans": [
            [(4, 11)],  # Span partly cover tokens
            [(14, 25)],  # Span matches token borders
        ],
    }
    input_index = [0, 1]
    # expected output specification
    expected_output_feature = Features({"tok_spans": Sequence(Sequence(Value("int32"), length=2))})
    expected_output_data = {"tok_spans": [[(1, 2)], [(3, 5)]]}  # Only fully covered tokens included


class TestChrToTokSpans_IncludePartialStart(BaseDataProcessorTest):
    # processor
    processor_type = ChrToTokSpans
    processor_config = ChrToTokSpansConfig(
        include_partial_start=True,
        include_partial_end=False,
    )
    # input specification
    input_features = Features(
        {
            "chr_spans": Sequence(Sequence(Value("int32"), length=2)),
            "query_spans": Sequence(Sequence(Value("int32"), length=2)),
        }
    )
    input_data = {
        "chr_spans": [
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
        ],
        "query_spans": [
            [(4, 11)],  # Span partly cover tokens
            [(14, 25)],  # Span matches token borders
        ],
    }
    input_index = [0, 1]
    # expected output specification
    expected_output_feature = Features({"tok_spans": Sequence(Sequence(Value("int32"), length=2))})
    expected_output_data = {
        "tok_spans": [
            [(0, 2)],  # Includes token at start
            [(3, 5)],
        ]
    }


class TestChrToTokSpans_IncludePartialEnd(BaseDataProcessorTest):
    # processor
    processor_type = ChrToTokSpans
    processor_config = ChrToTokSpansConfig(
        include_partial_start=False,
        include_partial_end=True,
    )
    # input specification
    input_features = Features(
        {
            "chr_spans": Sequence(Sequence(Value("int32"), length=2)),
            "query_spans": Sequence(Sequence(Value("int32"), length=2)),
        }
    )
    input_data = {
        "chr_spans": [
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
        ],
        "query_spans": [
            [(4, 11)],  # Span partly cover tokens
            [(14, 25)],  # Span matches token borders
        ],
    }
    input_index = [0, 1]
    # expected output specification
    expected_output_feature = Features({"tok_spans": Sequence(Sequence(Value("int32"), length=2))})
    expected_output_data = {
        "tok_spans": [
            [(1, 3)],  # Includes token at end
            [(3, 5)],
        ]
    }


class TestChrToTokSpans_IncludePartialBoth(BaseDataProcessorTest):
    # processor
    processor_type = ChrToTokSpans
    processor_config = ChrToTokSpansConfig(
        include_partial_start=True,
        include_partial_end=True,
    )
    # input specification
    input_features = Features(
        {
            "chr_spans": Sequence(Sequence(Value("int32"), length=2)),
            "query_spans": Sequence(Sequence(Value("int32"), length=2)),
        }
    )
    input_data = {
        "chr_spans": [
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
        ],
        "query_spans": [
            [(4, 11)],  # Span partly cover tokens
            [(14, 25)],  # Span matches token borders
        ],
    }
    input_index = [0, 1]
    # expected output specification
    expected_output_feature = Features({"tok_spans": Sequence(Sequence(Value("int32"), length=2))})
    expected_output_data = {
        "tok_spans": [
            [(0, 3)],  # Includes tokens at both start and end
            [(3, 5)],
        ]
    }

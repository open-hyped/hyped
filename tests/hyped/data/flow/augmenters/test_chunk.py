from datasets import Features, Sequence, Value

from hyped.data.flow.augmenters.chunk import ChunkSequence
from tests.hyped.data.flow.augmenters.base import BaseDataAugmenterTest


class TestChunkSequenceEmptyInputs(BaseDataAugmenterTest):
    augmenter_type = ChunkSequence
    augmenter_config = ChunkSequence.Config(chunk_size=2, stride=2)

    input_features = Features({"sequences": {}})
    expected_input_verification_error = RuntimeError


class TestChunkSequenceInvalidInputs(BaseDataAugmenterTest):
    augmenter_type = ChunkSequence
    augmenter_config = ChunkSequence.Config(chunk_size=2, stride=2)

    input_features = Features(
        {
            "sequences": {
                "a": Sequence(Value("int32"), length=6),
                "b": Sequence(Value("int32"), length=8),
            }
        }
    )

    expected_input_verification_error = RuntimeError


class TestChunkSequenceFixedLength(BaseDataAugmenterTest):
    augmenter_type = ChunkSequence
    augmenter_config = ChunkSequence.Config(chunk_size=2, stride=2)

    input_features = Features(
        {
            "sequences": {
                "a": Sequence(Value("int32"), length=6),
                "b": Sequence(Value("int32"), length=6),
            }
        }
    )

    expected_output_features = Features(
        {
            "chunks": {
                "a": Sequence(Value("int32"), length=2),
                "b": Sequence(Value("int32"), length=2),
            }
        }
    )


class TestChunkSequence(BaseDataAugmenterTest):
    augmenter_type = ChunkSequence
    augmenter_config = ChunkSequence.Config(chunk_size=2, stride=2)

    input_features = Features(
        {
            "sequences": {
                "a": Sequence(Value("int32")),
                "b": Sequence(Value("int32")),
            }
        }
    )
    input_data = {
        "sequences": [
            {"a": [0, 1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6, 7]},
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [3, 4, 5, 6, 7, 8],
            },
        ]
    }
    input_index = [0, 1]

    expected_output_features = Features(
        {
            "chunks": {
                "a": Sequence(Value("int32")),
                "b": Sequence(Value("int32")),
            }
        }
    )
    expected_output_data = {
        "chunks": [
            # chunks of sample 1
            {"a": [0, 1], "b": [2, 3]},
            {"a": [2, 3], "b": [4, 5]},
            {"a": [4, 5], "b": [6, 7]},
            # chunks of sample 2
            {"a": [1, 2], "b": [3, 4]},
            {"a": [3, 4], "b": [5, 6]},
            {"a": [5, 6], "b": [7, 8]},
        ]
    }


class TestChunkSequenceDifferentChunkSizeStride(BaseDataAugmenterTest):
    augmenter_type = ChunkSequence
    augmenter_config = ChunkSequence.Config(chunk_size=3, stride=2)

    input_features = Features(
        {
            "sequences": {
                "a": Sequence(Value("int32")),
                "b": Sequence(Value("int32")),
            }
        }
    )
    input_data = {
        "sequences": [
            {"a": [0, 1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14, 15]},
            {"a": [5, 6, 7, 8, 9, 10], "b": [15, 16, 17, 18, 19, 20]},
        ]
    }
    input_index = [0, 1]

    expected_output_features = Features(
        {
            "chunks": {
                "a": Sequence(Value("int32")),
                "b": Sequence(Value("int32")),
            }
        }
    )
    expected_output_data = {
        "chunks": [
            # chunks of sample 1
            {"a": [0, 1, 2], "b": [10, 11, 12]},
            {"a": [2, 3, 4], "b": [12, 13, 14]},
            {
                "a": [4, 5],
                "b": [14, 15],
            },  # Remaining chunk (shorter because it's the end of the sequence)
            # chunks of sample 2
            {"a": [5, 6, 7], "b": [15, 16, 17]},
            {"a": [7, 8, 9], "b": [17, 18, 19]},
            {
                "a": [9, 10],
                "b": [19, 20],
            },  # Remaining chunk (shorter because it's the end of the sequence)
        ]
    }


class TestChunkSequenceKeepLastTrueNoRemainder(BaseDataAugmenterTest):
    augmenter_type = ChunkSequence
    augmenter_config = ChunkSequence.Config(
        chunk_size=3, stride=2, keep_last=True
    )

    input_features = Features(
        {
            "sequences": {
                "a": Sequence(Value("int32"), length=5),
                "b": Sequence(Value("int32"), length=5),
            }
        }
    )
    input_data = {
        "sequences": [
            {"a": [0, 1, 2, 3, 4], "b": [10, 11, 12, 13, 14]},
            {"a": [5, 6, 7, 8, 9], "b": [15, 16, 17, 18, 19]},
        ]
    }
    input_index = [0, 1]

    expected_output_features = Features(
        {
            "chunks": {
                "a": Sequence(Value("int32"), length=3),
                "b": Sequence(Value("int32"), length=3),
            }
        }
    )
    expected_output_data = {
        "chunks": [
            # chunks of sample 1
            {"a": [0, 1, 2], "b": [10, 11, 12]},
            {"a": [2, 3, 4], "b": [12, 13, 14]},
            # chunks of sample 2
            {"a": [5, 6, 7], "b": [15, 16, 17]},
            {"a": [7, 8, 9], "b": [17, 18, 19]},
        ]
    }


class TestChunkSequenceKeepLastTrueWithRemainder(BaseDataAugmenterTest):
    augmenter_type = ChunkSequence
    augmenter_config = ChunkSequence.Config(
        chunk_size=3, stride=2, keep_last=True
    )

    input_features = Features(
        {
            "sequences": {
                "a": Sequence(Value("int32")),
                "b": Sequence(Value("int32")),
            }
        }
    )
    input_data = {
        "sequences": [
            {"a": [0, 1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14, 15]},
            {"a": [5, 6, 7, 8, 9, 10], "b": [15, 16, 17, 18, 19, 20]},
        ]
    }
    input_index = [0, 1]

    expected_output_features = Features(
        {
            "chunks": {
                "a": Sequence(Value("int32"), length=-1),
                "b": Sequence(Value("int32"), length=-1),
            }
        }
    )
    expected_output_data = {
        "chunks": [
            # chunks of sample 1
            {"a": [0, 1, 2], "b": [10, 11, 12]},
            {"a": [2, 3, 4], "b": [12, 13, 14]},
            {
                "a": [4, 5],
                "b": [14, 15],
            },  # Remaining chunk (shorter because it's the end of the sequence)
            # chunks of sample 2
            {"a": [5, 6, 7], "b": [15, 16, 17]},
            {"a": [7, 8, 9], "b": [17, 18, 19]},
            {
                "a": [9, 10],
                "b": [19, 20],
            },  # Remaining chunk (shorter because it's the end of the sequence)
        ]
    }

"""Feature checking functionality."""
from inspect import isclass
from typing import Any

import pyarrow as pa
from datasets import Features, Sequence, Value
from datasets.features.features import FeatureType

from .arrow import convert_features_to_arrow_schema

INT_TYPES = [
    Value("int8"),
    Value("int16"),
    Value("int32"),
    Value("int64"),
]

UINT_TYPES = [
    Value("uint8"),
    Value("uint16"),
    Value("uint32"),
    Value("uint64"),
]

FLOAT_TYPES = [
    Value("float16"),
    Value("float32"),
    Value("float64"),
]

STRING_LIKE_TYPES = [
    Value("string"),
    Value("large_string"),
    Value("binary"),
    Value("large_binary"),
]

NUMERIC_TYPES = INT_TYPES + UINT_TYPES + FLOAT_TYPES
SCALAR_TYPES = NUMERIC_TYPES + [Value("bool")]
INDEX_TYPES = INT_TYPES + UINT_TYPES


def check_feature_equals(
    feature: FeatureType, target: FeatureType | list[FeatureType]
) -> bool:
    """Check whether a given feature equals a target feature.

    This confirms exact matches, including for instance
    checking that the lengths of sequences match.

    Arguments:
        feature (FeatureType):
            the feature to check, for mapping the function is
            called recursive for each member
        target (FeatureType | list[FeatureType]):
            the target feature or list of valid target features.
            If the target is a list or tuple of length one that
            it is still considered to be a feature and not a list
            of features as this is also a definition of a sequence
            feature type.

    Returns:
        is_equal (bool):
            bool indicating whether the feature matches
            the target feature type
    """
    if isinstance(target, (list, tuple)):
        # if multiple valid targets are given the feature
        # should match any one of them
        if len(target) != 1:
            return any(check_feature_equals(feature, t) for t in target)
        # a list of length one is a valid definition of sequence
        return check_feature_is_sequence(
            feature, target[0]
        ) and check_sequence_lengths_match(feature, target)

    # a list of length one is a valid definition of a sequence
    if isinstance(feature, list):
        # at this point the target cannot be a list so to
        # align with the given feature is must be sequence
        if isinstance(target, Sequence):
            return check_feature_is_sequence(
                feature, target.feature
            ) and check_sequence_lengths_match(feature, target)

        return target is Sequence

    if isclass(target):
        # if only the target feature class and
        # not the exact target feature is specified
        return isinstance(feature, target)

    if isinstance(feature, dict):
        # make sure the target is also a mapping
        # with the same keys and ensure that the
        # features are equal as well
        return (
            isinstance(target, dict)
            and (feature.keys() == target.keys())
            and all(
                check_feature_equals(feature[k], target[k])
                for k in feature.keys()
            )
        )

    # otherwise it should just match the target
    return feature == target


def check_feature_is_sequence(
    feature: FeatureType,
    value_type: None | FeatureType | list[FeatureType] = None,
) -> bool:
    """Check Feature is Sequence.

    Check if a given feature is a sequence of values of
    a given value type (and arbitrary length).

    Arguments:
        feature (FeatureType):
            the feature to check
        value_type (None | FeatureType | list[FeatureType]):
            the expected sequence value type or a target
            of valid value types. If the value type is a list or
            tuple of length one, it is still considered to be a
            feature and not a list of features as that is also a
            valid definition of a sequence feature type. If the
            value type is set to None, the item type is not
            checked.

    Returns:
        is_sequence (bool):
            whether the feature is a sequence of (one of) the
            given value type(s)
    """
    if value_type is None:
        # only check if the feature is a sequence
        return isinstance(feature, (Sequence, list, tuple))

    return isinstance(
        feature, (Sequence, list, tuple)
    ) and check_feature_equals(get_sequence_feature(feature), value_type)


def get_sequence_length(seq: Sequence | list | tuple) -> int:
    """Get the length of a given sequence feature.

    Arguments:
        seq (Sequence | list | tuple): sequence to get the length of

    Returns:
        length (int):
            the length of the given sequence. Returns -1 for
            sequences of undefined length
    """
    assert isinstance(seq, (Sequence, list, tuple)), seq
    return seq.length if isinstance(seq, Sequence) else -1


def get_sequence_feature(seq: Sequence | list | tuple) -> FeatureType:
    """Get the item feature type of a given sequence feature.

    Arguments:
        seq (Sequence | list | tuple):
            sequence to get the item feature type of

    Returns:
        feature (FeatureType):
            the item feature type of the sequence
    """
    assert isinstance(seq, (Sequence, list, tuple))
    return seq.feature if isinstance(seq, Sequence) else seq[0]


def check_sequence_lengths_match(
    seq_A: Sequence | list | tuple,
    seq_B: Sequence | list | tuple,
    allow_arbitrary_length: bool = True,
) -> bool:
    """Check whether the lengths of two sequences match.

    Arguments:
        seq_A (Sequence | list | tuple): sequence A
        seq_B (Sequence | list | tuple): sequence B
        allow_arbitrary_length (bool): Wether to support -1 as valid sequence length

    Returns:
        match (bool): bool indicating if the lengths match
    """
    a_len = get_sequence_length(seq_A)
    b_len = get_sequence_length(seq_B)

    print(a_len, b_len)

    if (not allow_arbitrary_length) and (a_len == -1 or b_len == -1):
        return False

    return a_len == b_len


def check_object_matches_feature(obj: Any, feature: FeatureType):
    """Check whether the object is of the given feature type.

    Arguments:
        obj (Any): the object whichs type to check
        feature (FeatureType): the feature type to check for

    Returns:
        is_of_type (bool):
            boolean indicating whether the object is of the
            feature type or not
    """
    # preparation
    data = {"__obj__": [obj]}
    features = Features({"__obj__": feature})

    try:
        # try to encode data into pyarrow table which internally checks types
        pa.table(data=data, schema=convert_features_to_arrow_schema(features))
        return True

    except (pa.lib.ArrowTypeError, pa.lib.ArrowInvalid):
        # catch error in type check
        return False


def check_value_feature_is_castable(feat_A: Value, feat_B: Value) -> bool:
    """Check whether two Value features are castable to each other.

    Arguments:
        feat_A (Value): the first Value feature to check.
        feat_B (Value): the second Value feature to check.

    Returns:
        is_castable (bool):
            Boolean indicating whether the two Value features
            can be cast to each other or not.

    Raises:
        AssertionError: if either feat_A or feat_B is not an instance of Value.
    """
    assert isinstance(feat_A, Value)
    assert isinstance(feat_B, Value)

    # If both features are scalar types, they are castable
    return feat_A in SCALAR_TYPES and feat_B in SCALAR_TYPES


def raise_feature_equals(
    name: str, feature: FeatureType, target: FeatureType | list[FeatureType]
) -> None:
    """Check whether a given feature equals a target feature.

    This confirms exact matches, including for instance
    checking that the lengths of sequences match.

    Arguments:
        name (str):
            the name of the feature, only used in the error message
        feature (FeatureType):
            the feature to check
        target (FeatureType | list[FeatureType]):
            the target feature or list of valid target features.
            If the target is a list or tuple of length one that
            it is still considered to be a feature and not a list
            of features as that is also a definition of a sequence
            feature type.

    Raises:
        exc (TypeError): when the feature doesn't equal the target
    """
    if not check_feature_equals(feature, target):
        if isinstance(target, list):
            # slightly different error message for list of
            # value feature types
            raise TypeError(
                "Expected `%s` to be a one of the types "
                "in %s, got %s" % (name, target, type(feature))
            )

        raise TypeError(
            "Expected `%s` to be of type %s, got %s" % (name, target, feature)
        )


def raise_feature_is_sequence(
    name: str,
    feature: FeatureType,
    value_type: None | FeatureType | list[FeatureType] = None,
) -> None:
    """Raise Feature is Sequence.

    Check if a given feature is a sequence of values of
    a given value type (and arbitrary length).

    Arguments:
        name (str):
            the name of the feature, only used in the error message
        feature (FeatureType):
            the feature to check
        value_type (FeatureType | list[FeatureType]):
            the expected sequence value type or a target
            of valid value types. If the value type is a list or
            tuple of length one, it is still considered to be a
            feature and not a list of features as this is also a
            valid definition of a sequence feature type. If the
            value type is set to None, the item type is not
            checked.

    Raises:
        exc (TypeError):
            when the value type doesn't match the expected value type
    """
    if not check_feature_is_sequence(feature, value_type):
        if value_type is None:
            raise TypeError(
                "Expected `%s` to be a sequence, got %s" % (name, feature)
            )

        if isinstance(value_type, (list, tuple)) and (len(value_type) > 1):
            # slightly different error message for list of
            # value feature types
            raise TypeError(
                "Expected `%s` to be a sequence of one of the types "
                "in %s, got %s" % (name, value_type, feature)
            )

        raise TypeError(
            "Expected `%s` to be a sequence of type %s, got %s"
            % (name, value_type, feature)
        )


def raise_sequence_lengths_match(
    name_A: str,
    name_B: str,
    seq_A: FeatureType,
    seq_B: FeatureType,
    allow_arbitrary_length: bool = True,
) -> None:
    """Raise an error if the lengths of two sequences do not match.

    Arguments:
        name_A (str): The name of the first sequence (used in the error message).
        name_B (str): The name of the second sequence (used in the error message).
        seq_A (FeatureType): The first sequence feature to compare.
        seq_B (FeatureType): The second sequence feature to compare.
        allow_arbitrary_length (bool, optional):
            If True, allows comparison of sequences where one or both have arbitrary length.
            Defaults to True.

    Raises:
        TypeError: If the lengths of seq_A and seq_B do not match. The error message will include
        the names of the sequences and their respective lengths.
    """
    if not check_sequence_lengths_match(seq_A, seq_B, allow_arbitrary_length):
        a_len = get_sequence_length(seq_A)
        b_len = get_sequence_length(seq_B)
        raise TypeError(
            f"Sequence lengths do not match. {name_A} is {a_len}, {name_B} is {b_len}"
        )


def raise_value_feature_is_castable(
    name_A: str, name_B: str, feat_A: Value, feat_B: Value
) -> None:
    """Raise an error if two Value features cannot be cast to each other.

    Arguments:
        name_A (str): The name of the first feature (used in the error message).
        name_B (str): The name of the second feature (used in the error message).
        feat_A (Value): The first Value feature to compare.
        feat_B (Value): The second Value feature to compare.

    Raises:
        TypeError: If feat_A and feat_B cannot be cast to each other. The error message will include
        the names of the features and their respective data types.
    """
    if not check_value_feature_is_castable(feat_A, feat_B):
        raise TypeError(
            f"Feature types cannot be casted: {name_A} is {feat_A.dtype}, "
            f"{name_B} is {feat_B.dtype}"
        )


def raise_object_matches_feature(obj: Any, feature: FeatureType):
    """Check whether the object is of the given feature type.

    Arguments:
        obj (Any): the object whichs type to check
        feature (FeatureType): the feature type to check for

    Raises:
        exc (TypeError):
            when the object is not of the feature type
    """
    if not check_object_matches_feature(obj, feature):
        raise TypeError(
            "Expected object to be of type %s, got %s" % (feature, obj)
        )

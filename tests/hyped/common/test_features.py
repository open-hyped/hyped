from typing import List, Literal, TypedDict

import pytest
from datasets import ClassLabel, Features, Sequence, Value

from hyped.common._features import convert_features_to_arrow_schema, type_hint_to_feature
from hyped.common.feature_checks import check_feature_equals


@pytest.mark.parametrize(
    "features",
    [
        # easy cases
        Features({"Test": Value("int32")}),
        Features({"Test": Sequence(Value("int32"), length=16)}),
        Features(
            {
                "A": Sequence(Value("int32"), length=16),
                "B": Sequence(Value("int32"), length=32),
            }
        ),
        Features(
            {
                "A": {
                    "0": Sequence(Value("int32"), length=16),
                    "1": Sequence(Value("int32"), length=16),
                },
                "B": Sequence(Value("int32"), length=32),
            }
        ),
        Features({"A": Sequence({"A": Value("int32"), "B": Value("int32")})}),
        Features(
            {
                "A": Sequence({"A": Value("int32"), "B": Value("int32")}),
                "B": Sequence({"A": Value("int32"), "B": Value("int32")}, length=16),
                "C": Sequence(
                    {
                        "A": Value("int32"),
                        "B": Value("int32"),
                        "C": Value("string"),
                    }
                ),
            }
        ),
    ],
)
def test_convert_features_to_arrow_schema(features):
    # convert, reconstruct and check
    schema = convert_features_to_arrow_schema(features)
    assert check_feature_equals(features, Features.from_arrow_schema(schema))


class SimpleTypedDict(TypedDict):
    id: int
    name: str


class ComplexTypedDict(SimpleTypedDict):
    scores: List[float]
    label: Literal["positive", "negative"]
    nested: SimpleTypedDict


class TestTypeHintToFeature:
    def test_primitive_types(self):
        # Test for each primitive type
        assert type_hint_to_feature(int) == Value("int32")
        assert type_hint_to_feature(float) == Value("float32")
        assert type_hint_to_feature(str) == Value("string")
        assert type_hint_to_feature(bool) == Value("bool")

    def test_list_type(self):
        # Test for list of floats
        assert type_hint_to_feature(List[float]) == [Value("float32")]

    def test_typed_dict(self):
        # Test for SimpleTypedDict
        expected = Features({"id": Value("int32"), "name": Value("string")})
        assert type_hint_to_feature(SimpleTypedDict) == expected

        # Test for ComplexTypedDict
        expected_complex = Features(
            {
                "id": Value("int32"),
                "name": Value("string"),
                "scores": [Value("float32")],
                "label": ClassLabel(names=["positive", "negative"]),
                "nested": Features({"id": Value("int32"), "name": Value("string")}),
            }
        )
        assert type_hint_to_feature(ComplexTypedDict) == expected_complex

    def test_literal(self):
        # Test for Literal
        expected_literal = ClassLabel(names=["positive", "negative"])
        assert type_hint_to_feature(Literal["positive", "negative"]) == expected_literal

    def test_fallback_case(self):
        # Test for unsupported type
        with pytest.raises(TypeError, match="Unsupported type:"):
            type_hint_to_feature(object())  # int should raise an error since it's not a TypeHint

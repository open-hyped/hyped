import json

import pydantic
from datasets import ClassLabel, Features, Sequence, Value

from hyped.common._pydantic import pydantic_model_from_features
from hyped.nodes.parsers.json import JsonParser, JsonParserConfig
from tests.hyped.nodes.base import BaseDataProcessorTest


class TestJsonParser_Value(BaseDataProcessorTest):
    # processor config
    processor_type = JsonParser
    processor_config = JsonParserConfig(scheme=Features({"value": Value("int32")}))
    # input
    input_features = Features({"payload": Value("string")})
    input_data = {
        "payload": [
            json.dumps({"value": 0}),
            json.dumps({"value": 1}),
            json.dumps({"value": 2}),
        ]
    }
    input_index = [0, 1, 2]
    # expected output
    expected_output_data = {
        "obj": [
            {"value": 0},
            {"value": 1},
            {"value": 2},
        ],
        "exception": [None, None, None],
    }


class TestJsonParser_Sequence(BaseDataProcessorTest):
    # processor
    processor_type = JsonParser
    processor_config = JsonParserConfig(scheme=Sequence(Value("int32"), length=3))
    # input
    input_features = Features({"payload": Value("string")})
    input_data = {
        "payload": [
            json.dumps([0, 0, 0]),
            json.dumps([1, 2, 3]),
            json.dumps([3, 2, 1]),
        ]
    }
    input_index = [0, 1, 2]
    # expected output
    expected_output_data = {
        "obj": [
            [0, 0, 0],
            [1, 2, 3],
            [3, 2, 1],
        ],
        "exception": [None, None, None],
    }


class TestJsonParser_Nested(BaseDataProcessorTest):
    # processor
    processor_type = JsonParser
    processor_config = JsonParserConfig(
        scheme=Features(
            {
                "a": Value("int32"),
                "b": Sequence(Value("int32")),
                "c": {
                    "x": Value("int32"),
                    "y": Sequence({"i": Value("int32")}),
                },
            }
        )
    )
    # input
    input_features = Features({"payload": Value("string")})
    input_data = {
        "payload": [
            json.dumps(
                {
                    "a": 0,
                    "b": [0, 0],
                    "c": {
                        "x": 0,
                        "y": [
                            {"i": 0},
                            {"i": 0},
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "a": 1,
                    "b": [2, 2],
                    "c": {"x": 3, "y": [{"i": 4}, {"i": 5}, {"i": 6}]},
                }
            ),
        ]
    }
    input_index = [0, 1]
    # expected output
    expected_output_data = {
        "obj": [
            {
                "a": 0,
                "b": [0, 0],
                "c": {
                    "x": 0,
                    "y": [
                        {"i": 0},
                        {"i": 0},
                    ],
                },
            },
            {
                "a": 1,
                "b": [2, 2],
                "c": {"x": 3, "y": [{"i": 4}, {"i": 5}, {"i": 6}]},
            },
        ],
        "exception": [None, None],
    }


class TestJsonParser_ClassLabel(BaseDataProcessorTest):
    # processor config
    processor_type = JsonParser
    processor_config = JsonParserConfig(
        scheme=Features({"label": ClassLabel(names=["A", "B", "C"])}),
    )
    # input
    input_features = Features({"payload": Value("string")})
    input_data = {
        "payload": [
            json.dumps({"label": "A"}),
            json.dumps({"label": "B"}),
            json.dumps({"label": "C"}),
        ]
    }
    input_index = [0, 1, 2]
    # expected output
    expected_output_data = {
        "obj": [
            {"label": 0},
            {"label": 1},
            {"label": 2},
        ],
        "exception": [None, None, None],
    }


def get_validation_error():
    features = Features({"parsed": {"value": Value("int32")}})
    payload = json.dumps({"parsed": {"value": "A"}})
    model = pydantic_model_from_features(features)
    try:
        model.model_validate_json(payload)
    except pydantic.ValidationError as validation_error:
        return validation_error


class TestJsonParser_CatchExceptions(BaseDataProcessorTest):
    # processor config
    processor_type = JsonParser
    processor_config = JsonParserConfig(scheme=Features({"value": Value("int32")}))
    # input
    input_features = Features({"payload": Value("string")})
    input_data = {
        "payload": [
            json.dumps({"value": "A"}),
            json.dumps({"value": 0}),
            json.dumps({"value": "A"}),
        ]
    }
    input_index = [0, 1, 2]
    # expected output
    expected_output_data = {
        "obj": [None, {"value": 0}, None],
        "exception": [
            str(get_validation_error()),
            None,
            str(get_validation_error()),
        ],
    }

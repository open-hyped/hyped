from datasets import Features, Sequence, Value

from hyped.processors.parsers.cas import CasParser
from tests.hyped.processors.base import BaseDataProcessorTest

with open("./tests/artifacts/cas/cas.json", "r") as f:
    cas_json_content = f.read()

with open("./tests/artifacts/cas/cas.xmi", "r") as f:
    cas_xmi_content = f.read()


class TestCasParser(BaseDataProcessorTest):
    processor_type = CasParser
    processor_config = CasParser.Config(typesystem="./tests/artifacts/cas/typesystem.xml")

    input_features = Features({"payload": Value("string")})
    input_data = {"payload": [cas_json_content, cas_xmi_content]}

    expected_output_features = Features(
        {
            "obj": {
                "sofa": Value("string"),
                "uima.tcas.DocumentAnnotation:language": Sequence(Value("string")),
                "uima.tcas.DocumentAnnotation:begin": Sequence(Value("int32")),
                "uima.tcas.DocumentAnnotation:end": Sequence(Value("int32")),
                "cassis.Entity:entityType": Sequence(Value("string")),
                "cassis.Entity:begin": Sequence(Value("int32")),
                "cassis.Entity:end": Sequence(Value("int32")),
                "cassis.Label:label": Sequence(Value("string")),
                "cassis.Relation:source": Sequence(Value("int32")),
                "cassis.Relation:target": Sequence(Value("int32")),
            },
            "exception": Value("string"),
        }
    )
    expected_output_data = {
        "obj": [
            {
                "sofa": "U.N. official Ekeus heads for Baghdad.",
                "uima.tcas.DocumentAnnotation:language": [],
                "uima.tcas.DocumentAnnotation:begin": [],
                "uima.tcas.DocumentAnnotation:end": [],
                "cassis.Entity:entityType": ["ORG", "LOC"],
                "cassis.Entity:begin": [0, 30],
                "cassis.Entity:end": [4, 37],
                "cassis.Label:label": ["Document"],
                "cassis.Relation:source": [0],
                "cassis.Relation:target": [1],
            },
            {
                "sofa": "U.N. official Ekeus heads for Baghdad.",
                "uima.tcas.DocumentAnnotation:language": [],
                "uima.tcas.DocumentAnnotation:begin": [],
                "uima.tcas.DocumentAnnotation:end": [],
                "cassis.Entity:entityType": ["ORG", "LOC"],
                "cassis.Entity:begin": [0, 30],
                "cassis.Entity:end": [4, 37],
                "cassis.Label:label": ["Document"],
                "cassis.Relation:source": [0],
                "cassis.Relation:target": [1],
            },
        ],
        "exception": [None, None],
    }


class TestCasParser_SelectTypes(BaseDataProcessorTest):
    processor_type = CasParser
    processor_config = CasParser.Config(
        typesystem="./tests/artifacts/cas/typesystem.xml",
        types=[
            "cassis.Entity",
            "cassis.Label",
        ],
    )

    input_features = Features({"payload": Value("string")})
    input_data = {"payload": [cas_json_content, cas_xmi_content]}

    expected_output_features = Features(
        {
            "obj": {
                "sofa": Value("string"),
                "cassis.Entity:entityType": Sequence(Value("string")),
                "cassis.Entity:begin": Sequence(Value("int32")),
                "cassis.Entity:end": Sequence(Value("int32")),
                "cassis.Label:label": Sequence(Value("string")),
            },
            "exception": Value("string"),
        }
    )
    expected_output_data = {
        "obj": [
            {
                "sofa": "U.N. official Ekeus heads for Baghdad.",
                "cassis.Entity:entityType": ["ORG", "LOC"],
                "cassis.Entity:begin": [0, 30],
                "cassis.Entity:end": [4, 37],
                "cassis.Label:label": ["Document"],
            },
            {
                "sofa": "U.N. official Ekeus heads for Baghdad.",
                "cassis.Entity:entityType": ["ORG", "LOC"],
                "cassis.Entity:begin": [0, 30],
                "cassis.Entity:end": [4, 37],
                "cassis.Label:label": ["Document"],
            },
        ],
        "exception": [None, None],
    }


class TestCasParser_InvalidPayload(BaseDataProcessorTest):
    processor_type = CasParser
    processor_config = CasParser.Config(typesystem="./tests/artifacts/cas/typesystem.xml")

    input_features = Features({"payload": Value("string")})
    input_data = {"payload": ["INVALID_SERIALIZED_CAS"]}

    expected_output_features = Features(
        {
            "obj": {
                "sofa": Value("string"),
                "uima.tcas.DocumentAnnotation:language": Sequence(Value("string")),
                "uima.tcas.DocumentAnnotation:begin": Sequence(Value("int32")),
                "uima.tcas.DocumentAnnotation:end": Sequence(Value("int32")),
                "cassis.Entity:entityType": Sequence(Value("string")),
                "cassis.Entity:begin": Sequence(Value("int32")),
                "cassis.Entity:end": Sequence(Value("int32")),
                "cassis.Label:label": Sequence(Value("string")),
                "cassis.Relation:source": Sequence(Value("int32")),
                "cassis.Relation:target": Sequence(Value("int32")),
            },
            "exception": Value("string"),
        }
    )
    expected_output_data = {"obj": [None], "exception": ["INVALID"]}


class TestCasParser_InvalidType(BaseDataProcessorTest):
    processor_type = CasParser
    processor_config = CasParser.Config(
        typesystem="./tests/artifacts/cas/typesystem.xml", types=["INVALID_TYPE"]
    )

    input_features = Features({"payload": Value("string")})
    expected_output_features_error = TypeError


class TestCasParser_MissingDependencyType(BaseDataProcessorTest):
    processor_type = CasParser
    processor_config = CasParser.Config(
        typesystem="./tests/artifacts/cas/typesystem.xml", types=["cassis.Relation"]
    )

    input_features = Features({"payload": Value("string")})
    expected_output_features_error = RuntimeError

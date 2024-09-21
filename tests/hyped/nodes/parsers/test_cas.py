from datasets import Features, Sequence, Value

from hyped.nodes.parsers.cas import CasParser
from tests.hyped.nodes.base import BaseDataProcessorTest

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
    # we are not testing for the expected output because the order
    # of the entities is not deterministic, but this is essentially
    # what it looks like, however we do test for the output in the
    # tests below and in the tests for the cas dataset which builds
    # upon the cas parser
    _expected_output_data = {
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
                "cassis.Relation:source": [0, 1],
                "cassis.Relation:target": [1, None],
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
                "cassis.Relation:source": [0, 1],
                "cassis.Relation:target": [1, None],
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

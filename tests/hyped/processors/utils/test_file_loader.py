from datasets import Features, Value

from hyped.processors.utils.file_loader import FileLoader
from tests.hyped.processors.base import BaseDataProcessorTest

with open("./tests/artifacts/lorem_ipsum.txt") as f:
    lorem_ipsum = f.read()


class TestFileLoader(BaseDataProcessorTest):
    processor_type = FileLoader
    processor_config = FileLoader.Config()

    input_features = Features({"file_path": Value("string")})
    input_data = {
        "file_path": [
            "./tests/artifacts/lorem_ipsum.txt",
            "./tests/artifacts/lorem_ipsum.txt",
        ]
    }

    expected_output_features = Features({"content": Value("string")})
    expected_output_data = {"content": [lorem_ipsum, lorem_ipsum]}

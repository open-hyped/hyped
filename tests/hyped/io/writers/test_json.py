import json
import os

from datasets import Dataset

from hyped.io.writers.json import JsonDatasetWriter

from .base import BaseTestDatasetWriter


class TestJsonDatasetWriter(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(10))})
    writer_type = JsonDatasetWriter

    def check(self) -> None:
        cls = type(self)

        assert "shard-0.json" in os.listdir(".")
        with open("shard-0.json", "r") as f:
            output_samples = list(map(json.loads, f.readlines()))

        for actual, expected in zip(output_samples, cls.dataset):
            assert actual == expected

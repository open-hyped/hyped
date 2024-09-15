import json
import os

from hyped.io.writers.json import JsonDatasetWriter

from .base import BaseTestDatasetWriter


class TestJsonDatasetWriter(BaseTestDatasetWriter):
    samples = [{"obj": i} for i in range(10)]
    writer_type = JsonDatasetWriter

    def check(self) -> None:
        cls = type(self)

        assert "shard-0.json" in os.listdir(".")
        with open("shard-0.json", "r") as f:
            output_samples = list(map(json.loads, f.readlines()))

        assert output_samples == cls.samples

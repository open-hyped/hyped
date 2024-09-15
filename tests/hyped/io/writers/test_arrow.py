from datasets import Dataset, load_from_disk

from hyped.io.writers.arrow import ArrowDatasetWriter

from .base import BaseTestDatasetWriter


class TestArrowDatasetWriter(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(10))})
    writer_type = ArrowDatasetWriter

    def check(self) -> None:
        # load dataset from disk
        actual_ds = load_from_disk(".")
        # compare to source dataset
        for actual, expected in zip(actual_ds, type(self).dataset):
            assert actual == expected

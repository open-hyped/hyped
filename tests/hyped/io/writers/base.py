import os
from abc import ABC, abstractmethod
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hyped.common._worker import WorkerInfo
from hyped.common.typing import Sample
from hyped.common.utils import chdir
from hyped.io.writers.base import BaseDatasetWriter


class BaseTestDatasetWriter(ABC):
    samples: list[Sample]
    writer_type: type[BaseDatasetWriter]
    writer_args: dict[str, Any] = {}

    @pytest.fixture
    def writer(self, tmp_path) -> BaseDatasetWriter:
        cls = type(self)
        # build keyword arguments
        kwargs = cls.writer_args.copy()
        kwargs["save_dir"] = os.path.join(tmp_path, "data")
        # create writer instance
        return cls.writer_type(**kwargs)

    @abstractmethod
    def check(self, save_dir: str) -> None:
        ...

    def test_case(self, writer):
        cls = type(self)

        # create output directory
        os.makedirs(writer.save_dir, exist_ok=False)
        # create mock worker info
        info = WorkerInfo(rank=0, num_workers=1, seed=42)

        with chdir(writer.save_dir):
            # patch get worker info in writer module
            module = cls.writer_type.__module__
            with patch(f"{module}.get_worker_info", MagicMock(return_value=info)):
                # initialize worker
                writer.initialize()
                # write all samples
                for sample in cls.samples:
                    writer.write_sample(sample)
                # finalize worker
                writer.finalize()

            # check output
            self.check()

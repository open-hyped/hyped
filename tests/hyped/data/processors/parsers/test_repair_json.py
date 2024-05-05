import pytest

from hyped.data.processors.parsers.repair_json import (
    RepairJsonParser,
    RepairJsonParserConfig,
)

from .test_json import TestJsonParser as _TestJsonParser


class TestRepairJsonParser(_TestJsonParser):
    @pytest.fixture
    def processor(self, scheme):
        return RepairJsonParser(
            RepairJsonParserConfig(json="json", scheme=scheme)
        )

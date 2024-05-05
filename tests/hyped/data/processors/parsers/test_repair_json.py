import json

import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.parsers.repair_json import (
    RepairJsonParser,
    RepairJsonParserConfig,
)
from tests.hyped.data.processors.base import BaseTestDataProcessor


class TestRepairJsonParser(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            ({"val": 0}, Features({"val": Value("int32")})),
            (
                {"z": [0, 1, 2]},
                Features({"z": Sequence(Value("int32"), length=3)}),
            ),
            ([1, 2, 3], Sequence(Value("int32"))),
            ([{"A": 1}, {"A": 0}], Sequence(Features({"A": Value("int32")}))),
        ]
    )
    def obj_and_scheme(self, request):
        return request.param

    @pytest.fixture
    def obj(self, obj_and_scheme):
        return obj_and_scheme[0]

    @pytest.fixture
    def scheme(self, obj_and_scheme):
        return obj_and_scheme[1]

    @pytest.fixture
    def in_features(self):
        return Features({"json": Value("string")})

    @pytest.fixture
    def processor(self, scheme):
        return RepairJsonParser(
            RepairJsonParserConfig(json="json", scheme=scheme)
        )

    @pytest.fixture
    def in_batch(self, obj):
        return {"json": [json.dumps(obj)]}

    @pytest.fixture
    def expected_out_features(self, scheme):
        return Features({"parsed": scheme})

    @pytest.fixture
    def expected_out_batch(self, obj):
        return {"parsed": [obj]}

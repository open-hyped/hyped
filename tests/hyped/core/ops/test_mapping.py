from unittest.mock import MagicMock, patch

import pytest

from hyped.core.features.features import build_feature_from_annotation
from hyped.core.ops.mapping import MappingGetItem, mapping_get_item
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Bool, Mapping


class CustomMapping(Mapping):
    key: Bool


class TestMappingGetItem(BaseDataProcessorTest):
    processor = MappingGetItem(key="key")
    input_features = {"mapping": CustomMapping}
    input_data = [
        {"mapping": {"key": True}},
        {"mapping": {"key": False}},
    ]
    expected_output_feature = Bool
    expected_output_data = [True, False]


@patch("hyped.core.ops.mapping.MappingGetItem", MagicMock())
def test_mapping_get_item() -> None:
    mapping = build_feature_from_annotation(CustomMapping)
    mapping_get_item(mapping, "key")
    with pytest.raises(KeyError):
        mapping_get_item(mapping, "INVALID_KEY")

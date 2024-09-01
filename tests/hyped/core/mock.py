from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

from datasets import Value

from hyped.core.nodes.aggregator import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
)
from hyped.core.nodes.augmenter import (
    BaseDataAugmenter,
    BaseDataAugmenterConfig,
)
from hyped.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.core.refs.inputs import (
    FeatureValidator,
    InputRefs,
    InputRefsValidator,
)
from hyped.core.refs.outputs import OutputFeature, OutputRefs
from hyped.core.refs.ref import FeatureRef


class MockInputRefs(InputRefs):
    a: Annotated[FeatureRef, FeatureValidator(lambda *args: None)]
    b: Annotated[FeatureRef, FeatureValidator(lambda *args: None)]


class MockOutputRefs(OutputRefs):
    y: Annotated[FeatureRef, OutputFeature(Value("int64"))]


class MockProcessorConfig(BaseDataProcessorConfig):
    i: int = 0


mock_input_refs_validator = InputRefsValidator(
    MockProcessorConfig(), MockInputRefs
)


class MockProcessor(
    BaseDataProcessor[MockProcessorConfig, MockInputRefs, MockOutputRefs]
):
    # mock process function
    process = MagicMock(return_value={"y": 0})


class MockAugmenterConfig(BaseDataAugmenterConfig):
    i: int = 0


class MockAugmenter(
    BaseDataAugmenter[MockAugmenterConfig, MockInputRefs, MockOutputRefs]
):
    # mock process function
    process = MagicMock(return_value=[{"y": 0}, {"y": 0}])


class MockAggregatorConfig(BaseDataAggregatorConfig):
    i: int = 0


class MockAggregator(
    BaseDataAggregator[MockAggregatorConfig, MockInputRefs, MockOutputRefs]
):
    # mock abstract functions
    initialize = MagicMock()
    extract = AsyncMock()
    update = AsyncMock(return_value=({"y": 0}, None))

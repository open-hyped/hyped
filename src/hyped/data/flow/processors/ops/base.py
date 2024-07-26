from abc import ABC
from typing import Annotated

from datasets import Value

from hyped.data.flow.core.nodes.processor import BaseDataProcessorConfig
from hyped.data.flow.core.refs.inputs import AnyFeatureType, InputRefs
from hyped.data.flow.core.refs.outputs import OutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class BaseUnaryOpConfig(BaseDataProcessorConfig, ABC):
    """Configuration class for unary operations."""


class UnaryOpInputRefs(InputRefs):
    """Defines input references for unary operations."""

    a: Annotated[FeatureRef, AnyFeatureType()]
    """The input feature. Can be any type."""


class BaseUnaryOpOutputRefs(OutputRefs):
    """Defines output references for unary operations."""

    result: Annotated[FeatureRef, OutputFeature(None)]
    """The result of the numeric unary operation. Placeholder type."""


class BaseBinaryOpConfig(BaseDataProcessorConfig):
    """Configuration class for binary operations."""


class BinaryOpInputRefs(InputRefs):
    """Defines input references for binary operations."""

    a: Annotated[FeatureRef, AnyFeatureType()]
    """The first input feature. Can be any type."""

    b: Annotated[FeatureRef, AnyFeatureType()]
    """The second input feature. Can be any type."""


class BaseBinaryOpOutputRefs(OutputRefs, ABC):
    """Defines output references for binary operations."""

    result: Annotated[FeatureRef, OutputFeature(None)]
    """The result of the binary operation. Placeholder type."""


class BooleanOutputRefs(OutputRefs):
    """Defines output references for operations with boolean result."""

    result: Annotated[FeatureRef, OutputFeature(Value("bool"))]
    """The result of the operation."""

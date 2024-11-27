from typing import TypeVar

import pyarrow.compute as pc

from hyped.core.nodes.base import RunContext, process_mode
from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from hyped.core.typing import Float, Int, UInt

ScalarType = TypeVar("ScalarType", Float, Int, UInt)


class AddConfig(BaseDataProcessorConfig):
    """Configuration for the Add processor."""


class Add(BaseDataProcessor[AddConfig]):
    """Element-wise addition.

    This processor takes two numeric inputs, :code:`x` and :code:`y`, and
    computes their sum.
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: ScalarType, y: ScalarType) -> ScalarType:
        """Perform element-wise addition of two numeric inputs.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (Numeric): The first numeric operand.
            y (Numeric): The second numeric operand.

        Returns:
            Numeric: The result of adding :code:`x` and :code:`y`.
        """
        return pc.add(x, y)


class SubtractConfig(BaseDataProcessorConfig):
    """Configuration for the Subtract processor."""


class Subtract(BaseDataProcessor[SubtractConfig]):
    """Element-wise subtraction.

    This processor takes two numeric inputs, :code:`x` and :code:`y`, and
    computes their difference (:code:`x - y`).
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: ScalarType, y: ScalarType) -> ScalarType:
        """Perform element-wise subtraction of two numeric inputs.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (ScalarType): The first numeric operand.
            y (ScalarType): The second numeric operand.

        Returns:
            ScalarType: The result of subtracting :code:`y` from :code:`x`.
        """
        return pc.subtract(x, y)


class MultiplyConfig(BaseDataProcessorConfig):
    """Configuration for the Multiply processor."""


class Multiply(BaseDataProcessor[MultiplyConfig]):
    """Element-wise multiplication.

    This processor takes two numeric inputs, :code:`x` and :code:`y`, and
    computes their product (:code:`x * y`).
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: ScalarType, y: ScalarType) -> ScalarType:
        """Perform element-wise multiplication of two numeric inputs.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (ScalarType): The first numeric operand.
            y (ScalarType): The second numeric operand.

        Returns:
            ScalarType: The product of :code:`x` and :code:`y`.
        """
        return pc.multiply(x, y)

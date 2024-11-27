from typing import TypeVar

import pyarrow.compute as pc

from hyped.core.nodes.base import RunContext, process_mode
from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from hyped.core.typing import Bool, Float, Int, UInt


class AbsConfig(BaseDataProcessorConfig):
    ...


class Abs(BaseDataProcessor[AbsConfig]):
    T = TypeVar("T", bound=Int | Float | UInt)

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: T) -> T:
        return pc.abs(x)


class NegateConfig(BaseDataProcessorConfig):
    ...


class Negate(BaseDataProcessor[NegateConfig]):
    T = TypeVar("T", bound=Int | Float)

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: T) -> T:
        return pc.negate(x)


class InvertConfig(BaseDataProcessorConfig):
    ...


class Invert(BaseDataProcessor[InvertConfig]):
    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: Bool) -> Bool:
        return pc.invert(x)

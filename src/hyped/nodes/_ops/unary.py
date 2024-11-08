from typing import TypeVar

import pyarrow as pa
import pyarrow.compute as pc

from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig, RunContext
from hyped.core.typing import Bool, Float, Int, UInt


class AbsConfig(BaseDataProcessorConfig):
    ...


class Abs(BaseDataProcessor[AbsConfig]):
    T = TypeVar("T", bound=Int | Float | UInt)

    async def batch_process(self, ctx: RunContext, x: pa.Array) -> pa.Array:
        return pc.abs(x)

    def process(self, ctx: RunContext, x: T) -> T:
        raise NotImplementedError()


class NegateConfig(BaseDataProcessorConfig):
    ...


class Negate(BaseDataProcessor[NegateConfig]):
    T = TypeVar("T", bound=Int | Float)

    async def batch_process(self, ctx: RunContext, x: pa.Array) -> pa.Array:
        return pc.negate(x)

    def process(self, ctx: RunContext, x: T) -> T:
        raise NotImplementedError()


class InvertConfig(BaseDataProcessorConfig):
    ...


class Invert(BaseDataProcessor[InvertConfig]):
    async def batch_process(self, ctx: RunContext, x: pa.Array) -> pa.Array:
        return pc.invert(x)

    def process(self, ctx: RunContext, x: Bool) -> Bool:
        raise NotImplementedError()

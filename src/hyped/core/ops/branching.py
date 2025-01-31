from typing import Annotated, TypeVar

from ..features.features import (
    BoolFeature,
    Feature,
)
from ..features.validators import MatchFeatures
from ..nodes.base import RunContext
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Bool


class IfElseConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`IfElse` processor."""


T = TypeVar("T", bound=Feature)
Feat = Annotated[T, MatchFeatures()]


class IfElse(BaseDataProcessor[IfElseConfig]):
    def process(self, ctx: RunContext, cond: Bool, a: Feat, b: Feat) -> Feat:
        return a if cond else b


BoolFeature.register_method("ifelse")(IfElse().call)

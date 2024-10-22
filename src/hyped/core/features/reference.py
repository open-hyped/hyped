from dataclasses import dataclass

from hyped.common.typing import NodeId
from hyped.core.features.feature_key import FeatureKey

from ..abstract import AbstractDataFlowGraph


@dataclass(eq=True, frozen=True)
class Reference:
    _key: FeatureKey
    _node_id: NodeId
    _graph: AbstractDataFlowGraph

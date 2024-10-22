from dataclasses import dataclass

from hyped.common.feature_key import FeatureKey
from hyped.common.typing import NodeId

from ..abstract import AbstractDataFlowGraph


@dataclass(eq=True, frozen=True)
class Reference:
    _key: FeatureKey
    _node_id: NodeId
    _graph: AbstractDataFlowGraph

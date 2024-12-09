"""This module provides the classes for referencing nodes within a data flow system."""
from __future__ import annotations

import typing
from dataclasses import dataclass

from ..abstract import AbstractDataFlowGraph

NodeId: typing.TypeAlias = str
"""Node ID type in the data flow graph.

Represents the identifier for a node within a data flow graph. This is typically a string that
uniquely identifies a node, allowing for the tracking and referencing of nodes within the graph
structure.
"""


class DummyDataFlowGraph(AbstractDataFlowGraph):
    """Dummy Data Flow Graph."""

    def __init__(self) -> None:
        """Initialize Dummy Data Flow Graph."""
        pass


@dataclass(eq=True, frozen=True)
class Reference:
    """Represents a reference to a specific feature, node, and graph in a data flow system.

    This class encapsulates a feature key, a node identifier, and a reference to a data flow graph.
    It is used to track and reference elements in the context of a larger data processing or
    computational graph.
    """

    _node_id: NodeId = "DummyNodeId"
    """The unique identifier for the node within the graph."""

    _graph: AbstractDataFlowGraph = DummyDataFlowGraph()
    """The data flow graph that contains the node."""

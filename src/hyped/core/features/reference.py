"""This module provides the classes for referencing nodes within a data flow system.

It includes abstractions for both concrete references, which point to real nodes in a data flow
graph, and forward references, which serve as placeholders or unresolved references to be
resolved later.
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..abstract import AbstractDataFlowGraph
from .dtypes import Type

NodeId: typing.TypeAlias = str
"""
Node ID type in the data flow graph.

Represents the identifier for a node within a data flow graph. This is typically a string that
uniquely identifies a node, allowing for the tracking and referencing of nodes within the graph
structure.
"""


@dataclass(eq=True, frozen=True)
class BaseReference(ABC):
    """Abstract base class for references to nodes within a data flow graph.

    Subclasses of :class:`BaseReference` represent different types of references that can
    be used to track or resolve elements in data flow graph. Each subclass must implement
    the :func:`get_dtype` method, which provides the data type associated with the reference.
    """

    @abstractmethod
    def get_dtype(self) -> None | Type:
        """Retrieve the data type associated with the reference.

        Returns:
            Type: The data type corresponding to the reference.
        """
        ...


@dataclass(eq=True, frozen=True)
class ForwardReference(BaseReference):
    """Represents a forward reference within a data flow graph.

    A forward reference serves as a placeholder for a node or feature that is not yet resolved.
    This is often used in cases where the node or feature will be defined or linked at a later
    stage in the graph construction process.
    """

    dtype: Type | None = None
    """The data type associated with the forward reference.

    Defaults to :code:`None` indicating that the data type of the referenced feature is not
    clear yet.
    """

    def get_dtype(self) -> None | Type:
        """Retrieve the data type associated with the forward reference.

        Returns:
            None | Type: The data type corresponding to the forward reference.
        """
        return self.dtype


@dataclass(eq=True, frozen=True)
class ConcreteReference(BaseReference):
    """Represents a reference to a specific node within a data flow graph.

    This class encapsulates the node identifier and the graph instance it belongs to,
    allowing for precise tracking and interaction with nodes in a data flow graph.
    """

    _node_id: NodeId
    """The unique identifier for the node within the graph."""

    _graph: AbstractDataFlowGraph
    """The data flow graph that contains the node."""

    def get_dtype(self) -> Type:
        """Retrieve the data type associated with the node referenced by this object.

        The data type is determined by querying the associated graph.

        Returns:
            Type: The data type corresponding to the node.
        """
        return self._graph.get_dtype_from_reference(self)

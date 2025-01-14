"""Abstract base classes used to avoid circular imports."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: not covered
    # avoid circular imports
    from .features.dtypes import DType
    from .features.reference import ConcreteReference, NodeId


class AbstractDataFlowGraph(ABC):
    """Abstract data flow graph base class."""

    @abstractmethod
    def get_output_dtype(self, node_id: "NodeId") -> "DType":
        """Get the output data type of a specific node."""
        ...


class AbstractDataFlowGraphBuilder(ABC):
    """Abstract data flow graph builder base class."""

    @abstractmethod
    def source(self, *args: Any, **kwargs: Any) -> "ConcreteReference":
        """Add a the source node to the graph."""
        ...

    @abstractmethod
    def const(self, *args: Any, **kwargs: Any) -> "ConcreteReference":
        """Adds a constant node to the data flow graph."""
        ...

    @abstractmethod
    def cast(self, *args: Any, **kwargs: Any) -> "ConcreteReference":
        """Add a cast node to the data flow graph."""
        ...

    @abstractmethod
    def collect(self, *args: Any, **kwargs: Any) -> "ConcreteReference":
        """Adds a collect node to the data flow graph."""
        ...

    @abstractmethod
    def compute(self, *args, **kwargs) -> "ConcreteReference":
        """Add a compute node to the data flow graph."""
        ...

    @abstractmethod
    def attach_graph_to_node(self, *args, **kwargs) -> AbstractDataFlowGraph:
        """Attach a data flow graph to a specific node within the current graph."""
        ...


class AbstractDataFlow(ABC):
    """Abstract data flow base class."""

    _graph: AbstractDataFlowGraph
    _builder: AbstractDataFlowGraphBuilder

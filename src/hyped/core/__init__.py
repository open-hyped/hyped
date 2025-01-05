"""The core package for defining and executing data flows as directed acyclic graphs (DAGs).

This package provides the necessary classes and methods to construct, manage, and execute
complex data processing workflows. It uses a graph-based approach where each node
represents a data processor, and edges represent the flow of data between processors.

Modules:
    - :class:`executor`: Manages the execution of the data flow graph.
    - :class:`flow`: Provides the high-level interface for defining data processing workflows.
    - :class:`graph`: Defines the structure of the data flow graph and its components.
    - :class:`optim`: Defines an Optimizer to optimize the graph of the data flow.
    - :class:`typing`: Defines core type aliases including those used to define node interfaces.
    - :class:`nodes`: Defines the base classes for nodes of the DAG.
    - :class:`features`: The feature system.
    - :class:`ops`: Implementation of core operations.
    - :class:`testing`: Defines base classes to implement node tests.

While these modules are crucial for processor development, they are not intended for
direct use by end users interacting with the high-level data flow interface.
"""

__all__ = [
    "DataFlow",
    "NodeProtocol",
    "RunContext",
    "process_mode",
    "BaseDataProcessor",
    "BaseDataProcessorConfig",
    "BaseDataAugmenter",
    "BaseDataAugmenterConfig",
    "BaseDataAggregator",
    "BaseDataAggregatorConfig",
    "ValidationSession",
    "ops",
]

# import operators module to register all operators
from . import ops
from .features.session import ValidationSession
from .flow import DataFlow
from .nodes.aggregator import BaseDataAggregator, BaseDataAggregatorConfig
from .nodes.augmenter import BaseDataAugmenter, BaseDataAugmenterConfig
from .nodes.base import NodeProtocol, RunContext, process_mode
from .nodes.processor import BaseDataProcessor, BaseDataProcessorConfig

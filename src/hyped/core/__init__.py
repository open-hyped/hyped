"""The core package for defining and executing data flows as directed acyclic graphs (DAGs).

This package provides the necessary classes and methods to construct, manage, and execute
complex data processing workflows. It uses a graph-based approach where each node
represents a data processor, and edges represent the flow of data between processors.

Modules:
    - :class:`executor`: Manages the execution of the data flow graph.
    - :class:`flow`: Provides the high-level interface for defining data processing workflows.
    - :class:`graph`: Defines the structure of the data flow graph and its components.
    - :class:`optim`: Defines an Optimizer to optimize the graph of the data flow.
    - :class:`nodes`: Defines the base classes for nodes of the DAG.
    - :class:`refs`: Defines reference objects as well as input/output helpers for processors.

While these modules are crucial for processor development, they are not intended for
direct use by end users interacting with the high-level data flow interface.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .flow import DataFlow
    from .nodes.aggregator import BaseDataAggregator, BaseDataAggregatorConfig
    from .nodes.augmenter import BaseDataAugmenter, BaseDataAugmenterConfig
    from .nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
    from .refs.inputs import (
        CheckFeatureEquals,
        CheckFeatureIsSequence,
        FeatureValidator,
        GlobalValidator,
        InputRefs,
    )
    from .refs.outputs import (
        ConditionalOutputFeature,
        LambdaOutputFeature,
        OutputFeature,
        OutputRefs,
    )
    from .refs.ref import NONE_REF, FeatureRef

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        "DataFlow": "hyped.core.flow",
        "BaseDataAggregator": "hyped.core.nodes.aggregator",
        "BaseDataAggregatorConfig": "hyped.core.nodes.aggregator",
        "BaseDataAugmenter": "hyped.core.nodes.augmenter",
        "BaseDataAugmenterConfig": "hyped.core.nodes.augmenter",
        "BaseDataProcessor": "hyped.core.nodes.processor",
        "BaseDataProcessorConfig": "hyped.core.nodes.processor",
        "InputRefs": "hyped.core.refs.inputs",
        "GlobalValidator": "hyped.core.refs.inputs",
        "FeatureValidator": "hyped.core.refs.inputs",
        "CheckFeatureEquals": "hyped.core.refs.inputs",
        "CheckFeatureIsSequence": "hyped.core.refs.inputs",
        "OutputRefs": "hyped.core.refs.outputs",
        "OutputFeature": "hyped.core.refs.outputs",
        "LambdaOutputFeature": "hyped.core.refs.outputs",
        "ConditionalOutputFeature": "hyped.core.refs.outputs",
        "FeatureRef": "hyped.core.refs.ref",
        "NONE_REF": "hyped.core.refs.ref",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_imports=_lazy_imports,
    )

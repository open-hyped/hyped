"""A comprehensive framework for complex data processing pipelines.

This package provides a comprehensive framework for constructing, managing,
and executing complex data processing pipelines. The framework is designed
to be modular and flexible, allowing users to define data flows, processors,
and augmenters to handle a wide variety of data processing tasks.

Example:
    Define a data flow for processing text data:

    .. code-block:: python

        import datasets
        from hyped import DataFlow
        from hyped.processors import TransformersTokenizer

        # create the data flow instance
        features = datasets.Features({"text": datasets.Value("string")})
        flow = DataFlow(features=features)

        # Define a processing step to tokenize text
        tokenizer = TokenizerProcessor(model_name="bert-base-uncased")
        tokenized_features = tokenizer.call(text=data_flow.src_features.text)

        # Apply the data flow to a dataset
        dataset = datasets.load(...)
        processed_dataset, _ = flow.apply(dataset)
"""

from typing import TYPE_CHECKING

__all__ = [
    # modules
    "aggregators",
    "augmenters",
    "io",
    "ops",
    "processors",
    # typing
    "Sample",
    "Batch",
    "Index",
    "IndexList",
    "Rank",
    "Aggregate",
    "TraceIndexList",
    "NodeId",
    "PartitionId",
    "DataFlowGraphAlias",
    "FeatureKeyAlias",
    "Pointer",
    # core
    "DataFlow",
    "BaseDataAggregator",
    "BaseDataAggregatorConfig",
    "BaseDataAugmenter",
    "BaseDataAugmenterConfig",
    "BaseDataProcessor",
    "BaseDataProcessorConfig",
    "InputRefs",
    "GlobalValidator",
    "FeatureValidator",
    "CheckFeatureEquals",
    "CheckFeatureIsSequence",
    "OutputRefs",
    "OutputFeature",
    "LambdaOutputFeature",
    "ConditionalOutputFeature",
    "FeatureRef",
    "NONE_REF",
    "IOContext",
]

if TYPE_CHECKING:
    from . import aggregators, augmenters, io, ops, processors
    from .common.typing import (
        Aggregate,
        Batch,
        DataFlowGraphAlias,
        FeatureKeyAlias,
        Index,
        IndexList,
        NodeId,
        PartitionId,
        Pointer,
        Rank,
        Sample,
        TraceIndexList,
    )
    from .core import (
        NONE_REF,
        BaseDataAggregator,
        BaseDataAggregatorConfig,
        BaseDataAugmenter,
        BaseDataAugmenterConfig,
        BaseDataProcessor,
        BaseDataProcessorConfig,
        CheckFeatureEquals,
        CheckFeatureIsSequence,
        ConditionalOutputFeature,
        DataFlow,
        FeatureRef,
        FeatureValidator,
        GlobalValidator,
        InputRefs,
        IOContext,
        LambdaOutputFeature,
        OutputFeature,
        OutputRefs,
    )

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_modules = {
        "aggregators": "hyped.aggregators",
        "augmenters": "hyped.augmenters",
        "io": "hyped.io",
        "ops": "hyped.ops",
        "processors": "hyped.processors",
    }

    _lazy_imports = {
        # typing
        "Sample": "hyped.common.typing",
        "Batch": "hyped.common.typing",
        "Index": "hyped.common.typing",
        "IndexList": "hyped.common.typing",
        "Rank": "hyped.common.typing",
        "Aggregate": "hyped.common.typing",
        "TraceIndexList": "hyped.common.typing",
        "NodeId": "hyped.common.typing",
        "PartitionId": "hyped.common.typing",
        "DataFlowGraphAlias": "hyped.common.typing",
        "FeatureKeyAlias": "hyped.common.typing",
        "Pointer": "hyped.common.typing",
        # core
        "DataFlow": "hyped.core",
        "BaseDataAggregator": "hyped.core",
        "BaseDataAggregatorConfig": "hyped.core",
        "BaseDataAugmenter": "hyped.core",
        "BaseDataAugmenterConfig": "hyped.core",
        "BaseDataProcessor": "hyped.core",
        "BaseDataProcessorConfig": "hyped.core",
        "InputRefs": "hyped.core",
        "GlobalValidator": "hyped.core",
        "FeatureValidator": "hyped.core",
        "CheckFeatureEquals": "hyped.core",
        "CheckFeatureIsSequence": "hyped.core",
        "OutputRefs": "hyped.core",
        "OutputFeature": "hyped.core",
        "LambdaOutputFeature": "hyped.core",
        "ConditionalOutputFeature": "hyped.core",
        "FeatureRef": "hyped.core",
        "NONE_REF": "hyped.core",
        "IOContext": "hyped.core",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_imports=_lazy_imports,
        lazy_modules=_lazy_modules,
    )

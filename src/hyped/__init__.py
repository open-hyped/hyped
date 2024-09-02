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


__all__ = [
    "DataFlow",
    "aggregators",
    "augmenters",
    "io",
    "ops",
    "processors",
]

from . import aggregators, augmenters, io, ops, processors
from .core.flow import DataFlow

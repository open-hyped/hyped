"""Module for various data processors in data processing workflows.

This module provides a comprehensive collection of data processors designed to
handle different data modalities and perform a wide range of data transformations.
Data processors are the fundamental building blocks in a data flow graph, acting as
nodes that implement specific, modular data transformations. These processors are
the core building blocks of complex data processing pipelines, facilitating efficient
and scalable data handling.

Key Features of Data Processors in This Module:
  - **Modularity**: Data processors are designed to perform distinct transformations,
    allowing for easy composition and reusability within various data pipelines.
  - **Flexibility**: Supports a wide array of data modalities and transformation
    operations, providing the necessary tools for diverse data processing tasks.
  - **Scalability**: Enables the construction of scalable data pipelines capable of
    handling large volumes of data, ensuring efficiency and performance.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api.openai_chat import OpenAIChatCompletion
    from .metrics.prfs import PrecisionRecallFScoreSupport
    from .parsers.json import JsonParser
    from .spans.bio_tags import BioTags
    from .spans.chr_to_tok import ChrToTokSpans
    from .spans.overlaps import ResolveOverlaps
    from .spans.utils import ResolveOverlapsStrategy
    from .templates.jinja2 import Jinja2
    from .tokenizers.transformers import TransformersTokenizer

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        # api
        "OpenAIChatCompletion": "hyped.processors.api.openai_chat",
        # metrics
        "PrecisionRecallFScoreSupport": "hyped.processors.metrics.prfs",
        # parsers
        "JsonParser": "hyped.processors.parsers.json",
        # spans
        "BioTags": "hyped.processors.spans.bio_tags",
        "ChrToTokSpans": "hyped.processors.spans.chr_to_tok",
        "ResolveOverlaps": "hyped.processors.spans.overlaps",
        "ResolveOverlapsStrategy": "hyped.processors.spans.utils",
        # templates
        "Jinja2": "hyped.processors.templates.jinja2",
        # tokenizers
        "TransformersTokenizer": "hyped.processors.tokenizers.transformers",
    }

    sys.modules[__name__] = LazyModule(
        __name__, __doc__, globals()["__file__"], __spec__, _lazy_imports
    )

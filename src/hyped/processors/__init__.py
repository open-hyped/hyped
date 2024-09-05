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

__all__ = [
    "OpenAIChatCompletion",
    "OpenAIChatCompletionConfig",
    "PrecisionRecallFScoreSupport",
    "PrecisionRecallFScoreSupportConfig",
    "JsonParser",
    "JsonParserConfig",
    "BioTags",
    "BioTagsConfig",
    "ChrToTokSpans",
    "ChrToTokSpansConfig",
    "ResolveOverlaps",
    "ResolveOverlapsConfig",
    "ResolveOverlapsStrategy",
    "ResolveOverlapsStrategyConfig",
    "Jinja2",
    "Jinja2Config",
    "TransformersTokenizer",
    "TransformersTokenizerConfig",
    "ToClassLabel",
    "ToClassLabelConfig",
]

if TYPE_CHECKING:
    from .api.openai_chat import OpenAIChatCompletion, OpenAIChatCompletionConfig
    from .metrics.prfs import PrecisionRecallFScoreSupport, PrecisionRecallFScoreSupportConfig
    from .parsers.json import JsonParser, JsonParserConfig
    from .spans.bio_tags import BioTags, BioTagsConfig
    from .spans.chr_to_tok import ChrToTokSpans, ChrToTokSpansConfig
    from .spans.overlaps import ResolveOverlaps, ResolveOverlapsConfig
    from .spans.utils import ResolveOverlapsStrategy
    from .templates.jinja2 import Jinja2, Jinja2Config
    from .tokenizers.transformers import TransformersTokenizer, TransformersTokenizerConfig
    from .utils.to_class_label import ToClassLabel, ToClassLabelConfig

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        # api
        "OpenAIChatCompletion": "hyped.processors.api.openai_chat",
        "OpenAIChatCompletionConfig": "hyped.processors.api.openai_chat",
        # metrics
        "PrecisionRecallFScoreSupport": "hyped.processors.metrics.prfs",
        "PrecisionRecallFScoreSupportConfig": "hyped.processors.metrics.prfs",
        # parsers
        "JsonParser": "hyped.processors.parsers.json",
        "JsonParserConfig": "hyped.processors.parsers.json",
        # spans
        "BioTags": "hyped.processors.spans.bio_tags",
        "BioTagsConfig": "hyped.processors.spans.bio_tags",
        "ChrToTokSpans": "hyped.processors.spans.chr_to_tok",
        "ChrToTokSpansConfig": "hyped.processors.spans.chr_to_tok",
        "ResolveOverlaps": "hyped.processors.spans.overlaps",
        "ResolveOverlapsConfig": "hyped.processors.spans.overlaps",
        "ResolveOverlapsStrategy": "hyped.processors.spans.utils",
        "ResolveOverlapsStrategyConfig": "hyped.processors.spans.utils",
        # templates
        "Jinja2": "hyped.processors.templates.jinja2",
        "Jinja2Config": "hyped.processors.templates.jinja2",
        # tokenizers
        "TransformersTokenizer": "hyped.processors.tokenizers.transformers",
        "TransformersTokenizerConfig": "hyped.processors.tokenizers.transformers",
        # utils
        "ToClassLabel": "hyped.processors.utils.to_class_label",
        "ToClassLabelConfig": "hyped.processors.utils.to_class_label",
    }

    sys.modules[__name__] = LazyModule(
        __name__, __doc__, globals()["__file__"], __spec__, _lazy_imports
    )

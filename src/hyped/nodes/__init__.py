"""Module for various nodes in data processing workflows.

This module provides a comprehensive collection of data processors designed to
handle different data modalities and perform a wide range of data transformations.
Data processors are the fundamental building blocks in a data flow graph, acting as
nodes that implement specific, modular data transformations. These processors are
the core building blocks of complex data processing pipelines, facilitating efficient
and scalable data handling.
"""

from typing import TYPE_CHECKING

__all__ = [
    "OpenAIChatCompletion",
    "OpenAIChatCompletionConfig",
    "PrecisionRecallFScoreSupport",
    "PrecisionRecallFScoreSupportConfig",
    "MultiLabelConfusionMatrix",
    "MultiLabelConfusionMatrixConfig",
    "JsonParser",
    "JsonParserConfig",
    "CasParser",
    "CasParserConfig",
    "BioTags",
    "BioTagsConfig",
    "ChrToTokSpans",
    "ChrToTokSpansConfig",
    "ResolveOverlaps",
    "ResolveOverlapsConfig",
    "ResolveOverlapsStrategy",
    "Jinja2",
    "Jinja2Config",
    "TransformersTokenizer",
    "TransformersTokenizerConfig",
    "ToClassLabel",
    "ToClassLabelConfig",
    "FileLoader",
    "FileLoaderConfig",
    "FileChunkLoader",
    "FileChunkLoaderConfig",
]

if TYPE_CHECKING:
    from .api.openai_chat import OpenAIChatCompletion, OpenAIChatCompletionConfig
    from .metrics.confusion import MultiLabelConfusionMatrix, MultiLabelConfusionMatrixConfig
    from .metrics.prfs import PrecisionRecallFScoreSupport, PrecisionRecallFScoreSupportConfig
    from .parsers.cas import CasParser, CasParserConfig
    from .parsers.json import JsonParser, JsonParserConfig
    from .spans.bio_tags import BioTags, BioTagsConfig
    from .spans.chr_to_tok import ChrToTokSpans, ChrToTokSpansConfig
    from .spans.overlaps import ResolveOverlaps, ResolveOverlapsConfig
    from .spans.utils import ResolveOverlapsStrategy
    from .tokenizers.transformers import TransformersTokenizer, TransformersTokenizerConfig
    from .utils.file_loader import FileLoader, FileLoaderConfig
    from .utils.jinja2 import Jinja2, Jinja2Config
    from .utils.to_class_label import ToClassLabel, ToClassLabelConfig

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        # api
        "OpenAIChatCompletion": "hyped.nodes.api.openai_chat",
        "OpenAIChatCompletionConfig": "hyped.nodes.api.openai_chat",
        # metrics
        "PrecisionRecallFScoreSupport": "hyped.nodes.metrics.prfs",
        "PrecisionRecallFScoreSupportConfig": "hyped.nodes.metrics.prfs",
        "MultiLabelConfusionMatrix": "hyped.nodes.metrics.confusion",
        "MultiLabelConfusionMatrixConfig": "hyped.nodes.metrics.confusion",
        # parsers
        "JsonParser": "hyped.nodes.parsers.json",
        "JsonParserConfig": "hyped.nodes.parsers.json",
        "CasParser": "hyped.nodes.parsers.cas",
        "CasParserConfig": "hyped.nodes.parsers.cas",
        # spans
        "BioTags": "hyped.nodes.spans.bio_tags",
        "BioTagsConfig": "hyped.nodes.spans.bio_tags",
        "ChrToTokSpans": "hyped.nodes.spans.chr_to_tok",
        "ChrToTokSpansConfig": "hyped.nodes.spans.chr_to_tok",
        "ResolveOverlaps": "hyped.nodes.spans.overlaps",
        "ResolveOverlapsConfig": "hyped.nodes.spans.overlaps",
        "ResolveOverlapsStrategy": "hyped.nodes.spans.utils",
        # tokenizers
        "TransformersTokenizer": "hyped.nodes.tokenizers.transformers",
        "TransformersTokenizerConfig": "hyped.nodes.tokenizers.transformers",
        # utils
        "Jinja2": "hyped.nodes.utils.jinja2",
        "Jinja2Config": "hyped.nodes.utils.jinja2",
        "ToClassLabel": "hyped.nodes.utils.to_class_label",
        "ToClassLabelConfig": "hyped.nodes.utils.to_class_label",
        "FileLoader": "hyped.nodes.utils.file_loader",
        "FileLoaderConfig": "hyped.nodes.utils.file_loader",
        "ChunkSequence": "hyped.nodes.utils.chunk",
        "ChunkSequenceConfig": "hyped.nodes.utils.chunk",
    }

    sys.modules[__name__] = LazyModule(
        __name__, __doc__, globals()["__file__"], __spec__, _lazy_imports
    )

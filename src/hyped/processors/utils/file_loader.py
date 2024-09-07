"""Data Processor for file loading.

This module provides functionality for reading the entire content of a file asynchronously.
"""


from typing import Annotated

import aiofiles
from datasets import Value

from hyped.common.typing import Index, Rank, Sample
from hyped.core.nodes.base import IOContext
from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from hyped.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.core.refs.outputs import OutputFeature, OutputRefs
from hyped.core.refs.ref import FeatureRef


class FileLoaderConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`FileLoader`."""

    encoding: str = "utf-8"
    """The encoding to use for reading files."""

    encoding_errors: None | str = None
    """The error handling scheme for encoding errors."""


class FileLoaderInputRefs(InputRefs):
    """Input references for the :class:`FileLoader`."""

    file_path: Annotated[FeatureRef, CheckFeatureEquals(Value("string"))]
    """Reference to the file path feature."""


class FileLoaderOutputRefs(OutputRefs):
    """Output references for the :class:`FileLoader`."""

    content: Annotated[FeatureRef, OutputFeature(Value("string"))]
    """Reference to the file content feature."""


class FileLoader(BaseDataProcessor[FileLoaderConfig, FileLoaderInputRefs, FileLoaderOutputRefs]):
    """Data processor that reads the entire content of a file."""

    async def process(self, inputs: Sample, index: Index, rank: Rank, io: IOContext) -> Sample:
        """Reads the entire content of the file specified in the input references.

        Args:
            inputs (Sample): The sample containing the file path.
            index (Index): The index of the sample in the batch.
            rank (Rank): The rank of the sample in the batch.
            io (IOContext): Context for input/output operations.

        Returns:
            Sample: A sample containing the entire content of the file.
        """

        async with aiofiles.open(
            inputs["file_path"],
            "r",
            encoding=self.config.encoding,
            errors=self.config.encoding_errors,
        ) as f:
            return Sample(content=await f.read())

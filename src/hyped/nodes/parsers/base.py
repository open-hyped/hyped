from abc import ABC, abstractmethod
from typing import TypeVar

from datasets import Value
from typing_extensions import Annotated

from hyped.common.typing import Index, Rank, Sample
from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig, IOContext
from hyped.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.core.refs.outputs import OutputFeature, OutputRefs
from hyped.core.refs.ref import FeatureRef


class ParserException(Exception):
    """Exception raised during parsing.

    This exception is raised when an error occurs during the parsing process
    and is typically converted into an error feature.
    """


class BaseParserConfig(BaseDataProcessorConfig):
    """Base configuration class for parsers.

    Inherit from this class to define configuration options specific to different
    types of parsers.
    """


class BaseParserInputRefs(InputRefs):
    """Base class for input references used by parsers."""

    payload: Annotated[FeatureRef, CheckFeatureEquals([Value("string"), Value("binary")])]
    """A reference to the input payload, which is expected to be a string or bytes."""


class BaseParserOutputRefs(OutputRefs):
    """Base class for output references used by parsers."""

    obj: Annotated[FeatureRef, OutputFeature(None)]
    """A reference to the parsed output object."""

    exception: Annotated[FeatureRef, OutputFeature(Value("string"))]
    """A reference to any exceptions encountered during the parsing process."""


C = TypeVar("C", bound=BaseParserConfig)
I = TypeVar("I", bound=BaseParserInputRefs)
O = TypeVar("O", bound=BaseParserOutputRefs)


class BaseParser(BaseDataProcessor[C, I, O], ABC):
    """Abstract base class for all parsers.

    This class serves as the base for all parsers, handling the general processing flow
    for parsing operations. Specific parsing behavior should be implemented in the
    :func:`parse` method, which must be defined in subclasses.
    """

    async def process(self, inputs: Sample, index: Index, rank: Rank, io: IOContext) -> Sample:
        """Process the input data by parsing it and returning the result.

        This method calls the :func:`parse` method to transform the input data into the desired
        output format. If an exception occurs during parsing, it captures the error and
        returns it as part of the result.

        Args:
            inputs (Sample): The input data to be parsed.
            index (Index): The index associated with the input sample.
            rank (Rank): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the data processor's execution.

        Returns:
            Sample: A sample containing the parsed output object or an exception message.
        """

        try:
            return Sample(obj=await self.parse(inputs, index, rank, io), exception=None)
        except ParserException as e:
            return Sample(obj=None, exception=str(e))

    @abstractmethod
    async def parse(self, inputs: Sample, index: Index, rank: Rank, io: IOContext) -> Sample:
        """Abstract method to be implemented by subclasses for parsing input data.

        This method should be overridden in derived classes to define the actual parsing logic.

        Args:
            inputs (Sample): The input data to be parsed.
            index (Index): The index associated with the input sample.
            rank (Rank): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the data processor's execution.

        Returns:
            Sample: The parsed object as a :class:`Sample`.

        Raises:
            ParserException: If the parsing operation fails.
        """
        ...

"""This module provides a JSON parser .

The processor is designed to parse JSON strings into structured feature types
using Pydantic for deserialization and validation.
"""
import json
from typing import Annotated

from datasets.features.features import Features, FeatureType, Sequence, Value
from pydantic import BeforeValidator, ConfigDict, PlainSerializer
from pydantic_core import ValidationError
from typing_extensions import Unpack

from hyped.common._pydantic import pydantic_model_from_features
from hyped.common.typing import Batch, Index, IndexList, Rank, Sample
from hyped.core.refs.inputs import CheckFeatureEquals
from hyped.core.refs.outputs import LambdaOutputFeature
from hyped.core.refs.ref import FeatureRef

from .base import (
    BaseParser,
    BaseParserConfig,
    BaseParserInputRefs,
    BaseParserOutputRefs,
    IOContext,
    ParserException,
)


class JsonParserInputRefs(BaseParserInputRefs):
    """Holds references to the input features for the JSON parser."""

    payload: Annotated[FeatureRef, CheckFeatureEquals([Value("string"), Value("binary")])]
    """The input payload expected to be a serialized json string."""


class JsonParserOutputRefs(BaseParserOutputRefs):
    """Holds references to the output features of the JSON parser."""

    obj: Annotated[FeatureRef, LambdaOutputFeature(lambda c, _: c.scheme)]
    """A reference to the parsed object."""


class JsonParserConfig(BaseParserConfig):
    """Configuration class for the :class:`JsonParser`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scheme: Annotated[
        Features | FeatureType,
        # custom serialization
        PlainSerializer(
            lambda f: json.dumps(Features({"feature": f}).to_dict()["feature"]),
            return_type=str,
            when_used="unless-none",
        ),
        # custom deserialization
        BeforeValidator(
            lambda v: (
                Features(v)
                if isinstance(v, dict)
                else Sequence(v)
                if isinstance(v, list)
                else v
                if isinstance(v, FeatureType)
                else Features.from_dict({"feature": json.loads(v)})["feature"]
            )
        ),
    ]
    """A feature defining the structure of the JSON input."""


class JsonParser(BaseParser[JsonParserConfig, JsonParserInputRefs, JsonParserOutputRefs]):
    """The JSON parser data processor.

    This processor is designed to take a JSON string as input and parse it into
    structured data based on a predefined schema. The schema can be defined using
    either a :code:`Features` object, a single :code:`FeatureType` instances.

    The parsed data is then validated and transformed into the desired format using
    Pydantic models, ensuring that the data conforms to the specified schema. This
    processor can handle batch processing, where multiple JSON strings are parsed and
    validated in a single operation, improving efficiency and performance.

    This processor can handle batch processing, where multiple JSON strings are parsed
    and validated in a single operation. If a batch cannot be parsed as a whole, it
    falls back to parsing each sample individually and identifying those with errors.
    """

    def __init__(self, config: None | JsonParserConfig = None, **kwargs) -> None:
        """Initialize the JsonParser with the given configuration.

        Args:
            config (JsonParserConfig): Configuration for the JSON parser.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(JsonParser, self).__init__(config, **kwargs)
        self._feature_model = pydantic_model_from_features(features={"parsed": self.config.scheme})
        self._batch_feature_model = pydantic_model_from_features(
            features={"parsed_batch": Sequence(self.config.scheme)}
        )

    async def batch_process(
        self, inputs: Batch, index: IndexList, rank: Rank, io: IOContext
    ) -> Batch:
        """Parse a batch of JSON strings.

        This method attempts to parse a batch of JSON strings as a single operation,
        improving performance. If a validation error occurs, it falls back to parsing
        each sample individually and identifies which samples raised errors.

        Args:
            inputs (Batch): The batch of input samples containing JSON strings to process.
            index (IndexList): The indices associated with the input samples.
            rank (Rank): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the data processor's execution.

        Returns:
            Batch: A batch containing the parsed objects or a list of exceptions for
            individual samples that failed validation.
        """

        batch_json_string = (
            (b'{"parsed_batch": [%b]}' % b",".join(inputs["payload"]))
            if io.inputs["payload"].dtype == "binary"
            else ('{"parsed_batch": [%s]}' % ",".join(inputs["payload"]))
        )

        try:
            # try to load the batch in one operation
            batch_model = self._batch_feature_model.model_validate_json(batch_json_string)
            return Batch(
                obj=batch_model.model_dump()["parsed_batch"], exception=[None] * len(index)
            )
        except ValidationError:
            # fallback to processing each sample individually and identify the
            # samples that raise the validation error
            return await super().batch_process(inputs, index, rank, io)

    async def parse(self, inputs: Sample, index: Index, rank: Rank, io: IOContext) -> Sample:
        """Parse a single JSON-string to a dictionary object.

        This method parses a JSON string contained within the input sample and validates it
        against a predefined model. If the configuration is set to catch validation errors,
        it will handle any validation exceptions and return a default model with an error message.
        Otherwise, it will directly parse and validate the JSON string.

        Args:
            inputs (Sample): The input sample containing the JSON string to be processed.
            index (Index): The index associated with the input sample.
            rank (Rank): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Sample: The parsed object.

        Raises:
            ParserException: If validation of the JSON string fails and
                :code:`config.catch_validation_errors` is False.
        """

        json_string = (
            (b'{"parsed": %b}' % inputs["payload"])
            if io.inputs["payload"].dtype == "binary"
            else ('{"parsed": %s}' % inputs["payload"])
        )

        try:
            # try to parse the json string
            model = self._feature_model.model_validate_json(json_string)
            return model.model_dump()["parsed"]
        except ValidationError as e:
            # raise a parser exception from the validation error
            raise ParserException(str(e)) from e

    def call(self, **kwargs: Unpack[JsonParserInputRefs]) -> JsonParserOutputRefs:
        """Add the JsonParser node to the data flow.

        This method processes the input references for the JsonParser operation, adds
        the corresponding node to the data flow, and returns the references to the
        output features generated by the processor.

        Args:
            json_str (FeatureRef): The reference to the json string to parse.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            JsonParserOutputRefs: The output references produced by the JsonParser processor.
        """
        return super(JsonParser, self).call(**kwargs)

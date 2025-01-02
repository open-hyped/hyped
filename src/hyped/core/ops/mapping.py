"""This module defines a collection of data processors that implement common mapping operations."""

from typing import Annotated

import pyarrow.compute as pc

from ..features.features import MappingFeature
from ..features.validators import FeatureResolver
from ..nodes.base import RunContext, process_mode
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Feature, Mapping


class MappingGetItemConfig(BaseDataProcessorConfig):
    """Configuration class for the :class:`MappingGetItem` processor."""

    key: str
    """The key used to retrieve the value from the mapping."""


class MappingGetItem(BaseDataProcessor[MappingGetItemConfig]):
    """Processor to retrieve a value from a mapping based on a specified key.

    This processor extracts the value associated with a given key from a mapping structure.
    The key is defined in the processor's configuration.
    """

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, mapping: Mapping
    ) -> Annotated[
        Feature, FeatureResolver(lambda config, inputs, _: inputs["mapping"].dtype[config.key])
    ]:
        """Retrieve a value from a mapping using a specified key.

        Args:
            ctx (RunContext): Context object containing runtime information.
            mapping (Mapping): Input mapping structure (e.g., dictionary-like or struct).

        Returns:
            Feature: The value associated with the specified key in the mapping.
        """
        return pc.struct_field(mapping, self.config.key)


@MappingFeature.register_method("__getitem__")
def mapping_get_item(mapping: Mapping, key: str) -> Feature:
    """Retrieve a value from a mapping by key using the :class:`MappingGetItem` processor.

    This method retrieves the value associated with a given key in the mapping. If the key
    is not present in the mapping's defined schema, an error is raised.

    Args:
        mapping (Mapping): Input mapping feature.
        key (str): The key used to retrieve the value from the mapping.

    Returns:
        Feature: The value associated with the specified key in the mapping.

    Raises:
        KeyError: If the specified key is not present in the mapping's schema.
    """
    if key not in mapping.dtype.keys():
        raise KeyError(
            f"Key '{key}' not found in the mapping. Available keys: {list(mapping.dtype.keys())}"
        )

    return MappingGetItem(key=key).call(mapping)

"""Abstract base classes used to avoid circular imports."""
from abc import ABC, abstractmethod
from typing import Any

import pydantic
from pydantic_core import core_schema


class _BaseAbstract(ABC):
    """Base Abstract class."""

    @abstractmethod
    def __init__(self) -> None:
        ...


class AbstractDataFlowGraph(_BaseAbstract):
    """Abstract data flow graph base class."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:  # pragma: not covered
        """Get the pydantic core schema.

        Args:
            cls (type): The class for which to generate the schema.
            source_type (Any): The source type that is being validated.
            handler (pydantic.GetCoreSchemaHandler): A handler function used to generate
                the schema for the class.

        Returns:
            core_schema.CoreSchema: The generated core schema.
        """
        return core_schema.is_instance_schema(AbstractDataFlowGraph)


class AbstractDataFlow(_BaseAbstract):
    """Abstract data flow base class."""

    _graph: AbstractDataFlowGraph

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:  # pragma: not covered
        """Get the pydantic core schema.

        Args:
            cls (type): The class for which to generate the schema.
            source_type (Any): The source type that is being validated.
            handler (pydantic.GetCoreSchemaHandler): A handler function used to generate
                the schema for the class.

        Returns:
            core_schema.CoreSchema: The generated core schema.
        """
        return core_schema.is_instance_schema(AbstractDataFlow)

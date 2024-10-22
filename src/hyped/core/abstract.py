"""Abstract base classes used to avoid circular imports"""
from abc import ABC, abstractmethod

from pydantic_core import core_schema


class _BaseAbstract(ABC):
    """Base Abstract class"""

    @abstractmethod
    def __init__(self) -> None:
        ...


class AbstractDataFlowGraph(_BaseAbstract):
    """Abstract data flow graph base class"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.is_instance_schema(AbstractDataFlowGraph)


class AbstractDataFlow(_BaseAbstract):
    """Abstract data flow base class"""

    _graph: AbstractDataFlowGraph

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.is_instance_schema(AbstractDataFlow)

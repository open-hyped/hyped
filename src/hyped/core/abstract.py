"""Abstract base classes used to avoid circular imports"""
from abc import ABC, abstractmethod


class _BaseAbstract(ABC):
    """Base Abstract class"""

    @abstractmethod
    def __init__(self) -> None:
        ...


class AbstractDataFlow(_BaseAbstract):
    """Abstract data flow base class"""


class AbstractDataFlowGraph(_BaseAbstract):
    """Abstract data flow graph base class"""

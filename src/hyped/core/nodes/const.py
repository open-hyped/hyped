"""This module defines the :class:`ConstNode`.

The :class:`ConstNode` class is a specialized node that holds a constant value defined in its
configuration. Unlike other nodes in the data flow graph, the :class:`ConstNode` does not process
input or dynamically compute outputs. Instead, it serves as a static reference for constant data.
"""

from typing import Annotated, Any

import pyarrow as pa
import pydantic

from ..features.dtypes import build_dtype_from_arrow_type, build_dtype_from_dict
from .base import BaseNode, BaseNodeConfig


class ConstNodeConfig(BaseNodeConfig):
    """Configuration for a constant node."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    value: Annotated[
        pa.Array,
        # serialize pyarrow array
        pydantic.PlainSerializer(
            lambda x: {
                "value": x.to_pylist()[0],
                "dtype": build_dtype_from_arrow_type(x.type).to_dict(),
            }
        ),
        # parse pyarrow array
        pydantic.BeforeValidator(
            lambda x: x
            if isinstance(x, pa.Array)
            else (pa.array([x["value"]], type=build_dtype_from_dict(x["dtype"]).arrow_type))
        ),
    ]
    """A PyArrow array representing the constant value.

    This field supports custom serialization and deserialization logic:
     - Serialization: Converts the array to a dictionary containing the
       constant (:code:`value`) and its data type (:code:`dtype`).
     - Deserialization: Parses input into a PyArrow array based on the
       provided value and data type.
    """


class ConstNode(BaseNode[ConstNodeConfig]):
    """A constant node in the processing pipeline."""

    def __str__(self) -> str:
        """Returns the string representation of the constant node.

        Returns:
            str: The class name of the node instance.
        """
        return str(self.config.value.to_pylist()[0])  # pragma: not covered

    @property
    def signature(self) -> Any:  # pragma: not covered
        """Raises an error, as signature is not supported for this node."""
        raise EnvironmentError("The `signature` property is not available for collect nodes.")

    def call(self, *args: Any, **kwargs: Any) -> Any:  # pragma: not covered
        """Raises an error, as direct calls are not supported for this node."""
        raise EnvironmentError("The `call` method is not available for collect nodes.")

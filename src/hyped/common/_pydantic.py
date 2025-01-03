"""Pydantic helper functionality."""
from typing import TypeVar

import pydantic
from pydantic.type_adapter import _type_has_config


class BaseModelWithArbitraryTypesAllowed(pydantic.BaseModel):
    """A Pydantic base model that allows arbitrary types for fields.

    This base model sets :code:`arbitrary_types_allowed` to :code:`True` in
    the configuration, enabling fields to accept custom or non-Pydantic types
    without explicit type validation.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


T = TypeVar("T")


def TypeAdapterWithArbitraryTypesAllowed(  # noqa: N802
    type: type[T],
    *,
    config: pydantic.ConfigDict | None = None,
    _parent_depth: int = 2,
    module: str | None = None,
) -> pydantic.TypeAdapter[T]:
    """Creates a Pydantic TypeAdapter with :code:`arbitrary_types_allowed` enabled.

    This function allows for creating :class:`TypeAdapters` that support arbitrary
    (non-Pydantic) types without explicit validation. If a configuration is provided,
    it will be updated to include :code:`arbitrary_types_allowed=True`. If no
    configuration is provided and the type lacks a configuration, a default configuration
    with `arbitrary_types_allowed=True` is used.

    Args:
        type (type[T]): The type associated with the :class:`TypeAdapter`.
        config (pydantic.ConfigDict | None): Configuration for the `TypeAdapter`, should be
            a dictionary conforming to :class:`pydantic.ConfigDict`.
        _parent_depth (int): The depth of the parent stack frame for module resolution.
            Defaults to 2.
        module (str | None):  The module that passes to plugin if provided.
            Defaults to :code:`None`.

    Returns:
        pydantic.TypeAdapter[T]: A Pydantic :class:`TypeAdapter` configured to allow arbitrary
        types.
    """
    if config is None and not _type_has_config(type):
        config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    elif config is not None:
        config["arbitrary_types_allowed"] = True

    return pydantic.TypeAdapter(type, config=config, _parent_depth=_parent_depth, module=module)

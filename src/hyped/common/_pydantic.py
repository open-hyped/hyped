"""Pydantic helper functionality."""
import pydantic


class BaseModelWithArbitraryTypesAllowed(pydantic.BaseModel):
    """A Pydantic base model that allows arbitrary types for fields.

    This base model sets `arbitrary_types_allowed` to `True` in the configuration, enabling fields
    to accept custom or non-Pydantic types without explicit type validation.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

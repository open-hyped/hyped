"""Base Configuration Functionality."""
from __future__ import annotations

import importlib
import json
from abc import ABC
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic._internal._model_construction import ModelMetaclass
from typing_extensions import dataclass_transform

from .auto import BaseAutoClass
from .registry import RegisterMeta, RegisterTypeMixin, Registrable
from .utils import solve_typevar


@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class RegisterModelMeta(RegisterMeta, ModelMetaclass):
    """metaclass for registrable pydantic model."""


class BaseConfig(Registrable, BaseModel, metaclass=RegisterModelMeta):
    """Base Configuration Pydantic Model."""

    # validate default argument
    model_config = ConfigDict(validate_default=True, extra="forbid")

    type_id: str | None = None
    type_hash__: str | None = Field(default=None, exclude=False)

    def model_post_init(self, __context: Any, /) -> None:
        """Post-initialization hook for setting type identifiers.

        This method is automatically called after the model initialization
        to set the `type_id` and `type_hash__` attributes based on the
        type of the instance.

        Args:
            __context (Any): Context object that may contain additional
                information for post-initialization. Currently not used.
        """
        object.__setattr__(self, "type_id", type(self).get_type_id())
        object.__setattr__(self, "type_hash__", type(self).get_type_hash())

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> BaseConfig:
        """Convert dict to configuration instance.

        Arguments:
            dct (dict[str, Any]): dictionary to be converted

        Returns:
            config (BaseConfig): the constructed configuration object
        """
        dct = dct.copy()
        # pop type hash and type identifier as they are meta
        # information and not actual fields needing to be set
        h = dct.pop("type_hash__", None)
        t = dct.pop("type_id", None)

        # make sure hashes match up
        if (h is not None) and (h != cls.get_type_hash()):
            raise ValueError("Type hash in dict doesn't match type hash of config")
        # make sure type identifiers match up
        if (t is not None) and (t != cls.get_type_id()):
            raise ValueError(
                "Type identifier in dict doesn't match type identifier "
                "of config: %s != %s" % (t, cls.get_type_id())
            )
        # instantiate config
        return cls(**dct)

    @classmethod
    def from_json(cls, serialized: str) -> BaseConfig:
        """Deserialize a json string into a config.

        Arguments:
            serialized (str): the serialized string in json format

        Returns:
            config (BaseConfig): the constructed configuration object
        """
        return cls.from_dict(json.loads(serialized))

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration object to dictionary."""
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        """Serialize config object into json format.

        Arguments:
            **kwargs: arguments forwarded to `json.dumps`

        Returns:
            serialized_config (str): the serialized configuration string
        """
        return json.dumps(self.to_dict(), **kwargs)


C = TypeVar("C", bound=BaseConfig)


class BaseAutoConfig(BaseAutoClass[C]):
    """Auto Configuration."""

    @classmethod
    def from_dict(cls, dct: dict[str, Any], trust_remote_code: bool = False) -> C:
        """Convert dict to configuration object of appropriate type.

        The type is inferred by the following prioritization:

        1. based on the `type_hash__` if present in the dictionary
        2. based on the type identifier `t` if present in the dictionary
        3. use the root class, i.e. the class on which the function is called

        Arguments:
            dct (dict[str, Any]): dictionary to be converted
            trust_remote_code (bool): Allow importing the module if it is
                not yet registered.

        Returns:
            config (BaseConfig): the constructed configuration object
        """
        if "type_hash__" in dct:
            # get type from registry
            h = dct.get("type_hash__")
            var = cls.type_registry().get_type_by_hash(h)

        elif "type_id" in dct:
            # get type from type id
            t = dct.get("type_id")
            try:
                var = cls.type_registry().get_type_by_t(t)

            except ValueError:
                if trust_remote_code:
                    try:
                        module_path, _ = t.rsplit(".", 1)
                        importlib.import_module(module_path)  # TODO: Is this unsafe?
                    except (ImportError, AttributeError) as e:
                        raise ImportError(f"Could not import '{t}': {e}") from e

                    var = cls.type_registry().get_type_by_t(t)

                else:
                    raise TypeError(
                        f"Config of type '{t}' is not registered! To enable automatic "
                        "import of the module set `trust_remote_code=True`."
                    ) from None

        else:
            raise TypeError("Unable to resolve type of config: `%s`" % str(dct))

        # create instance
        return var.from_dict(dct)

    @classmethod
    def from_json(cls, serialized: str, trust_remote_code: bool = False) -> C:
        """Load configuration from json.

        Deserialize a json string into a configuration object of
        appropriate type.

        The type is inferred by the following prioritization:

        1. based on the `type_hash__` if present in the json string
        2. based on the type identifier `t` if present in the json string
        3. use the root class, i.e. the class on which the function is called

        Arguments:
            serialized (str): the serialized string in json format
            trust_remote_code (bool): Allow importing the module if it is
                not yet registered.

        Returns:
            config (BaseConfig): the constructed configuration object
        """
        return cls.from_dict(json.loads(serialized), trust_remote_code=trust_remote_code)

    @classmethod
    def from_file(cls, path: str | Path, trust_remote_code: bool = False) -> C:
        """Load configuration from a file.

        Reads the contents of a file and deserializes them into a configuration
        object of the appropriate type.

        Args:
            path (str | Path): The path to the file containing the configuration
                data in JSON format.
            trust_remote_code (bool): Allow importing the module if it is
                not yet registered.

        Returns:
            C: The constructed configuration object based on the file content.
        """
        return cls.from_json(Path(path).read_text(), trust_remote_code=trust_remote_code)


class AutoConfig(BaseAutoConfig[BaseConfig]):
    """Automatic configuration class derived from BaseConfig.

    This class extends the BaseAutoConfig specifically for configurations
    that adhere to the BaseConfig structure, providing automated handling
    and instantiation capabilities.
    """

    ...


U = TypeVar("U", bound=BaseConfig)


class BaseConfigurable(Generic[U], RegisterTypeMixin, ABC):
    """Base class for configurable types."""

    CONFIG_TYPE: None | type[U] = None

    def __init__(self, config: None | U = None, **kwargs) -> None:
        """Initialize the configurable.

        Args:
            config (C, optional): The configuration. If not provided, a configuration
                is created based on the given keyword arguments.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        if config is None:
            config = self.config_type()(**kwargs)
        elif len(kwargs) is not None:
            config = config.model_copy(update=kwargs)

        if not isinstance(config, self.config_type()):
            raise TypeError()

        self._config = config

    @property
    def config(self) -> U:
        """Retrieves the configuration of the object.

        Returns:
            C: The configuration object.
        """
        return self._config

    @classmethod
    def from_config(cls, config: U) -> BaseConfigurable:
        """Abstract construction method, must be implemented by sub-types.

        Arguments:
            config (T): configuration to construct the instance from

        Returns:
            inst (Configurable): instance
        """
        return cls(config)

    @classmethod
    def generic_config_type(cls) -> type[U] | None:
        """Config Type specified by generic type var `U`.

        Get the generic configuration type of the configurable specified
        by the type variable `U`.
        """
        # get config class
        t = solve_typevar(cls, U)
        # check type
        if (t is not None) and not issubclass(t, BaseConfig):
            raise TypeError(
                "Configurable config type `%s` doesn't inherit from `%s`"
                % (str(t), str(BaseConfig))
            )
        return t

    @classmethod
    def config_type(cls) -> type[U] | None:
        """Get the (final) configuration type of the configurable.

        The final configuration type is specified by the `CONFIG_TYPE`
        class attribute. Falls back to the generic config type if the
        class attribute is not specified. Also checks that the concrete
        configuration type is valid, i.e. inherits the generic configuration
        type.
        """
        generic_t = cls.generic_config_type()
        # concrete config type must inherit generic config type
        if (cls.CONFIG_TYPE is not None) and not issubclass(cls.CONFIG_TYPE, generic_t):
            raise TypeError(
                "Concrete config type `%s` specified by `CONFIG_TYPE` must "
                "inherit from generic config type `%s`" % (cls.CONFIG_TYPE, generic_t)
            )
        # return final config type and fallback to generic
        # type if not specified
        return cls.CONFIG_TYPE or generic_t

    @classmethod
    def get_type_id(cls) -> str | None:
        """Type Identifier.

        Type identifier used in type registry. Identifier is build
        from configuration type identifier by appending `.impl`.
        """
        # no concrete config type
        if cls.config_type() is None:
            return None
        # specify registry type identifier based on config type identifier
        return "%s.impl" % cls.config_type().get_type_id()


V = TypeVar("V", bound=BaseConfigurable)


class BaseAutoConfigurable(BaseAutoClass[V]):
    """Auto Class for configurable types."""

    @classmethod
    def from_config(cls, config: BaseConfig) -> V:
        """Create instance from given config.

        Arguments:
            config (BaseConfig): configuration

        Returns:
            V: instance created from config
        """
        # build type identifier of configurable corresponding
        # to the config
        t = "%s.impl" % config.get_type_id()
        var = cls.type_registry().get_type_by_t(t)
        # create instance
        return var.from_config(config)

    @classmethod
    def from_config_dict(cls, config: dict, trust_remote_code: bool = False) -> V:
        """Create instance from given config dict.

        Arguments:
            config (dict): configuration dict
            trust_remote_code (bool): Allow importing the module if it is
                not yet registered.

        Returns:
            V: instance created from config dict
        """
        return cls.from_config(AutoConfig.from_dict(config, trust_remote_code=trust_remote_code))

    @classmethod
    def from_config_file(cls, path: str | Path, trust_remote_code: bool = False) -> V:
        """Create an instance from a configuration file.

        Reads a configuration file and uses its content to instantiate an
        object of the appropriate type.

        Args:
            path (str | Path): The path to the configuration file.
            trust_remote_code (bool): Allow importing the module if it is
                not yet registered.

        Returns:
            V: An instance created using the configuration data from the file.
        """
        return cls.from_config(AutoConfig.from_file(path, trust_remote_code=trust_remote_code))


class AutoConfigurable(BaseAutoConfigurable[BaseConfigurable]):
    """Automatic configurable class derived from BaseConfigurable.

    This class extends the BaseAutoConfigurable specifically for types that
    adhere to the BaseConfigurable structure, providing automated management
    and instantiation capabilities for configurable objects.
    """

    ...

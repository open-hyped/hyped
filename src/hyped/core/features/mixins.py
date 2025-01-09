"""DType Mixins."""

import inspect
import logging
from typing import Any, Callable, ClassVar, TypeVar

logger = logging.getLogger(__name__)


class MethodRegistryMixin:
    """A mixin class to provide method registration functionality."""

    _methods: ClassVar[dict[tuple[str, str], Callable]] = {}
    """Class-level registry of methods for dynamically defined behavior.

    The keys are tuples of the form `(class_qualname, method_name)`, and the values
    are callable objects representing the registered methods.
    """

    Fn = TypeVar("Fn", bound=Callable)

    @classmethod
    def register_method(cls, name: str, force: bool = False) -> Callable[[Fn], Fn]:
        """Registers a method for the class.

        This decorator allows external sub-modules to register methods
        for the class, decoupling method implementations from the class
        definition itself.

        Args:
            name (str): The name of the method to register.
            force (bool): If :code:`True`, allows overriding an existing method.
                If :code:`False` and the method is already registered, raises a
                :class:`RuntimeError`. Defaults to :code:`False`.

        Returns:
            Callable[[Fn], Fn]: A decorator function that registers the method
            and returns it unmodified.

        Raises:
            RuntimeError: If a method with the same name is already registered and
                :code:`force` is set to :code:`False`.
        """

        def decorator(fn):
            key = (cls.__qualname__, name)
            if key in cls._methods and not force:
                raise RuntimeError(
                    f"Method '{name}' is already registered for class '{cls.__qualname__}'. "
                    "Use `force=True` to override."
                )
            cls._methods[key] = fn
            logger.debug(f"Registerd method {name} to {cls.__qualname__}")
            return fn

        return decorator

    @classmethod
    def get_method(cls, name: str) -> Callable:
        """Retrieve a registered method by name for the current class.

        This class method accesses the :code:`_methods` registry using a key
        composed of the class's qualified name and the method's name. If the
        specified method is not found in the registry, it raises a
        :class:`NotImplementedError`.

        Args:
            name (str): The name of the method to retrieve.

        Returns:
            Callable: The method registered under the specified name.

        Raises:
            NotImplementedError: If no method with the given name is registered
                for the current class.
        """
        for base in inspect.getmro(cls):
            key = (base.__qualname__, name)
            if issubclass(base, MethodRegistryMixin) and key in MethodRegistryMixin._methods:
                return MethodRegistryMixin._methods[key]
        else:
            raise NotImplementedError(
                f"Method '{name}' not registered for class '{cls.__qualname__}'."
            )

    def execute_method(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Executes a registered method by its name with the provided arguments.

        This method is a convenience function that combines the retrieval of a
        registered method using :func:`get_method` and its subsequent execution. It
        ensures that the method is executed in the context of the current instance.

        Args:
            name (str): The name of the method to execute.
            *args (Any): Positional arguments to pass to the method.
            **kwargs (Any): Keyword arguments to pass to the method.

        Returns:
            Any: The result of executing the registered method.

        Raises:
            NotImplementedError: If no method with the specified name is registered
                for the class of the current object.
        """
        return type(self).get_method(name)(self, *args, **kwargs)

"""Module for implementing common arithmetic and binary operations via mixins.

This module provides mixin classes that implement common arithmetic and binary operations
such as addition, subtraction, multiplication, division, and others. These mixins enable
objects to support operations like '+', '+=', '-', '-=', '*', '*=', '/', '/=', and their
reverse counterparts (e.g., '__radd__', '__rsub__', etc.) through method delegation.

Each mixin uses the :code:`execute_method` function to dynamically dispatch the operation to
the appropriate method, falling back to the base method if the specific operation is not
implemented. These mixins allow for customizable behavior and ensure that operations like
modulus, floor division, and negation can be consistently implemented.
"""


from typing import Any, Callable, ClassVar, TypeVar


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
            return fn

        return decorator

    def get_method(self, name: str) -> Callable:
        """Retrieves a method registered to the current class by its name.

        This method looks up the :code:`_methods` registry for the calling class
        (:code:`type(self)`) using a tuple key that includes the class's qualified
        name and the method's name. If the method is not found, it raises a
        :class:`NotImplementedError`.

        Args:
            name (str): The name of the method to retrieve.

        Returns:
            Callable: The registered method as a callable object.

        Raises:
            NotImplementedError: If no method with the specified name is registered
                for the class of the current object.
        """
        key = (type(self).__qualname__, name)
        if key not in self._methods:
            raise NotImplementedError(
                f"Method '{name}' not registered for class '{type(self).__qualname__}'."
            )
        return self._methods[key]

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
        return self.get_method(name)(self, *args, **kwargs)

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

from typing_extensions import Self


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


class NegMixin:
    """Mixin for implementing negation ('-')."""

    def __neg__(self: MethodRegistryMixin) -> Self:
        """Implements negation ('-').

        This method returns the negated value by calling the :code:`__neg__` method,
        enabling the use of the unary minus operator ('-').
        """
        return self.execute_method("__neg__")


class AbsMixin:
    """Mixin for implementing absolute value ('abs')."""

    def __abs__(self: MethodRegistryMixin) -> Self:
        """Implements absolute value ('abs').

        This method returns the absolute value by calling the :code:`__abs__` method,
        enabling the use of the abs() function or the 'abs' operator.
        """
        return self.execute_method("__abs__")


class AddMixin:
    """Mixin for implementing addition ('+', '+=') and reverse addition ('+')."""

    def __add__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements addition ('+').

        This method performs addition using the :code:`__add__` method, enabling
        the use of the binary addition operator ('+').
        """
        return self.execute_method("__add__", other)

    def __iadd__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements in-place addition ('+=').

        This method performs in-place addition using the :code:`__iadd__` method or
        falls back to using :code:`__add__` if :code:`__iadd__` is not implemented.
        """
        try:
            return self.execute_method("__iadd__", other)
        except NotImplementedError:
            return self.execute_method("__add__", other)

    def __radd__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements reverse addition ('+').

        This method attempts to execute the reverse addition method (:code:`__radd__`).
        If not implemented, it falls back to using :code:`__add__`.
        """
        try:
            return self.execute_method("__radd__", other)
        except NotImplementedError:
            return self.get_method("__add__")(other, self)


class SubMixin:
    """Mixin for implementing subtraction ('-', '-=') and reverse subtraction ('-')."""

    def __sub__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements subtraction ('-').

        This method performs subtraction using the :code:`__sub__` method, enabling
        the use of the binary subtraction operator ('-').
        """
        return self.execute_method("__sub__", other)

    def __isub__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements in-place subtraction ('-=').

        This method performs in-place subtraction using the :code:`__isub__` method or
        falls back to using :code:`__sub__` if :code:`__isub__` is not implemented.
        """
        try:
            return self.execute_method("__isub__", other)
        except NotImplementedError:
            return self.execute_method("__sub__", other)

    def __rsub__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements reverse subtraction ('-').

        This method attempts to execute the reverse subtraction method (:code:`__rsub__`).
        If not implemented, it falls back to using :code:`__sub__`.
        """
        try:
            return self.execute_method("__rsub__", other)
        except NotImplementedError:
            return self.get_method("__sub__")(other, self)


class MulMixin:
    """Mixin for implementing multiplication ('*', '*=') and reverse multiplication ('*')."""

    def __mul__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements multiplication ('*').

        This method performs multiplication using the :code:`__mul__` method, enabling
        the use of the binary multiplication operator ('*').
        """
        return self.execute_method("__mul__", other)

    def __imul__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements in-place multiplication ('*=').

        This method performs in-place multiplication using the :code:`__imul__` method or
        falls back to using :code:`__mul__` if :code:`__imul__` is not implemented.
        """
        try:
            return self.execute_method("__imul__", other)
        except NotImplementedError:
            return self.execute_method("__mul__", other)

    def __rmul__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements reverse multiplication ('*').

        This method attempts to execute the reverse multiplication method (:code:`__rmul__`).
        If not implemented, it falls back to using :code:`__mul__`.
        """
        try:
            return self.execute_method("__rmul__", other)
        except NotImplementedError:
            return self.get_method("__mul__")(other, self)


class DivMixin:
    """Mixin for implementing division ('/', '/=') and reverse division ('/')."""

    def __truediv__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements division ('/').

        This method performs division using the :code:`__truediv__` method, enabling
        the use of the division operator ('/').
        """
        return self.execute_method("__truediv__", other)

    def __itruediv__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements in-place division ('/=').

        This method performs in-place division using the :code:`__itruediv__` method or
        falls back to using :code:`__truediv__` if :code:`__itruediv__` is not implemented.
        """
        try:
            return self.execute_method("__itruediv__", other)
        except NotImplementedError:
            return self.execute_method("__truediv__", other)

    def __rtruediv__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements reverse division ('/').

        This method attempts to execute the reverse division method (:code:`__rtruediv__`).
        If not implemented, it falls back to using :code:`__truediv__`.
        """
        try:
            return self.execute_method("__rtruediv__", other)
        except NotImplementedError:
            return self.get_method("__truediv__")(other, self)


class FloorDivMixin:
    """Mixin for implementing floor division ('//', '//=') and reverse floor division ('//')."""

    def __floordiv__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements floor division ('//').

        This method performs floor division using the :code:`__floordiv__` method, enabling
        the use of the floor division operator ('//').
        """
        return self.execute_method("__floordiv__", other)

    def __ifloordiv__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements in-place floor division ('//=').

        This method performs in-place floor division using the :code:`__ifloordiv__` method or
        falls back to using :code:`__floordiv__` if :code:`__ifloordiv__` is not implemented.
        """
        try:
            return self.execute_method("__ifloordiv__", other)
        except NotImplementedError:
            return self.execute_method("__floordiv__", other)

    def __rfloordiv__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements reverse floor division ('//').

        This method attempts to execute the reverse floor division method (:code:`__rfloordiv__`).
        If not implemented, it falls back to using :code:`__floordiv__`.
        """
        try:
            return self.execute_method("__rfloordiv__", other)
        except NotImplementedError:
            return self.get_method("__floordiv__")(other, self)


class ModMixin:
    """Mixin for implementing modulo ('%', '%=') and right modulo ('%')."""

    def __mod__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements modulo ('%').

        This method performs modulo using the :code:`__mod__` method, enabling
        the use of the modulus operator ('%').
        """
        return self.execute_method("__mod__", other)

    def __imod__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements in-place modulo ('%=').

        This method performs in-place modulo using the :code:`__imod__` method or
        falls back to using :code:`__mod__` if :code:`__imod__` is not implemented.
        """
        try:
            return self.execute_method("__imod__", other)
        except NotImplementedError:
            return self.execute_method("__mod__", other)

    def __rmod__(self: MethodRegistryMixin, other: Any) -> Self:
        """Implements right modulo ('%').

        This method attempts to execute the right modulo method (:code:`__rmod__`).
        If not implemented, it falls back to using :code:`__mod__`.
        """
        try:
            return self.execute_method("__rmod__", other)
        except NotImplementedError:
            return self.get_method("__mod__")(other, self)

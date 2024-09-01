"""This module defines the :class:`LazyModule` class for lazy loading of module attributes.

The :class:`LazyModule` class provides a mechanism for lazy loading attributes of a
module. Lazy loading is a design pattern that defers the initialization of an object
until the point at which it is needed, thereby improving startup performance and
reducing memory consumption.

The :class:`LazyModule` class extends Python's built-in :class:`ModuleType` and
overrides methods to handle the deferred import of attributes. This is particularly
useful for modules that contain many submodules or attributes, but where not all of
them are needed immediately.

Key Features:
- Lazy Loading: Attributes are loaded only when accessed, rather than at module 
  import time.
- Dynamic Import: Uses :code:`importlib` to dynamically import modules when an 
  attribute is accessed.
- Custom `__dir__`: Includes lazy-loaded attributes in the list of available 
  attributes.

Usage Example:
    ```python
    import sys
    from hyped.common.lazy_module import LazyModule

    lazy_imports = {
        'foo': 'module_foo',
        'bar': 'module_bar'
    }

    lazy_module = LazyModule(
        'lazy_module',
        'A lazy-loaded module',
        '/path/to/module.py',
        'spec',
        lazy_imports
    )

    # Accessing 'foo' will trigger the import of 'module_foo'
    foo_value = lazy_module.foo
    ```

This module is particularly useful for applications where modules have many 
dependencies, but not all are needed right away. By using :class:`LazyModule`,
you can improve performance and resource management.
"""


import importlib
import os
from types import ModuleType
from typing import Any


class LazyModule(ModuleType):
    """A module type that supports lazy loading of its attributes.

    This class allows for lazy loading of submodules or attributes by deferring
    their import until they are actually accessed. This can help improve startup
    times and reduce memory usage in cases where not all components of a module
    are needed immediately.
    """

    def __init__(
        self,
        name: str,
        doc: str,
        module_file: str,
        module_spec: str,
        lazy_imports: dict[str, str] = {},
        lazy_modules: dict[str, str] = {},
    ) -> None:
        """Initialize a LazyModule instance.

        Args:
            name (str): The name of the module.
            doc (str): The module's docstring.
            module_file (str): The path to the module file.
            module_spec (str): The module specification.
            lazy_imports (dict[str, str]): A dictionary mapping attribute names
                to module paths for lazy imports.
            lazy_modules (dict[str, str]): A dictionary mapping sub-module names
                to module paths for lazy imports.
        """
        super(LazyModule, self).__init__(name, doc=doc)

        self._lazy_imports = lazy_imports
        self._lazy_modules = lazy_modules
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]

        shared_keys = set(lazy_imports.keys()) & set(lazy_modules.keys())
        assert (
            len(shared_keys) == 0
        ), f"Overlapping keys in 'lazy_imports' and 'lazy_modules': {shared_keys}"

    @property
    def __all__(self) -> list[str]:
        """Return a list of attribute names that are available for lazy loading.

        Returns:
            list[str]: A list of attribute names defined in the lazy imports.
        """
        return list(self._lazy_imports.keys()) + list(
            self._lazy_modules.keys()
        )

    def __dir__(self) -> list[str]:
        """Return a list of attributes available in this module.

        This includes both the attributes defined in the base class and those
        specified in the lazy imports.

        Returns:
            list[str]: A list of attribute names.
        """
        out = set(super(LazyModule, self).__dir__())
        out.update(set(self.__all__))
        return list(out)

    def __getattr__(self, name: str) -> Any:
        """Handle attribute access for lazy-loaded attributes.

        This method is called when an attribute is accessed that has not yet been
        loaded. It imports the necessary module and retrieves the attribute from it.

        Args:
            name (str): The name of the attribute to access.

        Returns:
            Any: The attribute from the imported module.

        Raises:
            AttributeError: If the attribute is not found in the lazy imports.
        """
        if name in self._lazy_imports:
            module = self._lazy_imports[name]
            module = importlib.import_module(module, self.__name__)
            return getattr(module, name)

        if name in self._lazy_modules:
            module = self._lazy_modules[name]
            return importlib.import_module(module, self.__name__)

        raise AttributeError(
            f"module '{self.__name__}' has no attribute '{name}'"
        )

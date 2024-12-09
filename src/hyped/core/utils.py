"""Utility functions and type mappings used throughout the core module.

This module provides various helper functions for working with data types,
feature mappings, and nested structures in the core module.
"""
from __future__ import annotations

from typing import Any, Callable, TypeAlias, TypeVar

T = TypeVar("T")
NestedType: TypeAlias = dict[str, "NestedType"] | list["NestedType"] | tuple["NestedType"] | T


def map_recursive(
    fn: Callable[[NestedType[Any]], None | NestedType[Any]],
    obj: NestedType[Any],
    path: tuple[str | int] = (),
) -> NestedType[Any]:
    """Apply a function recursively on a nested object.

    Arguments:
        fn (Callable[[NestedType[Any]], None | NestedType[Any]]): The function to apply.
        obj (NestedType[Any]): The nested object to apply the function to.
        path (tuple[str | int], optional): The current path in the object
            (default is an empty tuple).

    Returns:
        NestedType[Any]: The object after the function has been applied recursively.
    """
    obj = (
        {key: map_recursive(fn, val, path + (key,)) for key, val in obj.items()}
        if isinstance(obj, dict)
        else type(obj)(map_recursive(fn, val, path + (i,)) for i, val in enumerate(obj))
        if isinstance(obj, (list, tuple))
        else obj
    )
    out_obj = fn(path, obj)
    return out_obj if out_obj is not None else obj

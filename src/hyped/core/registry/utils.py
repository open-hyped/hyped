"""Utility functions for type registry."""
from functools import cache
from typing import TypeVar, get_args, get_origin


@cache
def build_typevar_mapping(t: type) -> dict[TypeVar, TypeVar | type | None]:
    """Build a mapping from TypeVars to their concrete types within a given type.

    This function analyzes a type, including its generic parameters and base classes,
    to create a dictionary that maps TypeVar instances to their corresponding concrete
    types. It handles cases where TypeVars are directly assigned, bound, or inherited.

    Args:
        t (type): The type to analyze. This could be a generic type
            (e.g., List[T], MyClass[T, U]) or a concrete type.

    Returns:
        dict[TypeVar, TypeVar | type | None]: A dictionary where keys are TypeVar
        instances found in the type's definition, and values are the corresponding
        concrete types (if available) or the TypeVar's bound (if no concrete
        type is available). If the type has no TypeVars, an empty dictionary
        is returned.
    """
    orig = get_origin(t) or t
    args = get_args(t)
    params = getattr(orig, "__parameters__", tuple())

    mapping = (
        {}
        if (len(params) == 0)
        else {var: var.__bound__ for var in params}
        if ((len(args) == 0) and (len(params) != 0))
        else dict(zip(params, args, strict=True))
    )

    for b in getattr(orig, "__orig_bases__", []):
        mapping.update(build_typevar_mapping(b))

    return mapping


def solve_typevar(t: type, var: TypeVar) -> type | None:
    """Resolve type variable from inheritance tree of given type.

    Arguments:
        t (type): type to analyze for specification of typevar
        var (TypeVar): type variable to resolve

    Returns:
        type|None: type if it can be resolved from the given type,
        otherwise falls back to typevars bound argument which might
        be None
    """
    # build the typevar mapping
    mapping = build_typevar_mapping(t)

    while isinstance(var, TypeVar):
        if var not in mapping:
            raise TypeError(f"Type variable '{var}' not found in the inheritance tree of '{t}'")
        assert var != mapping[var], (
            f"Type variable '{var}' should not map to itself in '{t}'. This indicates "
            "a potential infinite loop in type variable resolution or an invalid "
            "TypeVar usage."
        )
        var = mapping[var]

    return var

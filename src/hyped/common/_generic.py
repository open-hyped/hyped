"""Helper functionality to work with generic types."""
from typing import Generic, TypeVar, _GenericAlias, get_args, get_origin


def _get_typevar_index(t: type, var: TypeVar) -> None | int:
    """Internal helper function."""
    # trivial case
    if not hasattr(t, "__orig_bases__"):
        return None
    # search for typevar in base types
    for b in t.__orig_bases__:
        if isinstance(b, _GenericAlias):
            # check if base type is a generic alias
            a = get_args(b)
            if var in a:
                return a.index(var)

        else:
            o, a = get_origin(b), get_args(b)
            # return index of typevar if present
            if (o == Generic) and (var in a):
                return a.index(var)
    # typevar not found
    return None


def solve_typevar(t: type, var: TypeVar) -> type | None:
    """Resolve type variable from inheritance tree of given type.

    Arguments:
        t (type): type to analyse for specification of typevar
        var (TypeVar): type variable to resolve

    Returns:
        type|None: type if it can be resolved from the given type,
        otherwise falls back to typevars bound argument which might
        be None
    """

    def _solve(internal_t, internal_var):
        if not hasattr(internal_t, "__orig_bases__"):
            return internal_var.__bound__

        for b in internal_t.__orig_bases__:
            # check if the base is a generic type
            if isinstance(b, _GenericAlias):
                # get origin and arguments of generic type
                origin = get_origin(b)
                args = get_args(b)
                # search for typevar in origin
                index = _get_typevar_index(origin, internal_var)

                if index is not None:
                    return args[index]

                # recurse to base types
                candidate = _solve(origin, internal_var)
                if candidate is not None:
                    return candidate

            else:
                # not a generic type so just check the bases
                candidate = _solve(b, internal_var)
                if candidate is not None:
                    return candidate

    # follow typevar chain to specification
    while isinstance(var, TypeVar):
        var = _solve(t, var)

    return var

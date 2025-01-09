"""Utility functions and type mappings used throughout the core module.

This module provides various helper functions for working with data types,
feature mappings, and nested structures in the core module.
"""
from __future__ import annotations

from typing import Annotated, Any, Callable, TypeAlias, TypeVar

import pydantic
from datasets.features.features import FeatureType

from hyped.common._pydantic import TypeAdapterWithArbitraryTypesAllowed

from .features.dtypes import (
    UNDEFINED_SEQUENCE_LENGTH,
    DType,
    MappingType,
    PrimitiveType,
    SequenceType,
    build_dtype_from_hf_feature,
)
from .features.features import (
    PRIMITIVE_FEATURE_MAPPING,
    Feature,
    MappingFeature,
    SequenceFeature,
    build_feature_from_reference,
)
from .features.reference import ForwardReference
from .features.session import ValidationSession
from .features.validators import Len

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


def validate_hf_feature(
    hf_feature: FeatureType,
    annotation: Any,
    session: ValidationSession = ValidationSession(),
    context: dict[str, Any] = {"config": None},
) -> Feature:
    """Validate HuggingFace features against a type annotation.

    This function checks if the given HuggingFace feature structure is compatible
    with the provided type annotation and returns a validated feature instance.

    Args:
        hf_feature (FeatureType): The HuggingFace feature structure to validate.
        annotation (Any): The type annotation to validate against.
        session (ValidationSession, optional): The validation session instance
            to manage context and validation state. Defaults to a new session.
        context (dict[str, Any], optional): Additional context for validation.
            Defaults to an empty dictionary.

    Returns:
        Feature: A validated feature instance compatible with the provided annotation.

    Raises:
        RuntimeError: If the HuggingFace features are incompatible with the annotation.
    """
    # create a dummy feature instance according to the huggingface features
    hf_dtype = build_dtype_from_hf_feature(hf_feature)
    instance = build_feature_from_reference(ForwardReference(hf_dtype))
    # validate the feature instance with respect to the type annotation
    try:
        context = {**context, "session": session}
        adapter = TypeAdapterWithArbitraryTypesAllowed(annotation)
        with session:
            return adapter.validate_python(instance, context=context, strict=True)
    except pydantic.ValidationError as e:
        raise RuntimeError(
            f"The provided HuggingFace features '{hf_feature}' are "
            f"incompatible with the type annotation {annotation}."
        ) from e


def build_annotation_from_dtype(dtype: DType) -> Any:
    """Convert a dtype instance to a type annotation.

    Maps a :class:`DType` object into a Python type annotation suitable
    for describing the structure of a dataset feature.

    Args:
        dtype (DType): The data type to convert.

    Returns:
        Any: A type annotation corresponding to the provided dtype.
    """
    if isinstance(dtype, PrimitiveType):
        return PRIMITIVE_FEATURE_MAPPING[dtype]
    elif isinstance(dtype, SequenceType):
        seq_annotation = SequenceFeature[build_annotation_from_dtype(dtype.value_type)]
        return (
            Annotated[seq_annotation, Len(dtype.length)]
            if dtype.length != UNDEFINED_SEQUENCE_LENGTH
            else seq_annotation
        )
    elif isinstance(dtype, MappingType):
        annotations = {
            field_name: build_annotation_from_dtype(field_dtype)
            for field_name, field_dtype in dtype.fields
        }
        return type(
            f"DynamicMapping({annotations})", (MappingFeature,), {"__annotations__": annotations}
        )

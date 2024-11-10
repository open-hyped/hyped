from __future__ import annotations

import typing
from dataclasses import dataclass, replace
from functools import partial
from types import GenericAlias

import pydantic
from pydantic_core import PydanticCustomError, core_schema

from hyped.common.utils import is_python_version_less_than

from . import types
from .reference import FeatureKey, Reference

if is_python_version_less_than(3, 11):

    def get_original_bases(cls, /):
        """Return the class's "original" bases prior to modification by `__mro_entries__`."""
        try:
            return cls.__dict__.get("__orig_bases__", cls.__bases__)
        except AttributeError:
            raise TypeError(f"Expected an instance of type, not {type(cls).__name__!r}") from None

else:
    from types import get_original_bases  # noqa:


@dataclass(eq=True, frozen=True)
class _Feature(object):
    ref: Reference
    dtype: types.Type

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: typing.Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(cls)


class _Primitive(_Feature):
    ...


class _Bool(_Primitive):
    dtype: types.Type = types.BoolType

    def __post_init__(self) -> None:
        assert self.dtype is types.BoolType

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: typing.Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_before_validator_function(
            lambda v: (v if not isinstance(v, Reference) else _Bool(v, types.BoolType)),
            schema=handler(source_type),
        )


class _String(_Primitive):
    dtype: types.Type = types.BoolType

    def __post_init__(self) -> None:
        assert self.dtype is types.StringType


class Scalar(_Primitive):
    ...


class ExpectedScalarType(pydantic.WrapValidator):
    def __init__(self, dtype: types.Type):
        def validator_fn(inst, validator):
            if isinstance(inst, Reference):
                # create scalar feature with expected data type
                # from reference instance
                inst = Scalar(inst, dtype)

            # run core validator
            inst = validator(inst)

            # make sure the data type matches the expectation
            if inst.dtype is not dtype:
                raise PydanticCustomError(
                    "Type Mismatch",
                    "Data type '{actual}' doesn't match expected data type '{expected}'",
                    {"actual": inst.dtype, "expected": dtype},
                )

            return inst

        super(ExpectedScalarType, self).__init__(func=validator_fn)


_Int8 = typing.Annotated[Scalar, ExpectedScalarType(types.Int8Type)]
_Int16 = typing.Annotated[Scalar, ExpectedScalarType(types.Int16Type)]
_Int32 = typing.Annotated[Scalar, ExpectedScalarType(types.Int32Type)]
_Int64 = typing.Annotated[Scalar, ExpectedScalarType(types.Int64Type)]

_UInt8 = typing.Annotated[Scalar, ExpectedScalarType(types.UInt8Type)]
_UInt16 = typing.Annotated[Scalar, ExpectedScalarType(types.UInt16Type)]
_UInt32 = typing.Annotated[Scalar, ExpectedScalarType(types.UInt32Type)]
_UInt64 = typing.Annotated[Scalar, ExpectedScalarType(types.UInt64Type)]

_Float16 = typing.Annotated[Scalar, ExpectedScalarType(types.Float16Type)]
_Float32 = typing.Annotated[Scalar, ExpectedScalarType(types.Float32Type)]
_Float64 = typing.Annotated[Scalar, ExpectedScalarType(types.Float64Type)]

T = typing.TypeVar("T")


@dataclass(eq=True, frozen=True)
class Sequence(typing.Sequence[T], _Feature):
    dtype: types.SequenceType

    def __post_init__(self) -> None:
        assert isinstance(self.dtype, types.SequenceType)

    def __len__(self) -> int:
        return len(self.dtype)

    @typing.overload
    def __getitem__(self, index: int) -> T:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> Sequence[T]:
        ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        # build the reference to the indexed value
        ref = Reference(FeatureKey(self.ref._key + (index,)), self.ref._node_id, self.ref._graph)
        # get the output data type of the indexing operation
        # and infer build the corrsponding feature
        return build_feature_from_dtype(ref, self.dtype[index])

    @classmethod
    def __init_subclass__(cls) -> None:
        raise EnvironmentError("Cannot inherit from type")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: typing.Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        adapter = None

        if isinstance(source_type, GenericAlias):
            # get the expected item type from the generic annotation
            assert typing.get_origin(source_type) is cls
            (expected_item_type,) = typing.get_args(source_type)
            # create the type adapter to validate the item type
            adapter = pydantic.TypeAdapter(expected_item_type)

        def validator_fn(inst, validator, info):
            if isinstance(inst, Reference) and adapter is None:
                # no value type specified
                raise RuntimeError()

            elif isinstance(inst, Reference):
                # create instance from reference
                value_feature = adapter.validate_python(inst, context=info.context)
                inst = Sequence(inst, dtype=types.SequenceType(value_feature.dtype))

            # run the core validator checking that the instance
            # is a valid sequence feature
            validator(inst)

            if adapter is not None:
                # create a copy of the instance but with length 1
                # and validate the value feature at position 0 as a representative
                # of all the sequence values
                value_feature = replace(inst, dtype=replace(inst.dtype, length=1))[0]
                adapter.validate_python(value_feature, context=info.context)

            return inst

        return core_schema.with_info_wrap_validator_function(
            validator_fn, schema=core_schema.is_instance_schema(Sequence)
        )


class InvalidKeyError(Exception):
    """Exception raised when an invalid key is found."""

    def __init__(self, message: str, invalid_keys=None):
        super().__init__(message)
        self.invalid_keys = invalid_keys or []


class MissingKeyError(Exception):
    """Exception raised when a required key is missing."""

    def __init__(self, message: str, missing_keys=None):
        super().__init__(message)
        self.missing_keys = missing_keys or []


if typing.TYPE_CHECKING:

    class Mapping(typing.TypedDict, _Feature):
        dtype: types.MappingType

else:

    @dataclass(eq=True, frozen=True)
    class Mapping(typing.Mapping, _Feature):
        dtype: types.MappingType

        def __post_init__(self) -> None:
            assert isinstance(self.dtype, types.MappingType)

            # base mapping type allows arbitrary keys
            if type(self) is Mapping:
                return

            # get the set of valid keys
            valid_keys = set(typing.get_type_hints(type(self)).keys()) - set(
                typing.get_type_hints(Mapping).keys()
            )
            # get the specified keys from the factory entries
            set_keys = set(self.keys())
            # compute the invalid and missing keys
            invalid_keys = set_keys - valid_keys
            missing_keys = valid_keys - set_keys

            if len(invalid_keys) > 0:
                raise InvalidKeyError(f"Invalid Keys: {invalid_keys}", invalid_keys=invalid_keys)

            if len(missing_keys) > 0:
                raise MissingKeyError(f"Missing Keys: {missing_keys}", missing_keys=missing_keys)

        def __len__(self) -> int:
            return len(self.dtype)

        def __iter__(self) -> typing.Iterable[str]:
            return iter(self.dtype)

        def __getitem__(self, key: str) -> _Feature:
            # build the reference to the indexed value
            ref = Reference(FeatureKey(self.ref._key + (key,)), self.ref._node_id, self.ref._graph)
            # get the output data type of the indexing operation
            # and infer build the corrsponding feature
            return build_feature_from_dtype(ref, self.dtype[key])

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            ignore_keys = set(typing.get_type_hints(Mapping).keys())

            def build_validator_model(cls) -> pydantic.BaseModel:
                if not (
                    (isinstance(cls, type) and issubclass(cls, Mapping) and (cls is not Mapping))
                    or (
                        isinstance(cls, GenericAlias)
                        and issubclass(typing.get_origin(cls), Mapping)
                    )
                ):
                    # trivial case: the class is not a subclass
                    # of mapping or the base class itself
                    return cls

                # get the annotations of only this type
                annotations = typing.get_type_hints(cls, include_extras=True)
                annotations = {
                    k: (h, pydantic.Field()) for k, h in annotations.items() if k not in ignore_keys
                }
                # prepare the bsae classes
                bases = get_original_bases(cls)
                bases = map(build_validator_model, bases)
                bases = tuple(b for b in bases if b is not Mapping)
                # add the pydantic base model as a base class
                if not any(
                    (isinstance(base, type) and issubclass(base, pydantic.BaseModel))
                    for base in bases
                ):
                    bases = (pydantic.BaseModel,) + bases

                # create the validator model
                model = pydantic.create_model(
                    f"MappingValidatorFor{cls.__name__}", **annotations, __base__=bases
                )
                # apply the type variable values if the model is generic
                if isinstance(cls, GenericAlias):
                    model = model.__class_getitem__(typing.get_args(cls))

                return model

            # build the validation model in case in case the class is not the base mapping
            # class itself, otherwise no field validation is done
            model = build_validator_model(source_type) if cls is not Mapping else None
            model = model if model is not Mapping else None

            def validator_fn(inst, validator, info):
                if isinstance(inst, Reference) and model is None:
                    raise RuntimeError()

                elif isinstance(inst, Reference):
                    # infer member types from annotations using the validation model
                    members = {key: inst for key in model.model_fields.keys()}
                    members = {key: field.dtype for key, field in model.model_validate(members)}
                    # create the mapping instance from the member types
                    inst = cls(inst, types.MappingType.from_dict(members))

                # run the core validator checking that the instance
                # is a valid mapping
                validator(inst)

                if model is not None:
                    # validate the field types
                    model.model_validate(dict(inst), context=info.context)

                strict = (info.context or {}).get("strict", False)
                # convert the instance to the actual class type in case of
                # strict validation
                return inst if isinstance(inst, cls) or not strict else cls(inst.ref, inst.dtype)

            return core_schema.with_info_wrap_validator_function(
                validator_fn, schema=core_schema.is_instance_schema(Mapping)
            )


TYPE_TO_FEATURE_MAPPING = {
    types.BoolType: _Bool,
    types.StringType: _String,
    types.PrimitiveType: Scalar,
    types.SequenceType: Sequence,
    types.MappingType: Mapping,
}


def build_feature_from_dtype(ref: Reference, dtype: types.Type) -> _Feature:
    # infer the feature type from the data type
    ftype = TYPE_TO_FEATURE_MAPPING.get(dtype, TYPE_TO_FEATURE_MAPPING.get(type(dtype), None))
    # create the feature instance
    return ftype(ref, dtype)


def build_feature_from_annotation(
    ref: Reference,
    annotation: typing.Any,
    typevar_mapping: dict[typing.TypeVar, types.Type] = {},
    context: dict[str, typing.Any] = {},
) -> _Feature:
    # get the return annotation
    has_parameters = hasattr(annotation, "__parameters__") and len(annotation.__parameters__) > 0

    base = (pydantic.BaseModel,)

    if has_parameters:
        # add generic base in case the return annotation has any parameters
        base += (typing.Generic[annotation.__parameters__],)

    elif isinstance(annotation, typing.TypeVar):
        # add generic base in case the return annotation is just a typevar
        base += (typing.Generic[annotation],)

    # create the validation model
    builder = pydantic.create_model(
        f"FeatureBuilder({annotation})", field=(annotation, pydantic.Field()), __base__=base
    )

    if hasattr(builder, "__parameters__"):
        # lookup typevars in return annotation
        dtypes = [typevar_mapping[t] for t in builder.__parameters__]
        # create a feature annotation for each typevar
        # which resolves to the corresponding feature type
        features = [
            typing.Annotated[
                _Feature, pydantic.BeforeValidator(partial(build_feature_from_dtype, dtype=dtype))
            ]
            for dtype in dtypes
        ]
        # apply the feature annotations to the return model
        builder = builder.__class_getitem__(*features)

    # build the output feature
    return builder.model_validate({"field": ref}, context=context).field

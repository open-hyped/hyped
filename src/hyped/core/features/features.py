from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import GenericAlias
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Iterable, get_type_hints

import pyarrow as pa
from pydantic import BaseModel, Field, GetCoreSchemaHandler, TypeAdapter, create_model
from pydantic_core import core_schema
from typing_extensions import Self

from hyped.common.typing import ArrowType
from hyped.common.utils import is_python_version_less_than
from hyped.core.features.feature_key import FeatureKey

from .factories import BaseFeatureFactory, FeatureFactoryFromInstance
from .reference import Reference

if is_python_version_less_than(3, 11):

    def get_original_bases(cls, /):
        """Return the class's "original" bases prior to modification by `__mro_entries__`."""
        try:
            return cls.__dict__.get("__orig_bases__", cls.__bases__)
        except AttributeError:
            raise TypeError(f"Expected an instance of type, not {type(cls).__name__!r}") from None

else:
    from types import get_original_bases  # noqa:

if not TYPE_CHECKING:

    def TypeVar(*args, **kwargs):
        # create a type var validator
        from .engine import type_var_register

        validator = type_var_register.create_validator()
        # annotate the bound argument with the validator
        bound = kwargs.pop("bound", Any)
        bound = Annotated[bound, validator]
        # create the typevar
        _T = typing.TypeVar(*args, bound=bound, **kwargs)
        # register the typevar
        type_var_register.register(_T, validator)

        return _T

else:
    pass  # noqa:


@dataclass(eq=True, frozen=True)
class _Feature(ABC, Reference):
    @property
    @abstractmethod
    def _pa_type(self) -> ArrowType:
        ...

    @property
    def _type_hint(self) -> type[Self]:
        return getattr(self, "__orig_class__", type(self))


@dataclass(eq=True, frozen=True)
class _Primitive(_Feature):
    _primitive_pa_type: ClassVar[ArrowType]

    @property
    def _pa_type(self) -> ArrowType:
        return type(self)._primitive_pa_type


@dataclass(eq=True, frozen=True)
class _String(_Primitive):
    _primitive_pa_type = pa.string()


@dataclass(eq=True, frozen=True)
class _Bool(_Primitive):
    _primitive_pa_type = pa.bool_()


@dataclass(eq=True, frozen=True)
class _Int8(_Primitive):
    _primitive_pa_type = pa.int8()


@dataclass(eq=True, frozen=True)
class _Int16(_Primitive):
    _primitive_pa_type = pa.int16()


@dataclass(eq=True, frozen=True)
class _Int32(_Primitive):
    _primitive_pa_type = pa.int32()


@dataclass(eq=True, frozen=True)
class _Int64(_Primitive):
    _primitive_pa_type = pa.int64()


@dataclass(eq=True, frozen=True)
class _UInt8(_Primitive):
    _primitive_pa_type = pa.uint8()


@dataclass(eq=True, frozen=True)
class _UInt16(_Primitive):
    _primitive_pa_type = pa.uint16()


@dataclass(eq=True, frozen=True)
class _UInt32(_Primitive):
    _primitive_pa_type = pa.uint32()


@dataclass(eq=True, frozen=True)
class _UInt64(_Primitive):
    _primitive_pa_type = pa.uint64()


@dataclass(eq=True, frozen=True)
class _Float16(_Primitive):
    _primitive_pa_type = pa.float16()


@dataclass(eq=True, frozen=True)
class _Float32(_Primitive):
    _primitive_pa_type = pa.float32()


@dataclass(eq=True, frozen=True)
class _Float64(_Primitive):
    _primitive_pa_type = pa.float64()


_Type = typing.TypeVar("_Type", bound=_Feature)


@dataclass(eq=True, frozen=True)
class Sequence(typing.Sequence[_Type], _Feature):
    _itemtype: type[_Type]
    _factory: BaseFeatureFactory[_Type]
    _length: int

    @property
    def _pa_type(self) -> ArrowType:
        # TODO: look into fixed-size list instead of fixed shape tensor
        return (
            pa.list_(self._factory._pa_type)
            if self._length == -1
            else pa.fixed_shape_tensor(self._factory._pa_type, [self._length])
        )

    @property
    def _type_hint(self) -> type[Self]:
        return Sequence[self._itemtype]

    def __getitem__(self, index: int | slice) -> _Type:
        if isinstance(index, slice):
            # assume the length of the sequence is dynamic
            new_length = -1

            if self._length >= 0:
                # compute length of the subsequence
                start, stop, step = index.indices(self._length)
                new_length = (stop - start) // step

            return Sequence(
                FeatureKey(*self._key, index),
                self._node_id,
                self._graph,
                self._itemtype,
                self._factory,
                new_length,
            )

        if (self._length >= 0) and (index >= self._length):
            raise IndexError("Index out of bounds")  # TODO: add information about the key

        return self._factory(FeatureKey(*self._key, index), self._node_id, self._graph)

    T = typing.TypeVar("T", bound=_Feature)

    def __slice_constructor__(self, items: Iterable[T]) -> Sequence[T]:
        item = next(iter(items))
        # get the slice from the item key
        idx = item._key[len(self._key)]
        assert isinstance(idx, slice)
        # build the key from the item instance
        # the item is just one element of the slice of the sequence
        # to show this in the key we just need to remove the index from the key
        key = item._key[: len(self._key) + 1] + item._key[len(self._key) + 2 :]
        key = tuple.__new__(FeatureKey, key)

        if self._length >= 0:
            # compute length of the subsequence
            start, stop, step = idx.indices(self._length)
            length = (stop - start) // step

        else:
            length = -1

        return Sequence(
            _key=key,
            _node_id=self._node_id,
            _graph=self._graph,
            _itemtype=type(item),
            _factory=FeatureFactoryFromInstance(item),
            _length=length,
        )

    def __len__(self) -> int:
        return self._length

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # get the expected item type from the generic annotation
        assert typing.get_origin(source_type) is cls
        (expected_item_type,) = typing.get_args(source_type)
        # create the type adapter to validate the item type
        validator = TypeAdapter(expected_item_type)

        def validator_fn(inst, info):
            if isinstance(inst, dict) and ("_factory" in inst):
                # create the instance
                inst = cls(**inst)

            elif isinstance(inst, dict) and ("_factory" not in inst):
                # infer item type by creating an instance
                item = validator.validate_python(inst, context=info.context)
                inst = cls(
                    **inst,
                    _factory=FeatureFactoryFromInstance(item),
                    _itemtype=item._type_hint,
                    _length=-1,
                )

            # validate the item type of the instance
            item = inst._factory(("key",), inst._node_id, inst._graph)
            validator.validate_python(item, context=info.context)
            # return the instance
            return inst

        # build pydantic core schema including the item type validator
        return core_schema.with_info_before_validator_function(
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


if TYPE_CHECKING:

    class Mapping(typing.TypedDict, _Feature):
        pass

else:

    @dataclass(eq=True, frozen=True)
    class Mapping(typing.Mapping, _Feature):
        _factories: dict[str, BaseFeatureFactory]

        @property
        def _pa_type(self) -> ArrowType:
            field_names = sorted(self._factories.keys())
            return pa.struct([(key, self._factories[key]._pa_type) for key in field_names])

        def __post_init__(self) -> None:
            # get the set of valid keys
            valid_keys = set(get_type_hints(type(self)).keys()) - set(
                get_type_hints(Mapping).keys()
            )
            # get the specified keys from the factory entries
            set_keys = set(self._factories.keys())
            # compute the invalid and missing keys
            invalid_keys = set_keys - valid_keys
            missing_keys = valid_keys - set_keys

            if len(invalid_keys) > 0:
                raise InvalidKeyError(f"Invalid Keys: {invalid_keys}", invalid_keys=invalid_keys)

            if len(missing_keys) > 0:
                raise MissingKeyError(f"Missing Keys: {missing_keys}", missing_keys=missing_keys)

        def __getitem__(self, key: str) -> _Feature:
            return self._factories[key](FeatureKey(*self._key, key), self._node_id, self._graph)

        def __len__(self) -> int:
            return len(self._factories)

        def __iter__(self) -> str:
            return iter(self._factories)

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            ignore_keys = set(get_type_hints(Mapping).keys())

            def build_validator_model(cls) -> BaseModel:
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
                annotations = get_type_hints(cls, include_extras=True)
                annotations = {
                    k: (h, Field()) for k, h in annotations.items() if k not in ignore_keys
                }
                # prepare the bsae classes
                bases = get_original_bases(cls)
                bases = map(build_validator_model, bases)
                bases = tuple(b for b in bases if b is not typing.Mapping)
                # add the pydantic base model as a base class
                if not any(
                    isinstance(base, type) and issubclass(base, BaseModel) for base in bases
                ):
                    bases = (BaseModel,) + bases

                # create the validator model
                model = create_model(
                    f"MappingValidatorFor{cls.__name__}", **annotations, __base__=bases
                )
                # apply the type variable values if the model is generic
                if isinstance(cls, GenericAlias):
                    model = model.__class_getitem__(typing.get_args(cls))

                return model

            # build the validator model for the source type
            validator = build_validator_model(source_type)

            def validator_fn(inst, info):
                if isinstance(inst, dict) and ("_factories" in inst):
                    # create the instance
                    inst = cls(**inst)

                elif isinstance(inst, dict) and ("_factories" not in inst):
                    # create the instance and infer the member types from the annotations
                    members = {key: inst for key in validator.model_fields.keys()}
                    members = validator.model_validate(members, context=info.context)
                    # create the factories from the inferred members
                    factories = {key: FeatureFactoryFromInstance(item) for key, item in members}
                    inst = cls(**inst, _factories=factories)

                strict = (info.context or {}).get("strict", False)
                # instance must be a mapping
                if not isinstance(inst, Mapping):
                    raise TypeError()
                # validate the instance
                validator.model_validate(dict(inst), context=info.context)
                # convert the instance to the desired type
                return (
                    inst
                    if isinstance(inst, cls) or not strict
                    else cls(inst._key, inst._node_id, inst._graph, inst._factories)
                )

            # apply the validator function before the schema to
            # apply conversion to the desired type
            return core_schema.with_info_before_validator_function(
                validator_fn, schema=core_schema.is_instance_schema(Mapping)
            )

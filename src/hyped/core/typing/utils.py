from functools import partial

import datasets
from datasets.features.features import FeatureType

from .types import (
    Mapping,
    Sequence,
    _Bool,
    _Feature,
    _Float16,
    _Float32,
    _Float64,
    _Int8,
    _Int16,
    _Int32,
    _Int64,
    _String,
    _TypeFactory,
)

PRIMITIVE_FEATURE_TO_TYPE_MAPPING = {
    "bool": _Bool,
    "string": _String,
    "int8": _Int8,
    "int16": _Int16,
    "int32": _Int32,
    "int64": _Int64,
    "float16": _Float16,
    "float32": _Float32,
    "float64": _Float64,
}


def _type_from_feature(feature: FeatureType) -> type[_Feature]:
    if isinstance(feature, datasets.Value):
        return PRIMITIVE_FEATURE_TO_TYPE_MAPPING[feature.dtype]

    if isinstance(feature, datasets.Sequence):
        return Sequence[_type_from_feature(feature.feature)]

    if isinstance(feature, datasets.Features):
        annotations = {k: _type_from_feature(f) for k, f in feature.items()}
        return type("Mapping", (Mapping,), {"__annotations__": annotations})


def _type_factory_from_feature(feature: FeatureType) -> _TypeFactory[_Feature]:
    if isinstance(feature, datasets.Value):
        return PRIMITIVE_FEATURE_TO_TYPE_MAPPING[feature.dtype]

    if isinstance(feature, datasets.Sequence):
        _type = _type_from_feature(feature.feature)
        _type_factory = _type_factory_from_feature(feature.feature)
        return partial(
            Sequence[_type], _itemtype=_type, _factory=_type_factory, _length=feature.length
        )

    if isinstance(feature, datasets.Features):
        _type = _type_from_feature(feature)
        return partial(
            _type, _factories={k: _type_factory_from_feature(f) for k, f in feature.items()}
        )

    raise TypeError(feature)

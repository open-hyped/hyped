"""PyArrow helper functionality."""
import datetime
import json
from dataclasses import dataclass
from typing import Any, List, Literal, TypeVar, _TypedDictMeta, get_args, get_origin

import datasets
import pyarrow as pa
import pydantic
from datasets.features.features import FeatureType

from hyped.core.abstract import AbstractDataFlowGraph
from hyped.core.features.factories import BaseFeatureFactory
from hyped.core.features.features import (
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
)

from ..core.features.feature_key import FeatureKey
from .typing import NodeId

# Direct primitive type mapping, defaulting to 32-bit where possible
PRIMITIVE_TYPE_MAP = {
    bool: "bool",
    int: "int32",  # Default to 32-bit integer
    float: "float32",  # Default to 32-bit float
    str: "string",
    bytes: "binary",
    datetime.datetime: "date32",  # Default to 32-bit datetime
    datetime.time: "time32",  # Default to 32-bit time
}

PRIMITIVE_FEATURE_DTYPE_TO_TYPE_MAPPING = {
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


def get_sequence_length(seq: datasets.Sequence | list | tuple) -> int:
    """Get the length of a given sequence feature.

    Arguments:
        seq (Sequence | list | tuple): sequence to get the length of

    Returns:
        length (int):
            the length of the given sequence. Returns -1 for
            sequences of undefined length
    """
    assert isinstance(seq, (datasets.Sequence, list, tuple))
    return seq.length if isinstance(seq, datasets.Sequence) else -1


def get_sequence_feature(seq: datasets.Sequence | list | tuple) -> FeatureType:
    """Get the item feature type of a given sequence feature.

    Arguments:
        seq (Sequence | list | tuple):
            sequence to get the item feature type of

    Returns:
        feature (FeatureType):
            the item feature type of the sequence
    """
    assert isinstance(seq, (datasets.Sequence, list, tuple))
    return seq.feature if isinstance(seq, datasets.Sequence) else seq[0]


def get_nested_type(schema: FeatureType) -> pa.DataType:
    """Get nested arrow type.

    Converts a datasets.FeatureType into a pyarrow.DataType,
    and acts as the inverse of `datasets.Dataset.generate_from_arrow_type`.

    It performs double-duty as the implementation of Features.type and
    handles the conversion of datasets.Feature->pa.struct

    Source: https://github.com/huggingface/datasets/blob/1a598a0dfd699f7a7ebe9eb6273fb5ac4b9e519a/src/datasets/features/features.py#L1184

    Differs from the source in that it doesn't convert Sequence[dict]->dict[Sequence].
    Instead it follows the exact feature type provided.
    """  # noqa: E501
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, datasets.Features):
        # Features is subclass of dict, and dict order is
        # deterministic since Python 3.6
        return pa.struct({key: get_nested_type(schema[key]) for key in schema})
    elif isinstance(schema, dict):
        # however don't sort on struct types since the order matters
        return pa.struct({key: get_nested_type(schema[key]) for key in schema})
    elif isinstance(schema, (list, tuple)):
        if len(schema) != 1:
            raise ValueError(
                "When defining list feature, you should just "
                "provide one example of the inner type"
            )
        value_type = get_nested_type(schema[0])
        return pa.list_(value_type)
    elif isinstance(schema, datasets.Sequence):
        return pa.list_(get_nested_type(schema.feature), schema.length)

    # Other objects are callable which returns their data type
    # (ClassLabel, Array2D, Translation, Arrow datatype creation methods)
    return schema()


def convert_features_to_arrow_schema(features: datasets.Features) -> pa.Schema:
    """Convert dataset.Features to pyarrow.Schema.

    This is similar to the `datasets.Dataset.arrow_schema` property
    but parses the features slightly differently to ensure the schema
    matches the features exactly (see `get_nested_type` for more information)

    Args:
        features (Features): dataset features instance to convert

    Returns:
        schema (pa.Schema): pyarrow schema representing dataset features
    """
    dtype = get_nested_type(features)
    hf_metadata = {"info": {"features": features.to_dict()}}
    return pa.schema(dtype).with_metadata({"huggingface": json.dumps(hf_metadata)})


def type_hint_to_feature(type_hint: Any) -> FeatureType:
    """Convert a Python type hint into a compatible Hugging Face Features structure.

    This function takes a Python type alias or type hint and recursively converts
    it into a Hugging Face :class:`Features` structure. It supports the following cases:

    - Primitive types: Mapped to corresponding :class:`Value` types.
    - Lists: Recursively resolves the element type and creates a list structure.
    - TypedDicts: Converts each field in the dictionary to a corresponding feature.
    - Literals: Mapped to Hugging Face's :class:`ClassLabel` type.

    Args:
        type_hint (Any): The type hint to convert. Can be a primitive type,
            a list, a TypedDict, or a Literal.

    Returns: A Hugging Face :class:`FeatureType` structure representing the given type hint.

    Example:

    .. code-block:: python

        from typing import List, Literal, TypedDict

        class MyExample(TypedDict):
            id: int
            text: str
            scores: List[float]
            label: Literal['positive', 'negative']

        features = type_hint_to_feature(MyExample)
        print(features)
    """

    # Primitive types
    if type_hint in PRIMITIVE_TYPE_MAP:
        return datasets.Value(PRIMITIVE_TYPE_MAP[type_hint])

    origin = get_origin(type_hint)
    # List case: recursively resolve the list element type
    if (origin is list) or (origin is List):
        element_type = get_args(type_hint)[0]
        return [type_hint_to_feature(element_type)]

    # TypedDict case: convert each field in the dictionary
    if isinstance(type_hint, type) and isinstance(type_hint, _TypedDictMeta):
        return datasets.Features(
            {
                field: type_hint_to_feature(field_type)
                for field, field_type in type_hint.__annotations__.items()
            }
        )

    # Literal case: map to ClassLabel
    if origin is Literal:
        options = list(get_args(type_hint))
        return datasets.ClassLabel(names=options)

    # Fallback case
    raise TypeError(f"Unsupported type: {type_hint}")


def _build_type_from_feature(feature: FeatureType) -> type:
    if isinstance(feature, datasets.Value):
        return PRIMITIVE_FEATURE_DTYPE_TO_TYPE_MAPPING[feature.dtype]

    if isinstance(feature, datasets.Sequence):
        return Sequence[_build_type_from_feature(feature.feature)]

    if isinstance(feature, datasets.Features):
        annotations = {k: _build_type_from_feature(f) for k, f in feature.items()}
        return type("Mapping", (Mapping,), {"__annotations__": annotations})

    raise TypeError()


T = TypeVar("T", bound=_Feature)


@dataclass
class TypeFactoryFromFeature(BaseFeatureFactory[T]):
    feature: FeatureType

    def _create_instance(
        self, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph
    ) -> Any:
        if isinstance(self.feature, datasets.Value):
            # create instance for primitives
            return _build_type_from_feature(self.feature)(_key, _node_id, _graph)

        if isinstance(self.feature, datasets.Sequence):
            item_feature = get_sequence_feature(self.feature)
            item_type = _build_type_from_feature(item_feature)
            return Sequence[item_type](
                _key=_key,
                _node_id=_node_id,
                _graph=_graph,
                _itemtype=item_type,
                _factory=TypeFactoryFromFeature(item_feature),
                _length=get_sequence_length(self.feature),
            )

        if isinstance(self.feature, datasets.Features):
            mapping_type = _build_type_from_feature(self.feature)
            return mapping_type(
                _key=_key,
                _node_id=_node_id,
                _graph=_graph,
                _factories={
                    key: TypeFactoryFromFeature(feature) for key, feature in self.feature.items()
                },
            )

    def __call__(self, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph) -> T:
        # get the type hint if it is provided
        hint = getattr(self, "__orig_class__", None)
        hint = get_args(hint) if hint is not None else []
        hint = None if len(hint) == 0 else hint[0]
        assert (hint is None) or (not isinstance(hint, TypeVar))

        # create the instance
        inst = self._create_instance(_key, _node_id, _graph)

        if hint is None:
            # no validation
            return inst

        try:
            # validate and convert the type instance
            adapter = pydantic.TypeAdapter(hint)
            return adapter.validate_python(inst, context={"strict": True})

        except pydantic.ValidationError:
            raise RuntimeError()

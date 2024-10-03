"""PyArrow helper functionality."""
import datetime
import json
from typing import Any, List, Literal, _TypedDictMeta, get_args, get_origin

import pyarrow as pa
from datasets import ClassLabel, Features, Sequence, Value
from datasets.features.features import FeatureType

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
    if isinstance(schema, Features):
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
    elif isinstance(schema, Sequence):
        return pa.list_(get_nested_type(schema.feature), schema.length)

    # Other objects are callable which returns their data type
    # (ClassLabel, Array2D, Translation, Arrow datatype creation methods)
    return schema()


def convert_features_to_arrow_schema(features: Features) -> pa.Schema:
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
        return Value(PRIMITIVE_TYPE_MAP[type_hint])

    origin = get_origin(type_hint)
    # List case: recursively resolve the list element type
    if (origin is list) or (origin is List):
        element_type = get_args(type_hint)[0]
        return [type_hint_to_feature(element_type)]

    # TypedDict case: convert each field in the dictionary
    if isinstance(type_hint, type) and isinstance(type_hint, _TypedDictMeta):
        return Features(
            {
                field: type_hint_to_feature(field_type)
                for field, field_type in type_hint.__annotations__.items()
            }
        )

    # Literal case: map to ClassLabel
    if origin is Literal:
        options = list(get_args(type_hint))
        return ClassLabel(names=options)

    # Fallback case
    raise TypeError(f"Unsupported type: {type_hint}")

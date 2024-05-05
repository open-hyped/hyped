"""Pydantic helper functionality."""
import datetime

import datasets
import pydantic

# map datasets value dtype to
DATASETS_VALUE_TYPE_MAPPING = {
    "bool": bool,
    "int8": int,
    "int16": int,
    "int32": int,
    "int64": int,
    "uint8": int,
    "uint16": int,
    "uint32": int,
    "uint64": int,
    "float16": float,
    "float32": float,
    "float64": float,
    "string": str,
    "large_string": str,
    "date32": datetime.datetime,
    "date64": datetime.datetime,
    "time32": datetime.time,
    "time64": datetime.time,
}


def pydantic_model_from_features(
    features: datasets.Features,
) -> pydantic.BaseModel:
    """Create a pydantic model from dataset features.

    Arguments:
        features (Features): datasets features to build the pydantic model for

    Returns:
        model (pydantic.BaseModel):
            pydantic model matching the structure of the dataset features.
    """
    fields = {}
    for k, field_type in features.items():
        if isinstance(field_type, datasets.Value):
            # get data type for the given field
            dtype = DATASETS_VALUE_TYPE_MAPPING.get(
                field_type.dtype, field_type.pa_type.to_pandas_dtype()
            )
            # set field
            fields[k] = (
                dtype | None,
                None,
            )

        elif isinstance(field_type, datasets.ClassLabel):
            fields[k] = (Literal[tuple(field_type.names)] | None, None)

        elif isinstance(field_type, datasets.Sequence):
            # infer dtype for sequence values
            dtype = (
                pydantic_model_from_features({"field": field_type.feature})
                .model_fields["field"]
                .annotation
            )
            # set field
            fields[k] = (list[dtype], pydantic.Field(default_factory=list))

        elif isinstance(field_type, (dict, datasets.Features)):
            model = pydantic_model_from_features(field_type)
            # set field
            fields[k] = (
                model,
                pydantic.Field(default_factory=model),
            )

    return pydantic.create_model(
        "Model",
        **fields,
        __config__=pydantic.ConfigDict(
            arbitrary_types_allowed=True, validate_assignment=True
        ),
    )

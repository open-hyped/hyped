from typing import Any, Hashable, Iterable, NotRequired
from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Sequence, Value
from pydantic import BaseModel
from typing_extensions import Annotated

from hyped.base.config import BaseConfig
from hyped.data.flow.core.refs.inputs import (
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    FeatureValidator,
    InputRefs,
    InputRefsValidator,
)

# import hyped.data.processors.base
from hyped.data.flow.core.refs.ref import NONE_REF, FeaturePointer, FeatureRef


def ptr_set(refs: Iterable[FeatureRef]) -> set[FeaturePointer]:
    return {r.ptr for r in refs}


def ptr_dict(d: dict[Hashable, FeatureRef]) -> dict[Hashable, FeaturePointer]:
    return {k: r.ptr for k, r in d.items()}


def test_error_on_invalid_annotation():
    with pytest.raises(TypeError):

        class CustomInputRefs(InputRefs):
            x: FeatureRef

        InputRefsValidator(BaseConfig(), CustomInputRefs)

    with pytest.raises(TypeError):

        class CustomInputRefs(InputRefs):
            x: Annotated[FeatureRef, object]

        InputRefsValidator(BaseConfig(), CustomInputRefs)

    with pytest.raises(TypeError):

        class CustomInputRefs(InputRefs):
            x: Annotated[object, FeatureValidator(MagicMock())]

        InputRefsValidator(BaseConfig(), CustomInputRefs)

    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(MagicMock())]

    InputRefsValidator(BaseConfig(), CustomInputRefs)


def test_error_on_invalid_value():
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(MagicMock)]

    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)

    with pytest.raises(ValueError):
        validator.validate(**CustomInputRefs(x=int))


def test_feature_validator():
    # create mock validators
    x_validator = MagicMock()
    y_validator = MagicMock()

    # create custom input refs class using mock validators
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(x_validator)]
        y: Annotated[FeatureRef, FeatureValidator(y_validator)]

    # create validator instance from custom input refs type
    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)

    k, n, f = tuple(), "", None
    # create dummy input refs
    x_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("int32"))
    y_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("string"))
    # instantiate the input refs and make sure the
    # validators were called with the correct arguments
    refs = CustomInputRefs(x=x_ref, y=y_ref)
    validator.validate(**refs)

    x_validator.assert_called_with(validator.config, x_ref)
    y_validator.assert_called_with(validator.config, y_ref)


def test_check_feature_equals():
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, CheckFeatureEquals(Value("int32"))]
        y: Annotated[FeatureRef, CheckFeatureEquals(Value("string"))]

    # create validator instance from custom input refs type
    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)

    k, n, f = tuple(), "", None
    # create dummy input refs
    x_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("int32"))
    y_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("string"))
    # create input refs, all types match
    refs = CustomInputRefs(x=x_ref, y=y_ref)
    validator.validate(**refs)

    # create input refs but one type doesn't match the expectation
    with pytest.raises(RuntimeError):
        refs = CustomInputRefs(x=x_ref, y=x_ref)
        validator.validate(**refs)


def test_check_or_feature_validator():
    # create custom input refs class using mock validators
    class CustomInputRefs(InputRefs):
        x: Annotated[
            FeatureRef,
            CheckFeatureEquals(Value("int32"))
            | CheckFeatureEquals(Value("string")),
        ]

    # create validator instance from custom input refs type
    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)

    k, n, f = tuple(), "", None
    # create dummy input refs
    x_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("int32"))
    y_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("string"))
    # matches first validator
    refs = CustomInputRefs(x=x_ref)
    validator.validate(**refs)

    # matches second validator
    refs = CustomInputRefs(x=y_ref)
    validator.validate(**refs)

    z_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("float32"))

    with pytest.raises(RuntimeError):
        refs = CustomInputRefs(x=z_ref)
        validator.validate(**refs)


def test_check_feature_is_sequence():
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, CheckFeatureIsSequence(Value("int32"))]
        y: Annotated[FeatureRef, CheckFeatureIsSequence(Value("string"))]

    # create validator instance from custom input refs type
    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)

    k, n, f = tuple(), "", None
    # create dummy input refs
    x_ref = FeatureRef(
        key_=k, node_id_=n, flow_=f, feature_=Sequence(Value("int32"))
    )
    y_ref = FeatureRef(
        key_=k,
        node_id_=n,
        flow_=f,
        feature_=Sequence(Value("string"), length=2),
    )
    z_ref = FeatureRef(
        key_=k,
        node_id_=n,
        flow_=f,
        feature_=Sequence(Value("string"), length=4),
    )
    # create input refs, all types match
    refs = CustomInputRefs(x=x_ref, y=y_ref)
    validator.validate(**refs)

    # create input refs but one type doesn't match the expectation
    with pytest.raises(RuntimeError):
        refs = CustomInputRefs(x=x_ref, y=x_ref)
        validator.validate(**refs)

    # test with specified length
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, CheckFeatureIsSequence(Value("int32"))]
        y: Annotated[
            FeatureRef, CheckFeatureIsSequence(Value("string"), length=2)
        ]

    # create validator instance from custom input refs type
    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)

    # create input refs, all types match
    CustomInputRefs(x=x_ref, y=y_ref)
    # create input refs but one type doesn't match the expectation
    with pytest.raises(RuntimeError):
        refs = CustomInputRefs(x=x_ref, y=x_ref)
        validator.validate(**refs)
    # create input refs but sequence length doesn't match the expectation
    with pytest.raises(RuntimeError):
        refs = CustomInputRefs(x=x_ref, y=z_ref)
        validator.validate(**refs)


def test_required_input_refs():
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(MagicMock())]
        y: Annotated[FeatureRef, FeatureValidator(MagicMock())]

    # create validator instance from custom input refs type
    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)

    assert validator.required_keys == {"x", "y"}
    assert validator.optional_keys == set()

    k, n, f = tuple(), "", MagicMock
    # create dummy input refs
    x_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("int32"))
    y_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("string"))
    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref, y=y_ref)
    input_refs = validator.validate(**input_refs)

    # check properties
    assert ptr_set(input_refs.refs) == ptr_set([x_ref, y_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict(
        {"x": x_ref, "y": y_ref}
    )
    assert input_refs.features_ == Features(
        {"x": x_ref.feature_, "y": y_ref.feature_}
    )


def test_optional_input_refs():
    k, n, f = tuple(), "", MagicMock
    # create dummy input refs
    x_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("int32"))
    y_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("string"))

    class CustomInputRefs(InputRefs):
        x: NotRequired[Annotated[FeatureRef, FeatureValidator(MagicMock())]]
        y: NotRequired[Annotated[FeatureRef, FeatureValidator(MagicMock())]]

    # create validator instance from custom input refs type
    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)

    assert validator.required_keys == set()
    assert validator.optional_keys == {"x", "y"}

    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref, y=y_ref)
    input_refs = validator.validate(**input_refs)
    # check properties
    assert ptr_set(input_refs.refs) == ptr_set([x_ref, y_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict(
        {"x": x_ref, "y": y_ref}
    )

    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref)
    input_refs = validator.validate(**input_refs)
    # check properties
    assert ptr_set(input_refs.refs) == ptr_set([x_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict({"x": x_ref})

    # create input refs instance
    input_refs = CustomInputRefs(y=y_ref)
    input_refs = validator.validate(**input_refs)
    # check properties
    assert ptr_set(input_refs.refs) == ptr_set([y_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict({"y": y_ref})

    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(MagicMock())]
        y: NotRequired[Annotated[FeatureRef, FeatureValidator(MagicMock())]]

    # create validator instance from custom input refs type
    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)

    assert validator.required_keys == {"x"}
    assert validator.optional_keys == {"y"}

    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref, y=y_ref)
    input_refs = validator.validate(**input_refs)
    # check properties
    assert ptr_set(input_refs.refs) == ptr_set([x_ref, y_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict(
        {"x": x_ref, "y": y_ref}
    )

    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref)
    input_refs = validator.validate(**input_refs)
    # check properties
    assert ptr_set(input_refs.refs) == ptr_set([x_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict({"x": x_ref})


def test_missing_and_unexpected_input_arguments():
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(MagicMock())]
        y: Annotated[FeatureRef, FeatureValidator(MagicMock())]

    # create validator instance from custom input refs type
    validator = InputRefsValidator(BaseConfig(), CustomInputRefs)
    # create dummy input ref
    x_ref = FeatureRef(
        key_=tuple(), node_id_="", flow_=MagicMock(), feature_=Value("int32")
    )
    # no inputs provided
    with pytest.raises(TypeError):
        validator.validate(x=x_ref)
    # unexpected inputs provided
    with pytest.raises(TypeError):
        validator.validate(x=x_ref, y=x_ref, z=x_ref)

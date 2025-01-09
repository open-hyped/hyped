from typing import Annotated, Any, TypeVar
from unittest.mock import ANY, MagicMock

import pydantic
import pytest

from hyped.common._pydantic import TypeAdapterWithArbitraryTypesAllowed
from hyped.core.features.dtypes import Float32Type, Int32Type, SequenceType, StringType
from hyped.core.features.features import Int32Feature, SequenceFeature, StringFeature
from hyped.core.features.reference import ForwardReference
from hyped.core.features.session import ValidationSession
from hyped.core.features.validators import FeatureResolver, FeatureValidator, Len, MatchFeatures
from hyped.typing import Float, Int, Sequence, String


class TestFeatureValidator:
    def test_case(self) -> None:
        validator_fn = MagicMock(side_effect=lambda x, c, s: x)

        context = {"config": MagicMock(), "session": MagicMock()}

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int, FeatureValidator(validator_fn)],
        )
        adapter.validate_python(ForwardReference(), context=context)

        validator_fn.assert_called_once_with(ANY, context["config"], context["session"])

    def test_error_on_missing_context(self) -> None:
        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int, FeatureValidator(MagicMock())],
        )

        with pytest.raises(RuntimeError):
            adapter.validate_python(ForwardReference())


class TestFeatureResolver:
    def test_resolve_union(self) -> None:
        context = {"config": MagicMock(), "inputs": {}, "session": MagicMock(), "typevars": {}}

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | Float, FeatureResolver(lambda c, i, s: Int)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        feature = adapter.validate_python(ForwardReference(), context=context)

        assert isinstance(feature, Int)

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | Float, FeatureResolver(lambda c, i, s: Float)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        feature = adapter.validate_python(ForwardReference(), context=context)

        assert isinstance(feature, Float)

    def test_resolve_union_with_typevar(self) -> None:
        T = TypeVar("T")

        context = {
            "config": MagicMock(),
            "inputs": {},
            "session": MagicMock(),
            "typevars": {T: Float32Type},
        }

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | T, FeatureResolver(lambda c, i, s: Int)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        feature = adapter.validate_python(ForwardReference(), context=context)

        assert isinstance(feature, Int)

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | Float, FeatureResolver(lambda c, i, s: T)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        feature = adapter.validate_python(ForwardReference(), context=context)

        assert isinstance(feature, Float)

    def test_resolve_union_with_generic(self) -> None:
        T = TypeVar("T")

        context = {
            "config": MagicMock(),
            "inputs": {},
            "session": MagicMock(),
            "typevars": {T: Float32Type},
        }

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | Sequence[T], FeatureResolver(lambda c, i, s: Sequence[T])],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        feature = adapter.validate_python(ForwardReference(), context=context)

        assert isinstance(feature, SequenceFeature)
        assert feature.dtype.value_type == Float32Type

    def test_resolve_with_annotation(self) -> None:
        validator_fn = MagicMock(side_effect=lambda x, c, s: x)

        context = {"config": MagicMock(), "inputs": {}, "session": MagicMock(), "typevars": {}}

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[
                Int, FeatureResolver(lambda c, i, s: Annotated[Int, FeatureValidator(validator_fn)])
            ],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        adapter.validate_python(ForwardReference(), context=context)

        validator_fn.assert_called_once_with(ANY, context["config"], context["session"])

    def test_error_on_missing_context(self) -> None:
        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | Float, FeatureResolver(lambda c, i, s: Int)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )

        with pytest.raises(RuntimeError):
            adapter.validate_python(ForwardReference())


class TestLenValidator:
    @pytest.fixture
    def session(self) -> ValidationSession:
        return ValidationSession()

    @pytest.fixture
    def context(self, session: ValidationSession) -> dict[str, Any]:
        return {"config": MagicMock(), "session": session}

    def test_check_len(self, context: dict[str, Any]) -> None:
        inst = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type, length=2)))

        # adapter checking for sequence of length 2
        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Sequence[Int], Len(2)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        adapter.validate_python(inst, context=context)

        # adapter checking for sequence of length 3
        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Sequence[Int], Len(3)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        with pytest.raises(pydantic.ValidationError):
            adapter.validate_python(inst, context=context)

    def test_check_len_strict(self, context: dict[str, Any]) -> None:
        inst = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type)))

        # adapter checking for sequence of length 2
        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Sequence[Int], Len(2, strict=True)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )

        # instance length is undefined and validator is checking for length 2 in strict mode
        # expected to throw validation error
        with pytest.raises(pydantic.ValidationError):
            adapter.validate_python(inst, context=context)

    def test_simple_set_len(self, context: dict[str, Any]) -> None:
        # length undefined
        inst = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type)))

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Sequence[Int], Len(2)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        # instance length is undefined and validator is checking for length 2 in non-strict mode
        # expected to specify the sequence length of the instance
        inst: SequenceFeature = adapter.validate_python(inst, context=context)

        assert inst.dtype.length == 2

    def test_capture_length(self, context: dict[str, Any], session: ValidationSession) -> None:
        l = Len()

        class Model(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
            # fields should have dynamic but same length
            fieldA: Annotated[Sequence[Int], l]
            fieldB: Annotated[Sequence[Int], l]

        instA = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type, length=4)))
        instB = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type, length=4)))

        with session:
            # both features have the same length
            Model.model_validate({"fieldA": instA, "fieldB": instB}, context=context)

        instA = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type, length=2)))
        instB = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type, length=4)))
        # A has different length as B
        with session:
            with pytest.raises(pydantic.ValidationError):
                Model.model_validate({"fieldA": instA, "fieldB": instB}, context=context)

        instA = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type, length=2)))
        instB = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type)))
        # length of B is undefined and length is in non-strict mode
        # length of B should be specified
        with session:
            inst = Model.model_validate({"fieldA": instA, "fieldB": instB}, context=context)
        assert inst.fieldB.dtype.length == 2

    def test_compute_length(self, context: dict[str, Any], session: ValidationSession) -> None:
        def compute_len_fn(
            config: MagicMock, captured_length: int | None, session: ValidationSession
        ) -> int:
            return 20

        # length undefined
        inst = SequenceFeature(ForwardReference(dtype=SequenceType(Int32Type)))

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Sequence[Int], Len(compute_len_fn)],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        # instance length is undefined and validator is checking for length 2 in non-strict mode
        # expected to specify the sequence length of the instance
        inst: SequenceFeature = adapter.validate_python(inst, context=context)

        assert inst.dtype.length == 20


class TestMatchFeaturesValidator:
    @pytest.fixture
    def session(self) -> ValidationSession:
        return ValidationSession()

    @pytest.fixture
    def context(self, session: ValidationSession) -> dict[str, Any]:
        return {"config": MagicMock(), "session": session}

    def test_check_feature_match(self, context: dict[str, Any]) -> None:
        inst1 = Int32Feature(ForwardReference(dtype=Int32Type))
        inst2 = Int32Feature(ForwardReference(dtype=Int32Type))

        M = MatchFeatures()
        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | String, M],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )

        adapter.validate_python(inst1, context=context)
        adapter.validate_python(inst2, context=context)

    def test_check_feature_mismatch(self, context: dict[str, Any]) -> None:
        int_inst = Int32Feature(ForwardReference(dtype=Int32Type))
        str_inst = StringFeature(ForwardReference(dtype=StringType))

        M = MatchFeatures()
        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | String, M],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )

        with pytest.raises(TypeError):
            adapter.validate_python(int_inst, context=context)
            adapter.validate_python(str_inst, context=context)

    def test_check_feature_unrelated(self, context: dict[str, Any]) -> None:
        int_inst = Int32Feature(ForwardReference(dtype=Int32Type))
        str_inst = StringFeature(ForwardReference(dtype=StringType))

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | String, MatchFeatures()],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        adapter.validate_python(int_inst, context=context)

        adapter = TypeAdapterWithArbitraryTypesAllowed(
            Annotated[Int | String, MatchFeatures()],
            config=pydantic.ConfigDict(arbitrary_types_allowed=True),
        )
        adapter.validate_python(str_inst, context=context)

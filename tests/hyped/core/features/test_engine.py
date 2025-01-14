import inspect
from typing import TypeVar
from unittest.mock import MagicMock, patch

import pytest

from hyped.core.features.dtypes import BoolType, Int16Type
from hyped.core.features.engine import FeatureEngine
from hyped.core.features.features import BoolFeature, Int16Feature, PrimitiveFeature
from hyped.core.features.reference import ConcreteReference, ForwardReference


class TestFeatureEngine:
    def test_validate_signature(self) -> None:
        def fn(a: BoolFeature, b: Int16Feature) -> Int16Feature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        engine.validate_signature()

        def fn(a: BoolFeature, b) -> Int16Feature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        # missing annotation
        with pytest.raises(TypeError):
            engine.validate_signature()

        def fn(a: BoolFeature, b: Int16Feature):
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        with pytest.raises(TypeError):
            engine.validate_signature()

    def test_validate_arguments(self) -> None:
        a = PrimitiveFeature(ForwardReference(BoolType))
        b = PrimitiveFeature(ForwardReference(Int16Type))

        def fn(a: BoolFeature, b: Int16Feature) -> Int16Feature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))

        engine.validate_arguments(a, b)
        engine.validate_arguments(a, b=b)
        engine.validate_arguments(a=a, b=b)

        with pytest.raises(TypeError):
            # missing argument b
            engine.validate_arguments(a)

    def test_validate_keyword_arguments(self) -> None:
        a = PrimitiveFeature(ForwardReference(BoolType))
        b = PrimitiveFeature(ForwardReference(Int16Type))

        def fn(**kwargs: BoolFeature) -> Int16Feature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        engine.validate_arguments(x=a, y=a, z=a)

        with pytest.raises(TypeError):
            engine.validate_arguments(x=b, y=b, z=a)

    def test_validate_generic_arguments(self) -> None:
        a = PrimitiveFeature(ForwardReference(BoolType))
        b = PrimitiveFeature(ForwardReference(Int16Type))

        T = TypeVar("T")

        def fn(a: T, b: T) -> Int16Feature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        # arguments are expected to be of the same type
        engine.validate_arguments(a=a, b=a)
        engine.validate_arguments(a=b, b=b)

        with patch("hyped.core.features.engine.common_dtype") as mock_common_dtype:
            engine.validate_arguments(a=a, b=b)
            # build typevar mapping
            _ = engine.typevar_register.typevar_mapping
            # make sure typevar was resolved to common type
            mock_common_dtype.assert_called_once()
            assert set(mock_common_dtype.mock_calls[0].args) == {BoolType, Int16Type}

    def test_build_return_feature(self) -> None:
        a = PrimitiveFeature(ForwardReference(BoolType))
        b = PrimitiveFeature(ForwardReference(Int16Type))

        def fn(a: BoolFeature, b: Int16Feature) -> Int16Feature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        engine.validate_arguments(a, b)

        return_feature = engine.build_return_feature(MagicMock())
        assert return_feature.dtype == Int16Type

    def test_build_generic_return_feature(self) -> None:
        a = PrimitiveFeature(ForwardReference(BoolType))
        b = PrimitiveFeature(ForwardReference(Int16Type))

        T = TypeVar("T")

        def fn(a: T, b: T) -> T:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))

        engine.validate_arguments(a, a)
        return_feature = engine.build_return_feature(MagicMock())
        assert return_feature.dtype == BoolType

        engine.validate_arguments(b, b)
        return_feature = engine.build_return_feature(MagicMock())
        assert return_feature.dtype == Int16Type

    def test_get_references_and_consts(self) -> None:
        def fn(a: BoolFeature | bool, b: BoolFeature | bool) -> BoolFeature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        refs, vals, dtypes = engine.get_features_and_objects(a=True, b=False)

        assert len(refs) == 0
        assert vals == {"a": True, "b": False}
        assert dtypes == {"a": BoolType, "b": BoolType}

        a = PrimitiveFeature(ConcreteReference(MagicMock(), MagicMock(), MagicMock()))

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        refs, vals, dtypes = engine.get_features_and_objects(a=True, b=a)

        assert refs == {"b": a}
        assert vals == {"a": True}
        assert dtypes == {"a": BoolType}

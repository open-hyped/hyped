import inspect
from typing import TypeVar
from unittest.mock import MagicMock

import pytest

from hyped.core.features.engine import FeatureEngine
from hyped.core.features.features import BoolFeature, Int16Feature, PrimitiveFeature
from hyped.core.features.reference import Reference
from hyped.core.features.types import BoolType, Int16Type


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
        a = PrimitiveFeature(Reference(), BoolType)
        b = PrimitiveFeature(Reference(), Int16Type)

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
        a = PrimitiveFeature(Reference(), BoolType)
        b = PrimitiveFeature(Reference(), Int16Type)

        def fn(**kwargs: BoolFeature) -> Int16Feature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        engine.validate_arguments(x=a, y=a, z=a)

        with pytest.raises(TypeError):
            engine.validate_arguments(x=b, y=b, z=a)

    def test_validate_generic_arguments(self) -> None:
        a = PrimitiveFeature(Reference(), BoolType)
        b = PrimitiveFeature(Reference(), Int16Type)

        T = TypeVar("T")

        def fn(a: T, b: T) -> Int16Feature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        # arguments are expected to be of the same type
        engine.validate_arguments(a=a, b=a)
        engine.validate_arguments(a=b, b=b)

        with pytest.raises(TypeError):
            # arguments are of different types
            engine.validate_arguments(a=a, b=b)

    def test_build_return_feature(self) -> None:
        a = PrimitiveFeature(Reference(), BoolType)
        b = PrimitiveFeature(Reference(), Int16Type)

        def fn(a: BoolFeature, b: Int16Feature) -> Int16Feature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        engine.validate_arguments(a, b)

        return_feature = engine.build_return_feature(Reference(), MagicMock())
        assert return_feature.dtype == Int16Type

    def test_build_generic_return_feature(self) -> None:
        a = PrimitiveFeature(Reference(), BoolType)
        b = PrimitiveFeature(Reference(), Int16Type)

        T = TypeVar("T")

        def fn(a: T, b: T) -> T:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))

        engine.validate_arguments(a, a)
        return_feature = engine.build_return_feature(Reference(), MagicMock())
        assert return_feature.dtype == BoolType

        engine.validate_arguments(b, b)
        return_feature = engine.build_return_feature(Reference(), MagicMock())
        assert return_feature.dtype == Int16Type

    def test_get_references_and_consts(self) -> None:
        def fn(a: BoolFeature | bool, b: BoolFeature | bool) -> BoolFeature:
            ...

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        refs, vals, dtypes = engine.get_references_and_objects(a=True, b=False)

        assert len(refs) == 0
        assert vals == {"a": True, "b": False}
        assert dtypes == {"a": BoolType, "b": BoolType}

        a = PrimitiveFeature(Reference(), BoolType)

        engine = FeatureEngine("name", MagicMock(), inspect.signature(fn))
        refs, vals, dtypes = engine.get_references_and_objects(a=True, b=a)

        assert refs == {"b": a.ref}
        assert vals == {"a": True}
        assert dtypes == {"a": BoolType}

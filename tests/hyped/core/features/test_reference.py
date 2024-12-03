import pyarrow as pa
import pytest

from hyped.core.features.reference import FeatureKey
from hyped.core.features.types import BoolType, MappingType, SequenceType, Type


class TestFeatureKey:
    def test_basics(self) -> None:
        # initialize from tuple
        assert FeatureKey("a") == FeatureKey(("a",))
        # equal operator
        assert FeatureKey("a") == FeatureKey("a")
        assert FeatureKey("a") != FeatureKey("b")
        # hash
        assert hash(FeatureKey("a")) == hash(FeatureKey("a"))
        assert hash(FeatureKey("a")) != hash(FeatureKey("b"))
        assert hash(FeatureKey(slice(None))) == hash(FeatureKey(slice(None)))
        assert hash(FeatureKey(slice(None))) != hash(FeatureKey(slice(2)))
        # indexing
        assert FeatureKey("a")[0] == "a"
        assert FeatureKey("a")[:] == FeatureKey("a")

        with pytest.raises(TypeError):
            # invalid key type
            FeatureKey(1.4)

    @pytest.mark.parametrize(
        "dtype,key,expected_dtype",
        [
            (BoolType, FeatureKey(), BoolType),
            (SequenceType(BoolType), FeatureKey(0), BoolType),
            (MappingType.construct({"fieldA": BoolType}), FeatureKey("fieldA"), BoolType),
            (SequenceType(BoolType), FeatureKey(slice(None)), SequenceType(BoolType)),
            (
                MappingType.construct({"fieldA": SequenceType(BoolType)}),
                FeatureKey("fieldA"),
                SequenceType(BoolType),
            ),
            (
                MappingType.construct({"fieldA": SequenceType(BoolType)}),
                FeatureKey("fieldA", 0),
                BoolType,
            ),
        ],
    )
    def test_index_dtype(self, dtype: Type, key: FeatureKey, expected_dtype: Type) -> None:
        assert key.index_dtype(dtype) == expected_dtype

    def test_index_dtype_key(self):
        with pytest.raises(TypeError):
            FeatureKey(0).index_dtype(BoolType)

        with pytest.raises(TypeError):
            FeatureKey(0).index_dtype(MappingType.construct({"fieldA": BoolType}))

        with pytest.raises(TypeError):
            FeatureKey(slice(None)).index_dtype(MappingType.construct({"fieldA": BoolType}))

        with pytest.raises(TypeError):
            FeatureKey("fieldA").index_dtype(SequenceType(BoolType))

    @pytest.mark.parametrize(
        "array,key,expected_array",
        [
            (pa.array([0, 2], type=pa.int32()), FeatureKey(), pa.array([0, 2], type=pa.int32())),
            (
                pa.array([[0, 0], [2, 2], [4, 4]], type=pa.list_(pa.int32())),
                FeatureKey(0),
                pa.array([0, 2, 4], type=pa.int32()),
            ),
            (
                pa.array([[0, 1], [2, 3], [4, 5]], type=pa.list_(pa.int32())),
                FeatureKey(1),
                pa.array([1, 3, 5], type=pa.int32()),
            ),
            (
                pa.array([[0, 1], [2, 3], [4, 5]], type=pa.list_(pa.int32(), 2)),
                FeatureKey(-1),
                pa.array([1, 3, 5], type=pa.int32()),
            ),
            (
                pa.array([[0, 1], [2, 3], [4, 5]], type=pa.list_(pa.int32())),
                FeatureKey(slice(1)),
                pa.array([[0], [2], [4]], type=pa.list_(pa.int32())),
            ),
            (
                pa.array([[0, 1], [2, 3], [4, 5]], type=pa.list_(pa.int32(), 2)),
                FeatureKey(slice(-1)),
                pa.array([[0], [2], [4]], type=pa.list_(pa.int32())),
            ),
            (
                pa.array([{"fieldA": 0}, {"fieldA": 1}], type=pa.struct([("fieldA", pa.int32())])),
                FeatureKey("fieldA"),
                pa.array([0, 1], type=pa.int32()),
            ),
            (
                pa.array(
                    [{"fieldA": [0, 1, 2]}, {"fieldA": [3, 4, 5]}],
                    type=pa.struct([("fieldA", pa.list_(pa.int32()))]),
                ),
                FeatureKey("fieldA"),
                pa.array([[0, 1, 2], [3, 4, 5]], type=pa.list_(pa.int32())),
            ),
            (
                pa.array(
                    [{"fieldA": [0, 1, 2]}, {"fieldA": [3, 4, 5]}],
                    type=pa.struct([("fieldA", pa.list_(pa.int32()))]),
                ),
                FeatureKey("fieldA", 1),
                pa.array([1, 4], type=pa.int32()),
            ),
            (
                pa.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], type=pa.list_(pa.list_(pa.int32()))),
                FeatureKey(0),
                pa.array([[0, 1], [4, 5]], type=pa.list_(pa.int32())),
            ),
            (
                pa.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], type=pa.list_(pa.list_(pa.int32()))),
                FeatureKey(0, 0),
                pa.array([0, 4], type=pa.int32()),
            ),
            (
                pa.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], type=pa.list_(pa.list_(pa.int32()))),
                FeatureKey(slice(None), 0),
                pa.array([[0, 2], [4, 6]], type=pa.list_(pa.int32())),
            ),
        ],
    )
    def test_index_array(self, array: pa.Array, key: FeatureKey, expected_array: pa.Array) -> None:
        assert key.index_array(array).to_pylist() == expected_array.to_pylist()

    def test_index_array_errors(self):
        arr = pa.array([[0, 1], [2, 3]], type=pa.list_(pa.int32()))

        with pytest.raises(IndexError):
            # negative index for list of unspecified length
            FeatureKey(-1).index_array(arr)

        with pytest.raises(IndexError):
            # negative slicing for list of unspecified length
            FeatureKey(slice(-1)).index_array(arr)

        with pytest.raises(TypeError):
            FeatureKey("a").index_array(arr)

        arr = pa.array([{"fieldA": 0}], type=pa.struct([("fieldA", pa.int32())]))

        with pytest.raises(TypeError):
            FeatureKey(0).index_array(arr)

        with pytest.raises(TypeError):
            FeatureKey(slice(None)).index_array(arr)

import pyarrow as pa
import pytest

from hyped.common._pyarrow import flatten_list_array, unflatten_list_array


@pytest.mark.parametrize(
    "array",
    [
        # Simple ListArray
        pa.array([[1, 2, 3], [4, 5], [], [6, 7, 8, 9]]),
        pa.array([[b"a", b"b"], [b"c"]], type=pa.list_(pa.binary())),
        pa.array([[True, False], [False]], type=pa.list_(pa.bool_())),
        # FixedSizeListArray
        pa.FixedSizeListArray.from_arrays(pa.array([1, 2, 3, 4, 5, 6]), 2),
        # ChunkedArray of ListArrays
        pa.chunked_array([pa.array([[1, 2], [3]]), pa.array([[4, 5, 6]])]),
        # ChunkedArray of FixedSizeListArrays
        pa.chunked_array(
            [
                pa.FixedSizeListArray.from_arrays(pa.array([1, 2, 3, 4]), 2),
                pa.FixedSizeListArray.from_arrays(pa.array([5, 6]), 2),
            ]
        ),
    ],
)
def test_flatten_unflatten(array):
    # Flatten the array
    flattened, offsets = flatten_list_array(array)

    # Unflatten the array
    unflattened = unflatten_list_array(flattened, offsets)

    # Check that the unflattened array matches the original
    assert unflattened.to_pylist() == array.to_pylist()

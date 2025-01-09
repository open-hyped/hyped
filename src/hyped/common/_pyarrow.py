"""PyArrow helper functions."""
import pyarrow as pa


def flatten_list_array(
    array: pa.ListArray | pa.FixedSizeListArray | pa.ChunkedArray,
) -> tuple[pa.Array, int | pa.Int32Array | list[pa.Int32Array]]:
    """Flatten a PyArrow ListArray, FixedSizeListArray, or ChunkedArray into a single flat array.

    Args:
        array (pa.ListArray | pa.FixedSizeListArray | pa.ChunkedArray): The input array to flatten.
            This can be a ListArray, FixedSizeListArray, or a ChunkedArray containing such arrays.

    Returns:
        tuple: A tuple containing:
            - pa.Array: The flattened array of all elements.
            - int | pa.Int32Array | list[pa.Int32Array]: The metadata needed to reconstruct the
              original structure:
                - If `array` is a FixedSizeListArray, returns the fixed size (int).
                - If `array` is a ListArray, returns a single Int32Array of offsets.
                - If `array` is a ChunkedArray, returns a list of Int32Arrays (one for each chunk).
    """
    if isinstance(array, pa.ChunkedArray):
        flattened_chunks = []
        chunk_offsets = []
        for chunk in array.chunks:
            flat_chunk, offsets = flatten_list_array(chunk)
            flattened_chunks.append(flat_chunk)
            chunk_offsets.append(offsets)

        if len(chunk_offsets) > 0 and isinstance(chunk_offsets[0], int):
            assert all(x == chunk_offsets[0] for x in chunk_offsets[1:])
            chunk_offsets = chunk_offsets[0]

        flattened_combined = pa.concat_arrays(flattened_chunks)
        return flattened_combined, chunk_offsets

    elif pa.types.is_fixed_size_list(array.type):
        return array.flatten(), array.type.list_size

    elif pa.types.is_list(array.type):
        return array.flatten(), array.offsets


def unflatten_list_array(
    array: pa.Array,
    offsets: int | pa.Int32Array | list[pa.Int32Array],
) -> pa.ListArray | pa.FixedSizeListArray | pa.ChunkedArray:
    """Reconstruct a PyArrow ListArray, FixedSizeListArray, or ChunkedArray from a flat array.

    Args:
        array (pa.Array): The flat array of elements to reconstruct.
        offsets (int | pa.Int32Array | list[pa.Int32Array]): Metadata to reconstruct the original
            structure:
                - If reconstructing a FixedSizeListArray, provide the fixed size (int).
                - If reconstructing a ListArray, provide a single Int32Array of offsets.
                - If reconstructing a ChunkedArray, provide a list of Int32Arrays
                  (one for each chunk).

    Returns:
        pa.ListArray | pa.FixedSizeListArray | pa.ChunkedArray: The reconstructed structure. This
        will be a ListArray, FixedSizeListArray, or ChunkedArray of these types, depending on the
        input.
    """
    if isinstance(offsets, int):
        return pa.FixedSizeListArray.from_arrays(array, offsets)

    elif isinstance(offsets, pa.Array) and pa.types.is_integer(offsets.type):
        return pa.ListArray.from_arrays(offsets, array)

    elif isinstance(offsets, list):
        assert len(offsets) > 0

        chunks = []
        current_offset = 0
        for chunk_offsets in offsets:
            # get the flat chunk
            size = chunk_offsets[-1].as_py()
            flat_chunk = array[current_offset : current_offset + size]
            # unflatten the chunk
            unflattened_chunk = unflatten_list_array(flat_chunk, chunk_offsets)
            chunks.append(unflattened_chunk)
            # progress the offset
            current_offset += size

        return pa.chunked_array(chunks)

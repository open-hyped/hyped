"""Module providing the :class:`ArrowDatasetWriter` class.

The :class:`ArrowDatasetWriter` class writes dataset samples to individual Arrow shard files,
with each worker writing a separate shard. 
"""

import pyarrow as pa

from hyped.common._arrow import convert_features_to_arrow_schema
from hyped.common._worker import get_worker_info
from hyped.common.typing import Sample

from .base import BaseDatasetWriter


class ArrowDatasetWriter(BaseDatasetWriter):
    """A dataset writer for saving data in Arrow format.

    This class is responsible for writing dataset samples to Arrow files,
    with each worker writing its own shard. The writer converts dataset
    features to an Arrow schema and uses PyArrow for efficient serialization
    of the data.

    The output is compatible with the Hugging Face Datasets library's :func:`load_from_disk`
    function, making it easy to reload the saved dataset:

    .. code-block:: python

        writer = ArrowDatasetWriter(save_dir="./data")
        writer.write(ds)

        ds = datasets.load_from_disk("./data")
    """

    def initialize(self) -> None:
        """
        Initialize the Arrow writer.

        This method sets up the Arrow file writer for the current worker by:
        - Creating a shard file named :code:`shard-<rank>.arrow` for the current worker.
        - Converting dataset features to an Arrow schema.
        - Initializing the Arrow writer to stream data in Arrow format.

        The working directory is set to the save directory during the write process.
        """
        info = get_worker_info()
        # open shard file
        info.ctx.file_path = f"shard-{info.rank}.arrow"
        info.ctx.file = open(info.ctx.file_path, "wb")
        # build arrow schema from dataset features
        features = info.ctx.dataset_info.features
        info.ctx.schema = convert_features_to_arrow_schema(features)
        # create arrow writer
        info.ctx.writer = pa.ipc.new_stream(sink=info.ctx.file, schema=info.ctx.schema)

    def write_sample(self, sample: Sample) -> None:
        """Write a single sample to the Arrow shard.

        Args:
            sample (Sample): The sample to be written to the Arrow file.

        This method writes the sample as a record batch to the worker's shard file
        using the Arrow schema initialized during the :code:`initialize` phase.
        """
        info = get_worker_info()
        # write sample to file
        sample = pa.RecordBatch.from_pylist([sample], schema=info.ctx.schema)
        info.ctx.writer.write(sample)

    def finalize(self) -> None:
        """Finalize the writing process.

        This method closes the Arrow writer and the corresponding shard file.
        It ensures all written samples are properly flushed to disk.
        """
        # close writer and file
        info = get_worker_info()
        info.ctx.writer.close()
        info.ctx.file.close()

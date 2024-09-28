"""This module defines a custom dataset for reading lines from files.

The :class:`LinesDataset` class processes files line-by-line, either as raw bytes or strings,
depending on the specified encoding. The dataset can handle multiple files, and lines are returned
in chunks to optimize memory usage for large datasets. Users can configure file encoding, error
handling, and chunk sizes via the :class:`LinesDatasetConfig` class.
"""
import io
from dataclasses import dataclass
from itertools import chain, count

import datasets
import pyarrow as pa
from datasets.utils.file_utils import readline


@dataclass
class LinesDatasetConfig(datasets.BuilderConfig):
    """Configuration for the LinesDataset.

    The attributes of the configuration are typically set by providing
    them as keyword arguments to the :func:`datasets.load_dataset` function.
    """

    encoding: None | str = None
    """The encoding to use when reading the files.
    
    If None, lines are loaded as bytes. Defaults to None.
    """

    encoding_errors: None | str = None
    """Specifies how to handle encoding errors (e.g., 'strict', 'ignore')."""

    chunksize: int = 10 << 20  # 10MB
    """The size of chunks in bytes to read from each file at a time.
    
    Defaults to 10MB.
    """


class LinesDataset(datasets.ArrowBasedBuilder):
    """A dataset builder that loads lines from files.

    The dataset loads each line as an example, with support for both binary and string encodings.
    It generates tables of lines read from chunks of the input files, optimizing memory usage for
    large datasets.

    Typically used by call to :func:`datasets.load_dataset with appropriate keyword arguments
    (see :class:`LinesDatasetConfig` for defails).

    .. code-block:: python

        datasets.load_dataset('hyped.io.datasets.lines', **kwargs)
    """

    BUILDER_CONFIG_CLASS = LinesDatasetConfig

    def _info(self):
        """Provides dataset metadata.

        Returns:
            datasets.DatasetInfo: Information about the dataset, including its features.
        """
        return datasets.DatasetInfo(
            features=datasets.Features(
                {"line": datasets.Value("binary" if self.config.encoding is None else "string")}
            )
        )

    def _split_generators(self, dl_manager):
        """Defines dataset splits and handles file downloading or extraction.

        Args:
            dl_manager (DownloadManager): The download manager used to download and extract the
                files.

        Returns:
            List[datasets.SplitGenerator]: A list of SplitGenerator objects specifying the data
            splits.

        Raises:
            ValueError: If no data files are provided in the configuration.
        """
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got "
                f"data_files={self.config.data_files}"
            )
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download_and_extract(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_tables(self, files):
        """Yields batches of data from the input files as :code:`pyarrow` Tables.

        Args:
            files (iterable): An iterable of file paths for each split.

        Yields:
            Tuple[Tuple[int, int], pa.Table]: A tuple containing file index and chunk index,
            and the corresponding pyarrow Table with lines as data.
        """

        newline = b"\n" if self.config.encoding is None else "\n"
        mode = "rb" if self.config.encoding is None else "r"

        for fidx, fpath in enumerate(chain.from_iterable(files)):
            with open(
                fpath,
                mode,
                encoding=self.config.encoding,
                errors=self.config.encoding_errors,
            ) as f:
                # check if the file object supports readline
                try:
                    f.readline()
                    has_readline = True
                except (AttributeError, io.UnsupportedOperation):
                    has_readline = False

                # go back to start of the file
                f.seek(0)

                for chunk_idx in count():
                    # read chunk of the file
                    chunk = f.read(self.config.chunksize)
                    # nothing to read anymore
                    if len(chunk) == 0:
                        break

                    # finish current and seperate lines
                    chunk += f.readline() if has_readline else readline(f)
                    lines = chunk.strip(newline).split(newline)
                    # pack into table
                    yield (fidx, chunk_idx), pa.Table.from_pydict({"line": lines})

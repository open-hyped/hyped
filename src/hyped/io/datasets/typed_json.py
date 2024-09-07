"""Typed JSON Dataset Generator."""
import io
from dataclasses import dataclass
from itertools import chain, count

import datasets
import pyarrow as pa
from datasets.packaged_modules.json.json import Json, JsonConfig
from datasets.utils.file_utils import readline

from hyped import DataFlow
from hyped.common.typing import Batch
from hyped.processors import JsonParser


@dataclass
class TypedJsonDatasetConfig(JsonConfig):
    """Typed Json Dataset Configuration.

    Matches the huggingface datasets json dataset implementation.
    Please refer to the huggingface documentation for more information.

    The attributes of the configuration are typically set by providing
    them as keyword arguments to the :func:`datasets.load_dataset` function.
    """

    # features are required and not
    # optional as in the base json cofig
    features: datasets.Features = None
    """Dataset features, required for type checking."""

    def __post_init__(self) -> None:
        """Build pydantic models from feature description."""
        if self.features is None:
            raise ValueError(
                "No dataset features provided. Please specify the expeted "
                "dataset features for type checking."
            )

        payload = datasets.Features({"payload": datasets.Value("string")})
        self._flow = DataFlow(payload)
        # add the json parser
        parser = JsonParser(scheme=self.features)
        obj = parser.call(payload=self._flow.src_features.payload)
        # build the flow
        self._flow, _ = self._flow.build(collect=obj)

    def __getstate__(self):
        """Prepare the object state for serialization.

        This method removes the :code:`_flow` attribute from the state dictionary
        to prevent issues during serialization. The :code:`_flow` is rebuilt in
        the :func:`__post_init__` method when deserializing.

        Returns:
            dict: The object's state without the :code:`_flow` attribute.
        """
        state = self.__dict__.copy()
        _ = state.pop("_flow")
        return state

    def __setstate__(self, state):
        """Restore the object state after deserialization.

        This method restores the object's state and rebuilds the :code:`_flow`
        by calling the :func:`__post_init__` method.

        Args:
            state (dict): The deserialized object state.
        """
        self.__dict__ = state
        self.__post_init__()


class TypedJsonDataset(Json):
    """Typed Json Dataset.

    Typically used by call to :func:`datasets.load_dataset with appropriate
    keyword arguments (see :class:`TypedJsonDatasetConfig` for defails)

    ```python
    datasets.load_dataset('hyped.io.datasets.typed_json', **kwargs)
    ```
    """

    BUILDER_CONFIG_CLASS = TypedJsonDatasetConfig

    def _generate_tables(self, files):
        for fidx, fpath in enumerate(chain.from_iterable(files)):
            if self.config.field is not None:
                raise NotImplementedError()

            else:
                with open(
                    fpath,
                    "r",
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

                        # finish current line and remove trailing newline
                        chunk += f.readline() if has_readline else readline(f)
                        chunk = Batch(payload=chunk.strip().split("\n"))
                        # parse chunk using dataflow flow
                        index = range(len(chunk["payload"]))
                        chunk = self.config._flow.batch_process(batch=chunk, index=index, rank=0)
                        # yield the parsed chunk
                        yield (fidx, chunk_idx), pa.Table.from_pylist(chunk["obj"])

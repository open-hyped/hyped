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


class TypedJsonDataset(Json):
    """Typed Json Dataset.

    Typically used by call to :func:`datasets.load_dataset with appropriate
    keyword arguments (see :class:`TypedJsonDatasetConfig` for defails)

    ```python
    datasets.load_dataset('hyped.io.datasets.typed_json', **kwargs)
    ```
    """

    BUILDER_CONFIG_CLASS = TypedJsonDatasetConfig

    def _build_flow(self) -> DataFlow:
        payload = datasets.Features({"payload": datasets.Value("string")})
        flow = DataFlow(payload)
        # add the json parser
        parser = JsonParser(scheme=self.config.features)
        obj = parser.call(payload=flow.src_features.payload)
        # build the flow
        flow, _ = flow.build(collect=obj)

        return flow

    def _generate_tables(self, files):
        flow = self._build_flow()

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

                        # finish current line
                        chunk += f.readline() if has_readline else readline(f)
                        chunk = Batch(payload=chunk.strip().split("\n"))
                        # parse chunk using dataflow flow
                        index = range(len(chunk["payload"]))
                        chunk = flow.batch_process(batch=chunk, index=index, rank=0)
                        # yield the parsed chunk
                        yield (fidx, chunk_idx), pa.Table.from_pylist(chunk["obj"])

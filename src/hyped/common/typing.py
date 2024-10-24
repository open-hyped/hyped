"""This module defines common type aliases used throughout the project.

Type aliases are used to improve code readability and maintainability by providing descriptive
names for commonly used complex data structures or concepts within the codebase. These aliases
allow developers to quickly understand the intent and structure of data being passed around
without needing to parse through detailed type definitions.

The type aliases in this module cover a variety of concepts, including dataset samples, batches,
indices, partitions in data flow graphs, and more. They help to standardize the type annotations
used across different parts of the project, making the code more consistent and easier to manage.
"""


from typing import Any

import pyarrow as pa
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from typing_extensions import TypeAlias

ArrowType: TypeAlias = pa.lib.DataType


Sample: TypeAlias = dict[str, Any]
"""A sample of the dataset.

Represents a single data point or record in the dataset structured as a dictionary.
The keys are strings representing feature names, and the values can be of any type
(e.g., integers, floats, strings, or more complex data structures).
"""

Batch: TypeAlias = dict[str, pa.Array]
"""A batch of samples from the dataset.

Represents a collection of samples, structured as a dictionary where the keys are feature names,
and the values are arrow arrays containing data points for each feature across multiple samples.
"""

Index: TypeAlias = int
"""An index, usually corresponding to a sample.

Represents a single integer that refers to a specific sample within the dataset. This is often used
to retrieve or reference a particular sample from a dataset.
"""

IndexList: TypeAlias = list[Index]
"""A list of dataset indices, usually corresponding to a batch.

Contains integer indices that refer to specific samples within the dataset.
This is typically used to track which samples are included in a particular
batch or subset of the dataset.
"""

Rank: TypeAlias = int
"""The multi-processing execution rank.

In distributed or parallel computing, the rank is an integer identifier for a process.
This type alias is typically used to represent the rank of a process in a multi-processing
or distributed environment, where each process is assigned a unique rank.
"""

Aggregate: TypeAlias = dict[str, Any]
"""Dataset-wide aggregates.

Represents aggregated statistics or summary information across the entire dataset. This is the
output type of aggregators structured as a dictionary where the keys represent different aggregate
metrics or summary types, and the values can be of any type, depending on the nature of the
aggregate.
"""

TraceIndexList: TypeAlias = list[int]
"""A list of trace indices used to map outputs to their source samples in augmentation processes.

In data augmentation, a single input sample can generate multiple output samples. The
:class:`TraceIndexList` tracks the origin of each output sample by maintaining a list of indices.
Each index in this list corresponds to the position of the input sample in the original batch that
was used to generate the output sample.

For example, if `trace_index[i] = j`, it indicates that the `i`-th output sample was derived from
the  `j`-th input sample in the original batch.

Usage Context:
    - When a batch of input samples undergoes augmentation, this list provides a direct mapping
      from each output sample back to its corresponding input sample.
    - This type is commonly returned alongside the augmented batch, enabling users to track which
      input sample produced which output sample.
"""

NodeId: TypeAlias = str
"""Node ID type in the data flow graph.

Represents the identifier for a node within a data flow graph. This is typically a string that
uniquely identifies a node, allowing for the tracking and referencing of nodes within the graph
structure.
"""

PartitionId: TypeAlias = str
"""An identifier for a partition within the data flow graph.

The :class:`PartitionId` is a string that uniquely identifies these partitions, enabling the
tracking and management of different stages within the data flow graph.

A partition in the data flow graph represents a subgraph where each sample from the dataset is
processed or transformed independently of others. Partitions are often introduced during data
augmentation processes, where new samples are generated or existing samples are filtered out.
"""

# TODO: depricated, use hyped.core.abstract instead
DataFlowGraphAlias: TypeAlias = object
"""Data Flow Graph Alias type.

Alias for :class:`hyped.core.graph.DataFlowGraph`.

Represents an alias or reference to the entire data flow graph. This can be used to refer to the
graph in contexts where the actual structure of the graph is abstracted away. The type is generic
(object) to accommodate various possible representations of a data flow graph.
"""

# TODO: depricated, use hyped.core.features.reference.FeatureKey
FeatureKeyAlias: TypeAlias = tuple[str | int | slice, ...]
"""A key identifying specific features within a data flow graph node's outputs.

Alias for :class:`hyped.core.features.feature_key.FeatureKey`.

Represents a tuple that serves as a key for accessing specific features of a node's output within
the data flow graph.
"""

# TODO: depricated, use hyped.core.features.reference.Reference instead
Pointer: TypeAlias = tuple[NodeId, FeatureKeyAlias, DataFlowGraphAlias]
"""Pointer pointing to output features of a node in the data flow graph.

Represents a tuple used to point to specific output features of a node within a data flow graph.
The tuple consists of:
- :class:`NodeId`: The identifier of the node.
- :class:`FeatureKeyAlias`: The key identifying the specific feature within the node's outputs.
- :class:`DataFlowGraphAlias`: The data flow graph in which the node resides.

This type alias is typically used in scenarios where specific outputs from a graph's node need to
be tracked or referenced.
"""

DatasetType: TypeAlias = Dataset | DatasetDict | IterableDataset | IterableDatasetDict
"""Dataset Type Alias.

Alias for the different dataset types supported, including:

- :class:`Dataset`: A single dataset containing features and samples.
- :class:`DatasetDict`: A dictionary-like structure containing multiple datasets, often split into
  training, validation, and test sets.
- :class:`IterableDataset`: A dataset that is lazily loaded, allowing for streaming data processing.
- :class:`IterableDatasetDict`: A dictionary-like structure containing multiple iterable datasets.

This type alias is used to represent any of the aforementioned dataset types when processing or
consuming datasets in various contexts.
"""

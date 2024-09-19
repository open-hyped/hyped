"""CAS Dataset Generator.

This module defines a dataset generator for processing CAS (Common Analysis Structure) files using 
:code:`datasets` from the Hugging Face library. It builds on the :class:`CasParser` to extract and
convert CAS annotations into a structured dataset. The module is configured using the
:class:`CasDatasetConfig` and can be used by calling :func:`datasets.load_dataset` with appropriate
keyword arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Iterator

import datasets

from hyped import DataFlow
from hyped.common.typing import Sample
from hyped.ops import collect
from hyped.processors import CasParser, FileLoader


@dataclass
class CasDatasetConfig(datasets.BuilderConfig):
    """Cas Dataset Configuration.

    The attributes of the configuration are typically set by providing
    them as keyword arguments to the :func:`datasets.load_dataset` function.
    """

    typesystem: str = None
    """Path to the CAS typesystem XML file.
    
    Defaults to the dkpro-cassis core typesystem (see :code:`cassis.load_dkpro_core_typesystem`)
    """

    types: None | list[str] = None
    """List of annotation types to parse from the CAS.
    
    By default, loads all types present in the typesystem.
    """


class CasDatasetBuilder(datasets.GeneratorBasedBuilder):
    """A dataset builder that processes CAS files to create a structured dataset.

    The :class:`CasDataset` extracts CAS annotations and represents them in a structured
    format,  leveraging the :class:`CasParser` to load and parse CAS data. This builder is
    typically used in combination with the :func:`datasets.load_dataset` function.

    The dataset structure is built based on CAS annotations, where the parser converts
    them into a dictionary of lists. Each list represents a specific annotation attribute,
    and attributes from the same annotation are aligned by their index.

    Example usage:

    .. code-block:: python

        datasets.load_dataset('hyped.io.datasets.cas', **kwargs)

    """

    BUILDER_CONFIG_CLASS = CasDatasetConfig

    def _build_flow(self) -> DataFlow:
        """Build a data flow pipeline for loading and parsing CAS files.

        This method defines the data processing steps, which include:
        1. Loading file paths using the :class:`FileLoader`.
        2. Parsing the CAS content using :class:`CasParser` to extract annotations.
        3. Collecting and transforming the parsed data into the desired format.

        Returns:
            DataFlow: The constructed data flow pipeline for processing CAS files.
        """

        features = datasets.Features({"file_path": datasets.Value("string")})
        flow = DataFlow(features)

        # create cas parser
        parser = CasParser(typesystem=self.config.typesystem, types=self.config.types)

        # load file from file path and parse content
        payload = FileLoader().call(file_path=flow.src_features.file_path).content
        parsed_obj = parser.call(payload=payload).obj

        # collect all outputs
        output = {key: parsed_obj[key] for key in parsed_obj.feature_.keys()}
        output["file_path"] = flow.src_features.file_path

        # build flow
        flow, _ = flow.build(collect=collect(output))

        return flow

    @property
    def features(self) -> datasets.Features:
        """Define the features of the dataset, representing the CAS annotations.

        The features are determined by the output of the data flow pipeline, which
        includes various CAS attributes (both primitive and non-primitive).

        Returns:
            datasets.Features: The features of the dataset.
        """
        return self._build_flow().out_features.feature_

    def _info(self):
        """
        Provides metadata about the dataset, such as its description and features.

        Returns:
            datasets.DatasetInfo: Metadata about the dataset, including features and
            description.
        """
        return datasets.DatasetInfo(
            description="Cas Dataset", features=self.features, supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Define the dataset splits based on the provided data files.

        Args:
            dl_manager (datasets.DownloadManager): Manager to handle downloading and extraction
                of files.

        Returns:
            list[datasets.SplitGenerator]: List of dataset splits to be generated.

        Raises:
            ValueError: If no data files are provided or if the data files are not in the expected
                format.
        """
        # check data files argument
        if self.config.data_files is None:
            raise ValueError(
                "No data files specified. Please specify `data_files` in "
                "call to `datasets.load_dataset`."
            )
        if not isinstance(self.config.data_files, dict):
            raise ValueError(
                "Expected `data_files` to be a dictionary mapping splits "
                "to files, got %s" % type(self.config.data_files).__name__
            )

        # prepare data files
        data_files = dl_manager.download_and_extract(self.config.data_files)
        assert isinstance(self.config.data_files, dict), (
            "Expected dict but got %s" % type(data_files).__name__
        )

        splits = []
        # generate data split generators
        for split_name, files in self.config.data_files.items():
            # generate split generator
            files = [dl_manager.iter_files(file) for file in files]
            split = datasets.SplitGenerator(
                name=split_name,
                gen_kwargs=dict(files=files),
            )
            split.split_info.num_examples = len(files)
            # add to splits
            splits.append(split)

        return splits

    def _generate_examples(self, files: list[Iterator[str]]) -> Iterator[Sample]:
        """Generate examples by applying the data flow to the dataset of file paths.

        Args:
            files (list[Iterator[str]]): List of iterators over file paths.

        Yields:
            Iterator[Sample]: Samples containing CAS annotations processed from the files.
        """

        flow = self._build_flow()
        # create dataset of file paths
        ds = datasets.Dataset.from_dict(
            {"file_path": chain.from_iterable(files)}, features=flow.src_features.feature_
        )
        # apply the flow to the dataset
        ds = ds.to_iterable_dataset()
        ds, _ = flow.apply(ds)
        # yield examples from dataset
        yield from enumerate(ds)

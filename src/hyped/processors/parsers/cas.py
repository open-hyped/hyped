"""
Module for parsing CAS (Common Analysis Structure).

The module uses the :code:`dkpro-cassis` library to handle UIMA CAS data and builds a feature set
for annotations in the CAS.
"""

from itertools import chain
from typing import Annotated

import cassis
from datasets import Features, Sequence, Value
from typing_extensions import Unpack

from hyped.common.typing import Index, Rank, Sample
from hyped.core.refs.inputs import CheckFeatureEquals
from hyped.core.refs.outputs import LambdaOutputFeature
from hyped.core.refs.ref import FeatureRef

from .base import (
    BaseParser,
    BaseParserConfig,
    BaseParserInputRefs,
    BaseParserOutputRefs,
    IOContext,
    ParserException,
)

_PRIMITIVE_TYPE_MAP = {
    "uima.cas.Boolean": Value("bool"),
    "uima.cas.Byte": Value("binary"),
    "uima.cas.Short": Value("int16"),
    "uima.cas.Integer": Value("int32"),
    "uima.cas.Long": Value("int64"),
    "uima.cas.Float": Value("float32"),
    "uima.cas.Double": Value("float64"),
    "uima.cas.String": Value("string"),
}


def _load_typesystem(typesystem: None | str) -> cassis.TypeSystem:
    """Loads the cas typesystem

    Either loads the typesystem from a provided path or load the core typesystem in case no path
    was provided.

    Args:
        typesystem (None | str): Path to the typesystem XML file, or None for an empty typesystem.

    Returns:
        cassis.TypeSystem: The loaded typesystem object.
    """
    if typesystem is None:
        return cassis.load_dkpro_core_typesystem()

    with open(typesystem, "rb") as f:
        return cassis.load_typesystem(f)


def _get_types_from_typesystem(
    typesystem: cassis.TypeSystem, type_names: None | list[str] = None
) -> list[cassis.typesystem.Type]:
    """Retrieves types from a typesystem, based on specified type names or all available types.

    Args:
        typesystem (cassis.TypeSystem): The typesystem containing the types.
        type_names (None|list[str]): List of types to retrieve. If None, retrieves all types.

    Returns:
        list[cassis.typesystem.Type]: List of requested type objects.
    """
    # fallback to all types in typesystem
    if type_names is None:
        return typesystem.get_types()

    # ensure that all requested types are present in the typesystem
    for type_name in type_names:
        if not typesystem.contains_type(type_name):
            raise TypeError("Annotation Type `%s` not found in typesystem" % type_name)

    # get requested types from typesystem
    return list(map(typesystem.get_type, type_names))


class CasParserConfig(BaseParserConfig):
    """Configuration for the :class:`CasParser`."""

    typesystem: None | str = None
    """Path to the CAS typesystem XML file.
    
    Defaults to the dkpro-cassis core typesystem (see :code:`cassis.load_dkpro_core_typesystem`)
    """

    types: None | list[str] = None
    """List of annotation types to parse from the CAS.
    
    By default, loads all types present in the typesystem.
    """


class CasParserInputRefs(BaseParserInputRefs):
    """Input references for the :class:`CasParser`."""

    payload: Annotated[FeatureRef, CheckFeatureEquals(Value("string"))]
    """Reference to the payload feature containing the CAS data."""


def _build_cas_features(config: CasParserConfig, inputs: CasParserInputRefs) -> Features:
    """Output features generator function.

    Builds a :class:`Features` object for CAS annotations based on the provided typesystem and
    configuration.

    Args:
        config (CasParserConfig): Configuration object containing the typesystem and types to
            extract.
        inputs (CasParserInputRefs): Input references for the CAS parser.

    Returns:
        Features: A dictionary-like object mapping feature names to data types.
    """

    # load the typesystem
    typesystem = _load_typesystem(config.typesystem)

    # get all requested types
    all_types = typesystem.get_types()
    req_types = list(_get_types_from_typesystem(typesystem, config.types))

    all_type_names = {_type.name for _type in all_types}
    req_type_names = {_type.name for _type in req_types}

    # check if all required types are present
    for t in req_types:
        for f in t.all_features:
            # check if the feature refers to other annotations
            if (f.rangeType.name in all_type_names) and (f.rangeType.name not in req_type_names):
                raise RuntimeError(
                    "Annotation type `%s` requires type `%s`" % (t.name, f.rangeType.name)
                )

    # all primitive features of all requested types
    primitive_features = {
        "%s:%s" % (t.name, f.name): Sequence(_PRIMITIVE_TYPE_MAP[f.rangeType.name])
        for t in _get_types_from_typesystem(typesystem, config.types)
        for f in t.all_features
        if typesystem.is_primitive(f.rangeType)
    }

    # all nested features that point to other annotations
    nested_features = {
        "%s:%s" % (t.name, f.name): Sequence(Value("int32"))
        for t in req_types
        for f in t.all_features
        if f.rangeType.name in req_type_names
    }

    # extract features from typesystem
    return Features({"sofa": Value("string")} | primitive_features | nested_features)


class CasParserOutputRefs(BaseParserOutputRefs):
    """Output references for the CasParser."""

    obj: Annotated[FeatureRef, LambdaOutputFeature(_build_cas_features)]
    """Reference to the parsed CAS object, including extracted features."""


class CasParser(BaseParser[CasParserConfig, CasParserInputRefs, CasParserOutputRefs]):
    """A parser for UIMA CAS data in JSON or XMI format.

    A parser that processes CAS (Common Analysis Structure) annotations and converts them into a
    structured dictionary of lists. Each key in the dictionary represents a combination of the
    annotation type and attribute in the format :code:`"<TYPE>:<ATTR>"`.

    The parser supports both primitive types (e.g., strings, integers) and non-primitive types
    (e.g., references to other annotations). For primitive attributes, values are directly added
    to the lists. For non-primitives, where an annotation references another annotation, the value
    in the list is the index of the referenced annotation.

    The lists are ordered so that different attributes of the same annotation are at the same index
    across all lists.

    Assume we have two types of annotations in the CAS:

    1. **Person** with attributes:
        - :code:`name` (primitive, string)
        - :code:`age` (primitive, integer)

    2. **Address** with attributes:
        - :code:`city` (primitive, string)
        - :code:`resident` (non-primitive, reference to `Person`)

    Example data:

    - `Person` annotations:
        - :code:`name="Alice", age=30`
        - :code:`name="Bob", age=25`

    - `Address` annotations:
        - :code:`city="New York", resident="Alice"`
        - :code:`city="Los Angeles", resident="Bob"`

    The resulting dictionary will look like this:

    .. code-block:: python

        {
            "Person:name": ["Alice", "Bob"],         # List of names (primitives)
            "Person:age": [30, 25],                  # List of ages (primitives)

            "Address:city": ["New York", "Los Angeles"],  # List of cities (primitives)
            "Address:resident": [0, 1],                   # Index of referenced person annotations
        }

    - For :code:`Person:name` and :code:`Person:age`, the values are stored directly in lists.
    - For :code:`Address:resident`, the value refers to the index of the linked :code:`Person`
      annotation. "Alice" (index 0) is the resident of "New York," and "Bob" (index 1) is the
      resident of "Los Angeles."

    This structure ensures that attributes of the same annotation are aligned by index across lists.

    Raises:
        ParserException: If the CAS payload is invalid or if the required types are missing from the
            typesystem.
    """

    def __init__(self, config: None | CasParserConfig = None, **kwargs) -> None:
        """
        Initializes the :class:`CasParser` with the provided configuration.

        Args:
            config (None|CasParserConfig): The parser configuration.
            kwargs: Additional arguments for the parser.
        """
        super().__init__(config, **kwargs)

        self._typesystem = _load_typesystem(self.config.typesystem)

    async def parse(self, inputs: Sample, index: Index, rank: Rank, io: IOContext) -> Sample:
        """Parses the CAS content and extracts annotations based on the configured typesystem.

        Args:
            inputs (Sample): The sample containing the CAS payload.
            index (Index): The index of the sample in the batch.
            rank (Rank): The rank of the sample in the batch.
            io (IOContext): Context for input/output operations.

        Returns:
            Sample: A sample containing the extracted annotations and features.
        """

        # get content
        payload = inputs["payload"]

        try:
            # try to parse content to cas object
            if payload.startswith("{"):
                cas = cassis.load_cas_from_json(payload, typesystem=self._typesystem)
            elif payload.startswith("<?xml"):
                cas = cassis.load_cas_from_xmi(payload, typesystem=self._typesystem)
            else:
                # invalid format
                raise ParserException("INVALID")  # TODO: write error message

        except Exception as e:
            raise ParserException(str(e)) from e

        types = list(_get_types_from_typesystem(self._typesystem, self.config.types))
        # collect all annotations and create a fixed ordering over
        # the annotations of each type
        annotations = {t.name: [a.xmiID for a in cas.select(t)] for t in types}

        # check ids
        assert all(xmi_id is not None for xmi_id in chain.from_iterable(annotations.values()))

        sample = {key: [] for key in io.outputs["obj"].keys() if key != "sofa"}
        sample["sofa"] = cas.sofa_string

        for t in types:
            # get all features of interest for the annotation type
            primitive_feature_types = [
                f for f in t.all_features if self._typesystem.is_primitive(f.rangeType)
            ]
            nested_feature_types = [
                f for f in t.all_features if f.rangeType.name in annotations.keys()
            ]

            # iterate over all annotations of the current type
            for annotation in cas.select(t):
                # add primitive features to dict
                for feature_type in primitive_feature_types:
                    key = "%s:%s" % (t.name, feature_type.name)
                    sample[key].append(annotation.get(feature_type.name))
                # add nested features to dict
                for feature_type in nested_feature_types:
                    key = "%s:%s" % (t.name, feature_type.name)
                    sample[key].append(
                        annotations[feature_type.rangeType.name].index(
                            annotation.get(feature_type.name).xmiID
                        )
                    )

        return sample

    def call(self, **kwargs: Unpack[CasParserInputRefs]) -> CasParserOutputRefs:
        """Add the :class:`CasParser` node to the data flow.

        This method processes the input references for the CAS parsing operation, adds
        the corresponding node to the data flow, and returns the references to the
        output features generated by the processor.

        Args:
            payload (FeatureRef): The reference to the CAS payload to parse.
            **kwargs (FeatureRef): Additional keyword arguments passed to the call method.

        Returns:
            CasParserOutputRefs: The output references produced by the :class:`CasParser`
            processor.
        """
        return super(CasParser, self).call(**kwargs)

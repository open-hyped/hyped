"""Jinja2 Template Data Processor."""
from functools import partial
from typing import Any, Callable

from datasets import Features, Value
from jinja2 import Environment

from hyped.common.feature_key import FeatureKey
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


def _map_to_none(*args, **kwargs):
    """Helper function mapping always to none."""
    return None


class _set_filters(object):
    """Context manager setting and resetting environment filters."""

    def __init__(
        self, env: Environment, filters: dict[str, Callable[[Any], Any]]
    ) -> None:
        """Initializer.

        Arguments:
            env (jinja2.Environment):
                Jinja2 environment to set filters for
            filters (dict[str, Callable[[Any], Any]):
                collection of filters to set when entering the context manager
        """
        self.env = env
        self.filters = filters
        self.cached_filters = None

    def __enter__(self) -> None:
        """Set filters."""
        self.cached_filters = self.env.filters.copy()
        self.env.filters.update(self.filters)

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        """Reset filters."""
        self.env.filters = self.cached_filters


class Jinja2Config(BaseDataProcessorConfig):
    """Jinja2 Template Data Processor Config.

    Creates a new feature by applying a jinja2 template
    to the given datapoints.

    Attributes:
        template (str):
            string template to apply
        output (str):
            output column to store the rendered template at
    """

    template: str
    output: str


class Jinja2(BaseDataProcessor[Jinja2Config]):
    """Jinja2 Template Data Processor Config.

    Creates a new feature by applying a jinja2 template
    to the given datapoints.
    """

    def __init__(self, config: Jinja2Config) -> None:
        """Instantiate a new Jinja2 Template Data Processor.

        Arguments:
            config (Jinja2Config):
                config of the data processor
        """
        super(Jinja2, self).__init__(config)
        # set up the jinja environment
        self.env = Environment()
        self.env.filters = {
            "index_example": _map_to_none,
            "index_features": _map_to_none,
        }
        # create template
        self.template = self.env.from_string(self.config.template)
        # collect feature keys mentioned in template
        self.feature_keys = self._collect_required_feature_keys()

    def _collect_required_feature_keys(self) -> set[FeatureKey]:
        """Collect all feature keys referenced in the template.

        Returns:
            feature_keys (set[FeatureKey]):
                a set of all feature keys referenced in the template
        """
        feature_keys = set()

        with _set_filters(
            self.env,
            {
                "index_example": lambda key: feature_keys.add(key),
                "index_features": lambda key: feature_keys.add(key),
            },
        ):
            try:
                # TODO: aborts processing of the template at first
                #       exception and doesn't capture feature keys
                #       mentioned after that, the exception itself
                #       might be raised because the filters return
                #       None while the template further processes
                #       the filter output
                self.template.render(FeatureKey=FeatureKey)
            except:
                pass

        return feature_keys

    def map_features(self, features: Features) -> Features:
        """Map features.

        Checks whether the feature keys referenced in the template are valid
        and returns the output feature.

        Arguments:
            features (datasets.Features):
                input dataset features

        Returns:
            out_features: (datasets.Features):
                output dataset features
        """
        # check if all features exist
        for key in self.feature_keys:
            key.index_features(features)
        # create output feature
        return {self.config.output: Value("string")}

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Process example.

        Renders the template based on the given example and it's
        features.

        Arguments:
            example (dict[str, Any]):
                example to process
            index (int):
                dataset index of the example
            rank (int):
                execution process rank

        Returns:
            out (dict[str, Any]):
                output exmaple containing the rendered template
        """
        # set filters to use actual values
        with _set_filters(
            self.env,
            {
                "index_example": lambda key: key.index_example(example),
                "index_features": lambda key: key.index_features(
                    self.in_features
                ),
            },
        ):
            return {
                self.config.output: self.template.render(FeatureKey=FeatureKey)
            }

"""Hyped."""
import importlib.metadata

try:
    # get version from package metadata
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # package not installed
    __version__ = None

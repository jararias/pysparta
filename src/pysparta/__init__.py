
import importlib.metadata

from .model import SPARTA  # noqa: F401

try:
    __version__ = importlib.metadata.version("pysparta")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["SPARTA", "__version__"]

"""
nicci: A collection of functions for the analysis of astronomical data cubes and images
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nicci")
except PackageNotFoundError:
    # package not installed (e.g. running directly from source)
    __version__ = "0.0.0+unknown"

# Import order matters: utils/io have no internal nicci dependencies;
# finder depends on utils; plotter depends on finder/io/utils;
# dynamics depends on utils/plotter; stacking depends on finder/io/plotter/utils.
from . import cosmoscanvas
from . import utils
from . import io
from . import finder
from . import plotter
from . import dynamics
from . import stacking
from . import photometry

#Explicit public namespace
__all__ = [
    "cosmoscanvas",
    "utils",
    "io",
    "finder",
    "plotter",
    "dynamics",
    "stacking",
    "photometry",
]
"""
nicci: A collection of functions for the analysis of astronomical data cubes and images
"""
__version__ = "1.0"

#Public API
from .main import (
    assign_to_cells,
    assign_to_rings,
    chanmap,
    convert_to_HI,
    convert_from_HI,
    fits_cutout,
    hi_mass,
    lines_finder,
    load_data,
    make_figure,
    plot_mom0,
    plot_mom1,
    plot_mom2,
    plot_noise,
    plot_pvslice,
    pvslice,
    RMS,
    shuffle,
    stack,
    velfi,
)

#Explicit public namespace
__all__ = [
    "assign_to_cells",
    "assign_to_rings",
    "chanmap",
    "convert_to_HI",
    "convert_from_HI",
    "fits_cutout",
    "hi_mass",
    "lines_finder",
    "load_data",
    "make_figure",
    "pvslice",
    "plot_mom0",
    "plot_mom1",
    "plot_mom2",
    "plot_noise",
    "plot_pvslice",
    "RMS",
    "shuffle",
    "stack",
    "velfi",
]
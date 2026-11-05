from . import config      # applies matplotlib rcParams on import
from . import colormaps   # registers custom colormaps on import

from .plotter import (
    chanmap,
    make_figure,
    plot_mom0,
    plot_mom1,
    plot_mom2,
    plot_noise,
    plot_pvslice,
)

__all__ = [
    "chanmap",
    "make_figure",
    "plot_mom0",
    "plot_mom1",
    "plot_mom2",
    "plot_noise",
    "plot_pvslice",
]
# Naive functIons Collection for Cubes and Images
A collection of functions for the analysis of astronomical data cubes and images
* Tested on Windows 10 Enterprise (build 19045.2251) with 16 GB of RAM in the following environments:
  - Jupyter notebook 6.5.7
  - Python 3.13.9
  
## Authors
- Simone Veronese (veronese.astro@gmail.com)

## Installation
Install nicci with pip
```bash
  pip install git+https://github.com/SJVeronese/nicci-package.git
```

## Upgrade
Upgrade nicci with pip
```bash
  pip install --upgrade git+https://github.com/SJVeronese/nicci-package.git
```

## Requirements
* astropy 7.1.0
* matplotlib 3.10.6
* numpy 2.3.3
* pandas 2.2.2
* pvextractor 0.4
* scipy 1.16.2
* tqdm 4.67.1

## List of functions
* assign_to_cells: Partition a 2D array into a specified number of  approximately equal, rectangular regions and return the pixel coordinates belonging to each region.
* assign_to_rings: Assign pixels of a 2D array to a set of concentric (possibly elliptical) rings.
* chanmap: Create a channel map figure from a spectral-line data cube.
* convert_to_HI: Convert input data to HI column density units.
* convert_from_HI: Convert HI data values using a beam area and velocity integration.
* fits_cutout: Create a spatial (and optionally spectral) cutout from a FITS file.
* hi_mass: Compute the total HI mass from an integrated HI intensity map.
* lines_finder: Detects and characterizes spectral line features in one or more 1D spectra.
* load_data: Load astronomical data from a FITS file, a NumPy array, or an (array, header) tuple and return a standardized dictionary with data, metadata, WCS information, and derived quantities.
* make_figure: Create a Matplotlib figure and axes with flexible layout options, including support for WCS projections, multiple panels, shared formatting, and aliases for common parameters.
* pvslice: Extract a position–velocity (PV) slice from a spectral data cube and optionally write the result to a FITS file or plot it.
* plot_mom0: Plot a moment-0 (integrated intensity) map.
* plot_mom1: Plot a moment-1 (intensity-weighted velocity) map.
* plot_mom2: Plot a moment-2 (velocity dispersion) map.
* plot_noise: Compute and visualize spatial and spectral noise properties of a 3D datacube.
* plot_pvslice: Plot a position–velocity (PV) slice with optional contours, WCS axes, physical scale annotations, and colorbar.
* RMS: Estimate the root-mean-square (RMS) noise of an array.
* shuffle: Spectrally align a datacube using a velocity field by circularly shifting each spectrum.
* stack: Stack a set of 1D spectra using optional noise-based weighting.
* velfi: Extend or model a 2D velocity field using a tilted-ring formalism.





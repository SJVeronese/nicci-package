# Pre-alpha/Early access

# Naive functIons Collection for Cubes and Images
A collection of functions for the analysis of astronomical data cubes and images
* Tested on Windows 10 Enterprise (build 19045.2251) with 16 GB of RAM in the following environments:
  - Jupyter notebook 6.5.4
  - Python 3.10.8
  
## Authors
- Simone Veronese

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
* astropy 5.3
* configparser 5.3.0
* matplotlib 3.7.1
* numpy 1.24.3
* pandas 2.0.2
* pvextractor 0.3
* scipy 1.10.1
* tqdm 4.65.0

## Jupyter Notebook 
* To show progress bars in Jupyter Notebbok:
    Install ipywidgets
    ```bash
    pip install ipywidgets
    ```
    Run in your console
    ```bash
    jupyter nbextension enable --py widgetsnbextension
    ```

## List of functions
* import_parameters: Readsan INI-structured text file and store each row in a dictionary entry that can be passed to other functions. This is the main function used to control the arguments of other functions using a parameter file.
* chanmap: Plot the channel maps of a data cube in a defined channel range every given number of channels.
* cubedo: Perform various operations on a FITS data cube, such as blanking, clipping, cropping, cutting, extending, mirroring, calculating moment 0 map, shuffling, and converting to an integer cube.
  - 'clip': Blanks all the data below the cliplevel value.
  - 'crop': Crops the data cube to remove the blanked edges.
  - 'cut': Extracts the subcube from chanmin to chanmax.
  - 'extend': Adds channels to the spectral axis.
  - 'mirror': Mirrors the data around a rotation point (x, y, z).
  - 'mom0': Computes the moment 0 map from chanmin to chanmax.
  - 'shuffle': Aligns the spectral profile with the galaxy rotation.
  - 'toint': Converts the data cube into an integer cube.
* cubestat: Calculate the detection limit of a data cube and optionally its root mean square (rms), spectral resolution, beam major axis, beam minor axis, beam position angle, and beam area. It also computes the errors on rms and sensitivity
* fitsarith: Perform arithmetic operations (sum, subtraction, multiplication, division) between two FITS files.
* fixmask: Fix a 3D detection mask by setting the voxels corresponding to negative detections to 0, based on a reference data cube. It is assumed that a value > 0 in the mask indicates a detection.
* gaussfit: Perform a Gaussian fit on a spectral cube. This function fits a single Gaussian profile to each spaxel (spatial pixel) in the spectral cube.
* getpv: Extracts a position-velocity slice from a given data cube along a path defined by specified points, angle, and width. Optionally, it can plot the slice and save it to a FITS file.
* lines_finder: One-dimensional source finding algorithm for spectral lines detection. It resembles the The Source Finding Application, SoFiA (Serra et al. 2015, Westmeier et al. 2021).
* noise_variations: Determine the noise variations of a data cube along the spatial and spectral axes.
* plotmom: Plot the given moment maps. All (moment 0, moment 1, and moment 2) can be plotted simultaneously or only one.
* removemod: Remove a model from a data cube using one of the five available methods: data-model (subtraction), data blanking (blanking), data-model after data blanking (b+s), data-model and negative residual blanking (negblank), and a combination of blanking and negblank (all).
* rotcurve: Compute the rotation curve of a galaxy.
* stacking: Stack the spectra extracted from a given number of regions around a center starting from a minimum radius up to a maximum radius. Afterwards, it runs a source-finding algorithm on each stacked spectrum to check for detected lines. It optionally stores diagnostic plots of the stacking and the source-finding routines and also optionally stores relevant diagnostic fits file of the source-finding routine..
* velfi: Compute a synthetic velocity field or extend a given velocity field.
* converttoHI: Convert an array of flux values into HI column dens
* cosmo: Given a cosmological model and the redshift of an object, this function calculates various cosmological quantities such as distances, age of the Universe, and light travel time for that object.
* getHImass: Convert an nD-array of flux densities into HI mass.
* flux: Calculate the flux over a given region.
* create_config: create a default parameter file with the given name.

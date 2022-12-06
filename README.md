# Pre-alpha/Early access

# New analysIs Code for Cubes and Images
A collection of functions for the analysis of astronomical data cubes and images

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
* astropy 4.3.1
* configparser 5.2.0
* matplotlib 3.4.3
* numpy 1.21.2
* pandas 1.3.3
* pvextractor 0.2
* scipy 1.7.1

## List of functions
* importparameters: main function if you want to use the parameter file to control the arguments of the functions. It reads an ini-structured text file and stores each row in a dictionary entry that can be pass to the functions.
* chanmap: plot the channel maps of a data cube in a defined channel range every given number of channels.
* cubedo: perform an operation on the given data cube between:
  - blank, blank all the data in a given channels range
  - clip, blank all the data below a given value
  - crop, crop the data cube to remove the blanked edges
  - cut, extract the subcube from a given channel range
  - extend, add channels to the spectral axis
  - mirror, mirror the data around a rotation point (x,y,z)
  - mom0, compute the moment 0 map in a given channel range
  - shuffle, align the spectral profile with the galaxy rotation
  - toint, convert the data cube into an integer cube 
* cubestat: calculate the detection limit of a data cube and (optional) its: rms, spectral resolution, beam major axis, beam minor axis, beam position angle and beam area. It also computes the errors on rms and detection limit.
* fitsarith: perform an arithmetical operation between two fits file. Available operations:
  - sum, sum the two fits file
  - sub, subtract the two fits file
  - mul, multiply the two fits file
  - div, divide the two fits file (the zeroes in the second fits will be blanked)
* fixmask: fix a 3D detection mask by setting to 0 the voxel corresponding to negative detections (using a reference data cube). It is assumed that a value > 0 in the mask is marking a detection.
* gaussfit: perform a gaussian fit on a spectral cube. Each spaxel will be fitted with a single gaussian.
* getpv: extract the position-velocity slice of a given data cube along a path defined by the given points, angle and width. It also (optionally) plots the slice and (optionally) save it to a fits file.
* plotmom: plot the given moment maps. All (moment 0, moment 1 and moment 2) can be plotted simultaneously or only one.
* removemod: remove a given 3D model cube from a given data cube with a method between:
  - all, apply blanking and negblank methods (see below)
  - blanking, blank the voxel in the data cube whose value in the model cube is > than a given threshold
  - b+s, same has above but it subtracts the model from the data AFTER the blanking
  - negblank, subtract the model from the data and blank the negative residuals
  - subtraction, subtract the model from the data
* rotcurve: compute the rotation curve of a galaxy and plot it and (optionally) save it as a csv file.
* stacking: stack the spectra extracted from a given number of conic regions around a center starting from a minimum radius up to a maximum radius. Afterwards, it runs a source-finding algorithm on each stacked spectrum to check for detected lines. It optionally store diagnostic plots of the stacking and the source-finding routines and also optionally stores relevant diagnostic fits file of the source-finding routine.
* velfi: compute a syntethic velocity field or (optional) extend a given velocity field.
* converttoHI: convert an nD-array of flux densities values into HI column density values.
* cosmo: given a cosmological model and the redshift of an object, it calculates the relevant cosmological  quantitities (distances, age of Unvierse, light travel time, ...) for that object.
* getHImass: convert an nD-array of flux densities into HI mass.
* flux: same as converttoHI but returns a flux.
* create_config: create a default parameter file with the given name.

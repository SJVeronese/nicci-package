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
* importparameters: import the parameter file and create the python dictionary with all the variables needed for the package
* chanmap: plot the channel map of a given datacube in a given range of channels with a given channel separation (from chanmin to chanmax each chansep channels) 
* cubedo: perform an operation on the given datacube between:
  - blank, blank all the data in a given channels range
  - clip, blank all the data below a given value
  - crop, crop the data cube to remove the blanked edges
  - cut, extract the subcube from a given channel range
  - extend, add channels to the spectral axis
  - mirror, mirror the data around a rotation point (x,y,z)
  - mom0, compute the moment 0 map in a given channel range
  - shuffle, align the spectral profile with the galaxy rotation
  - toint, convert the data cube into an integer cube 
* cubestat: get the relevant information from a given datacube: rms, detection limit, beam size, beam area, beam position angle, spectral resolution
* fitsarith: perform an arithmetical operation between two fits file. Available operations:
  - sum, sum the two fits file
  - sub, subtract the two fits file
  - mul, multiply the two fits file
  - div, divide the two fits file (the zeroes in the second fits will be blanked)
* fixmask: fix a detection mask by setting to 0 the pixel corresponding to negative detections (using a reference datacube). It is assumed that a value >0 in the mask is a detection
* gaussfit: perform a gaussian fit on a spectral cube. Each spaxel will be fitted with a single gaussian
* getpv: extract the position-velocity slice of a given datacube along a path defined by the given points, angle and width. It also (optionally) plots the slice and (optionally) save it to a fits file
* plotmom: plot the given moment maps. All (moment 0, moment 1 and moment 2) can be plotted simultaneously or only one
* removemod: remove a given model cube from a given datacube with a method between:
  - all, apply blanking and negblank methods (see below)
  - blanking, blank the pixel in the data cube whose value in the model cube is > than a given threshold
  - b+s, same has above but it subtracts the model from the data AFTER the blanking
  - negblank, subtract the model from the data and blank the negative residuals
  - subtraction, subtract the model from the data
* rotcurve: extract the rotation curve from a given moment 1 map and (optionally) plot it and (optionally) save it into a csv file
* converttoHI: convert an nD-array of flux densities values into HI column density values
* getHImass: convert an nD-array of flux densities into HI mass
* flux: same as converttoHI but return a flux
* create_config: create a default parameter file with the given name

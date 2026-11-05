# NICCI — Naive functIons Collection for Cubes and Images

A collection of Python tools for the analysis and visualization of astronomical data cubes and images, with a focus on HI/radio data: FITS I/O, spectral-line source finding, rotation-curve and vertical-structure dynamics, surface-density/brightness profile fitting, optical source cross-matching, custom colour maps, and plotting utilities.

## Authors
- Simone Veronese (veronese.astro@gmail.com)

Portions of `nicci.cosmoscanvas` (custom colour maps for velocity and spectral-index maps) are adapted from work by Gilles Ferrand, Mark Richardson, and Jayanne English.

## Installation
Install nicci with pip:
```bash
pip install git+https://github.com/SJVeronese/nicci-package.git
```

## Upgrade
```bash
pip install --upgrade git+https://github.com/SJVeronese/nicci-package.git
```

## Requirements
- Python >= 3.9
- numpy
- scipy
- matplotlib
- astropy
- pandas
- tqdm
- mcfit
- pvextractor
- colourspace-explorer
- photutils
- astroquery

## Package layout
```
src/nicci/
├── cosmoscanvas/   galaxy-image plotting and custom colour maps
├── dynamics/       rotation curves, dark-matter haloes, vertical structure, profile fitting
├── finder/         spectral-line source finding, optical contaminant masking
├── io/             FITS loading, cutouts, PV slices, cube shuffling
├── photometry/      ePSF building, Gaia queries
├── plotter/        figure/axis helpers, moment-map/channel-map/PV plots, colormaps
├── stacking/       spectral stacking, tilted-ring velocity-field modelling
└── utils/          array helpers, physical constants, JSON (de)serialization, regions, unit conversions
```

## List of functions

### `nicci.cosmoscanvas`
- `galfits.plot_galaxy(fits_file, RA, DEC, ImgSize, shift, cmap, ...)` — Plot a WCS-projected cutout of a FITS image centred on given RA/DEC, with optional beam ellipse, colorbar, and rectangular trimming.
- `galfits.get_galaxy_data(fits_file, RA, DEC, ImgSize, shift, ...)` — Return the WCS and data array of an image cutout (used internally by `plot_galaxy`, exposed for custom plotting).
- `galfits.get_galaxy_range(fits_file, RA, DEC, ImgWidth, ImgHeight, shift, ...)` — Return the min/max pixel value within a circular region around given RA/DEC.
- `specindex.create_cmap_specindex(min_p, max_p, steep_p=-0.8, flat_p=-0.1, ...)` — Build/register a diverging colour map for spectral-index maps with fixed steep/flat break points.
- `specindex.create_cmap_specindex_constantL(L_0, C_0, H_start, H_dir, ...)` — Build/register a constant-luminosity/chroma spectral-index colour map.
- `specindex.create_cmap_specindex_error(...)` — Build/register a colour map for spectral-index uncertainty maps.
- `velmap.create_cmap_velocity(min_p, max_p, div=0.0, width=0.05, ...)` — Build/register a single-divergent-point colour map for velocity (moment-1) maps.
- `velmap.create_cmap_doubleVelocity(minvalue, maxvalue, div=0.0, ...)` — Build/register a double-complementary-colour variant of the velocity colour map.
- `velmap.create_cmap_chromaVelocity(min_p, max_p, div=0.0, ...)` — Fully customizable version of `create_cmap_velocity`, tuned for white backgrounds.
- `velmap.create_cmap_velocity_constantL(L_0, C_0, H_start, H_dir, ...)` — Build/register a constant-luminosity velocity colour map.

### `nicci.dynamics`
- `dynamics.circular_velocity(radii, sdens, *, sigma_r, sigma_phi, **kwargs)` — Circular velocity of a disc from its surface density, with optional asymmetric-drift correction.
- `dynamics.circular_velocity_dm(radii, halo, **params)` — Circular velocity of a dark-matter halo (`iso`, `burkert`, `nfw`, or `dc14`).
- `dynamics.cored_halo_params(M200, halo='burkert')` — Solve for the core density/radius of a cored halo matching a given M200.
- `dynamics.fit_dark_matter(radii, *, vobs, vstars, vgas, ..., halo='iso', ...)` — Least-squares fit of a dark-matter halo (and optionally the stellar M/L ratio) to an observed rotation curve.
- `dynamics.sdens_dm(radii, halo, ...)` — Projected surface-density profile of a dark-matter halo.
- `dynamics.rho_dm(radii, halo, ...)` — 3D volume-density profile of a dark-matter halo.
- `dynamics.rotation_curve(radii, sdens, *, ..., mc_iter=1, ...)` — Circular velocity of a disc, with optional Monte Carlo error propagation; wraps `circular_velocity`.
- `dynamics.vertical_structure(radii, components, ...)` — Self-consistently solve the vertical (hydrostatic) structure of a multi-component disc, optionally including a dark-matter halo and CGM confinement.
- `dynamics.virial_temperature(M200, mu=0.6)` — Virial temperature of a dark-matter halo.
- `dynamics.volume_density(z_grid, z0, sdens, z_layer)` — Midplane and 3D volume density from a surface density and scale height.
- `dynamics.volume_density_dm(radii, z, halo_model, halo_params)` — 3D volume density of a dark-matter halo on a (z, R) grid, from its rotation curve.
- `profiles.effective_radius(r, Sigma)` — Projected half-light radius of a surface density/brightness profile.
- `profiles.exponential(R, Sigma_0, R_d)` — Exponential surface density profile.
- `profiles.fit_profile(radii, sdens, *, model='sersic', ...)` — Least-squares fit of a composite exponential/Sersic/broken-exponential surface density model.
- `profiles.sersic(R, Sigma_e, R_e, n)` — Sersic surface density profile.
- `profiles.smooth_broken_exponential(R, Sigma_0, alpha, beta, R_b, gamma)` — Smooth broken-exponential surface density profile (Erwin 2008).

### `nicci.finder`
- `finder.lines_finder(spectrum, spectrum_rms=None, ...)` — Detect, link, and optionally assess the reliability/gaussianity of line-like sources in one or more spectra.
- `finder.optical_finder(data, *, sources_pos, gal_pos, ...)` — Detect sources in an optical image via segmentation and mask contaminants near a galaxy, matching against candidate source positions (e.g. from Gaia).

### `nicci.io`
- `fits.cutout(input_fits, *, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, output=None)` — Extract a spatial/spectral cutout from a FITS file or array, updating the WCS.
- `fits.load_data(input_data, *, ext=None)` — Load data/header/WCS/derived metadata (pixel scale, beam, units, reference coordinates) from a FITS file, array, or (array, header) tuple into a standard dict.
- `fits.pvslice(datacube, *, width=3, start_point=None, end_point=None, length=None, angle=None, ...)` — Extract (and optionally plot) a position-velocity slice from a cube.
- `fits.shuffle(input_fits, velocity_field, *, ...)` — Circularly shift each spectrum of a cube so a velocity field aligns to a single channel (GIPSY SHUFFLE equivalent).

### `nicci.photometry`
- `optical.download_gaia(radius=60, *, centre=None, ra=None, dec=None, output=None)` — Query the Gaia archive for sources within a cone around a target.
- `optical.epsf(data, *, threshold=10.0, fwhm=1.5, cutout_size=25.0, ...)` — Build an effective PSF from stars detected in an image via `DAOStarFinder`/`EPSFBuilder`.

### `nicci.plotter`
- `colormaps` — Import-time side effect: registers the `mom1`, `mom1_alt`, `mom1_cosmos`, `mom2`, and `mom2_ctr` colormaps with matplotlib if not already present.
- `config.config()` — Set default matplotlib rcParams (fonts, tick direction/size, mathtext) for consistent plot style; runs automatically on import.
- `plotter.chanmap(datacube, channel_step, *, ...)` — Plot a grid of channel maps with contours, beam, colorbar, and physical scale bar.
- `plotter.make_figure(**kwargs)` — Create a single- or multi-panel WCS-aware figure with consistent axis labels/ticks/limits; used internally by most plotting functions.
- `plotter.plot_mom0(data, **kwargs)` / `plot_mom1(data, **kwargs)` / `plot_mom2(data, **kwargs)` — Plot moment-0/1/2 maps with contours, beam, scale bar, and colorbar.
- `plotter.plot_noise(datacube, *, ...)` — Compute and plot the spatial and per-channel noise of a data cube.
- `plotter.plot_pvslice(pv_data, pv_header, *, ...)` — Plot a position-velocity slice with contours and annotations.

### `nicci.stacking`
- `stacking.stack(spectra, *, weighting='none', statistics='mad', ...)` — Stack spectra with optional inverse-variance weighting, tracking running rms vs. the noise-limited expectation.
- `stacking.velfi(velocity_field, *, radii, pa, inc, vrad, vsys, vrot, ...)` — Extend or model a 2D velocity field from a tilted-ring rotation model (GIPSY VELFI equivalent).

### `nicci.utils`
- `array.extend(var, target_length)` — Extend a scalar/array to a target length by repeating its last entry.
- `array.insert_key(dictionary, key, value, target_key)` — Insert a key/value pair into a dict immediately after a given key, preserving order.
- `array.normalise(arr, new_min, new_max)` — Rescale an array to a new [min, max] range.
- `array.RMS(data, *, mode='mad', fluxrange='negative', ...)` — Estimate noise via standard deviation or MAD, over a chosen flux subset.
- `constants` — Physical constants: `kB`, `G`, `H0`, `Mp`, `Msun_to_MH`, `kpc_cm`.
- `json.to_json(obj)` / `from_json(obj)` — Recursively convert NumPy arrays/scalars to/from native Python types for JSON (de)serialization.
- `json.load_json(path)` / `save_json(data, path)` — Load/save a JSON file, converting NumPy arrays automatically.
- `regions.assign_to_cells(data, *, nregions)` — Partition a 2D array into a near-square grid of rectangular regions.
- `regions.assign_to_rings(data, *, radii, pa, inc, xpos, ypos, ...)` — Assign pixels of a 2D array to a set of concentric (tilted) elliptical rings.
- `units.convert_to_HI(data, beamarea, *, spectralres=None, from_moment_map=False)` — Convert flux density to HI column density.
- `units.convert_from_HI(data, beamarea, *, spectralres=None, from_moment_map=False)` — Inverse of `convert_to_HI`.
- `units.hi_mass(data, *, beamarea=None, distance=None, pixelres=None, ...)` — Compute total HI mass from a flux map or flux value.

## License
See `LICENSE`.
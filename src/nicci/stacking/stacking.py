from astropy.io import fits
from astropy.wcs import WCS
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

from nicci.finder import lines_finder
from nicci.io.fits import load_data
from nicci.plotter.plotter import _plot_spectrum, make_figure, plot_mom1
from nicci.utils.array import extend, RMS
from nicci.utils.regions import _radius


def stack(spectra, *,
          weighting='none', statistics='mad', fluxrange='negative',
          run_finder=False, plot=False,
          show_progress_bar=True, output=None,
          **kwargs):
    """Stack a collection of spectra, with optional noise weighting.

    Spectra are co-added incrementally (optionally inverse-variance
    weighted), tracking the running rms of the partial stack and the rms
    expected from pure noise averaging (``rms_1 / sqrt(N)``).

    Parameters
    ----------
    spectra : sequence of ndarray
        Spectra to stack; all must have the same length.
    weighting : {'none', 'rms'}, default 'none'
        Weighting scheme: equal weights, or inverse-variance weights
        based on each spectrum's noise.
    statistics : {'mad', 'std'}, default 'mad'
        Statistic used to estimate the noise, passed to `RMS`.
    fluxrange : {'negative', 'positive', 'all'}, default 'negative'
        Flux range used to estimate the noise, passed to `RMS`.
    output : str, optional
        If given, a plot of the stacked spectrum is saved to this path.
    run_finder : bool, default False
        If True, run `lines_finder` on the stacked spectrum.
    plot : bool, default False
        If True, display a plot of the stacked spectrum.
    show_progress_bar : bool, default True
        Whether to display a `tqdm` progress bar.
    **kwargs
        Additional keyword arguments forwarded to `lines_finder` (if
        `run_finder` is True) and to `_plot_spectrum` (if `output` or
        `plot` is set).

    Returns
    -------
    stack : ndarray
        The stacked (weighted-average) spectrum.
    stack_rms : ndarray
        Running rms of the partial stack after each spectrum is added.
    exp : ndarray
        Expected rms from pure noise averaging after each spectrum is
        added.
    """

    # initialize the number of stacked spectra
    N = 1

    # initialize the weights
    weights = 0

    # initialize the stacked spectrum
    stack = np.zeros(spectra[0].shape[0])

    # initialize the stacked rms list
    stack_rms = []

    # initialize the expected rms list
    exp = []

    # stack the spectra
    for i in tqdm(range(len(spectra)),
                  desc='Spectra stacked', total=len(spectra),
                  disable=not show_progress_bar):
        # extract the spectrum
        spectrum = spectra[i]

        # if the spectrum to be stacked is empty
        if np.all(np.isnan(spectrum)):
            continue

        # if no weighting
        if weighting == 'none':
            # the weight is 1
            w = 1
        # if noise weighting
        elif weighting == 'rms':
            # the weight is 1/rms**2 of the input spectrum
            w = 1 / RMS(spectrum, mode=statistics, fluxrange=fluxrange) ** 2

        # stack the input spectrum
        stack = np.nansum((stack, spectrum * w), axis=0)

        # stack the weight
        weights += w

        # calculate the rms of the stacked spectrum
        stack_rms.append(RMS(stack / weights, mode=statistics, fluxrange=fluxrange))

        # calculate the expected value for the rms
        exp.append(stack_rms[0] / np.sqrt(N))

        # increase the number of stacked spectra
        N += 1

    # apply the weights
    stack /= weights

    # run the line finder
    if run_finder:
        finder_output = kwargs.get('finder_output', None)
        mask = lines_finder(stack, spectrum_rms=stack_rms[0], output=finder_output, **kwargs)[0]
    else:
        mask = None

    # save stacked spectrum
    if output:
        # store the kwargs
        dpi = kwargs.get('dpi', 300)

        # do the plot
        fig, ax = _plot_spectrum(stack, mask=mask, **kwargs)

        # save the figure
        fig.savefig(output, dpi=dpi, bbox_inches='tight')

    # plot the stacked spectrum
    if plot:
        # do the plot
        fig, ax = _plot_spectrum(stack, mask=mask, **kwargs)
        plt.show(fig)

    return stack, np.array(stack_rms), np.array(exp)


def velfi(velocity_field, *,
          radii, pa, inc, vrad, vsys, vrot,
          pixelres=None, center=None, extend_only=True,
          plot=False, cmap='mom1_cosmos',
          show_progress_bar=True, output=None,
          **kwargs):
    """Extend (or model) a velocity field using a tilted-ring rotation model.

    Interpolates the rotation-curve parameters (position angle,
    inclination, systemic/radial/rotation velocities) onto each pixel's
    elliptical radius and evaluates the corresponding line-of-sight
    velocity, following GIPSY's VELFI task.

    Parameters
    ----------
    velocity_field : str, ndarray, or tuple
        Input 2D velocity field, passed to `load_data`.
    radii : array_like
        Ring radii, in the same physical units as `pixelres` (e.g. arcsec
        if `pixelres` is in arcsec/pixel).
    pa, inc : float or array_like
        Position angle and inclination per ring, in degrees.
    vrad, vsys, vrot : float or array_like
        Radial, systemic, and rotation velocity per ring, in the same
        units as the velocity field.
    pixelres : float, optional
        Pixel resolution, used to convert `radii` to pixel units. Read
        from the cube metadata if not given.
    center : tuple of (float, float), optional
        Rotation centre, as (y, x) in pixel units. Defaults to the array
        centre.
    extend_only : bool, default True
        If True, only pixels without an existing (finite) velocity-field
        value are filled in; otherwise the full model is computed
        everywhere within the ring range.
    plot : bool, default True
        If True, plot the original and extended/modelled velocity fields
        side by side.
    cmap : str, default 'mom1_cosmos'
        Colormap used for the velocity field plots.
    output : str, optional
        If given, the result is written to this path as a FITS file.
    show_progress_bar : bool, default True
        Whether to display a `tqdm` progress bar.
    **kwargs
        Additional keyword arguments forwarded to `make_figure` if `plot`
        is True.

    Returns
    -------
    result : ndarray
        The extended/modelled velocity field, only if `output` is None.
    fig : matplotlib.figure.Figure
        The comparison figure, only if `plot` is True.
    axes : list of matplotlib.axes.Axes
        The comparison figure's axes, only if `plot` is True.

    Notes
    -----
    Depending on which of `output` and `plot` are set, the return
    signature is ``result`` (neither), ``(fig, axes)`` (`output` only, via
    the FITS write inside this call), or ``(result, fig, axes)`` (`plot`
    only, with `output` None). If both are given, the FITS file is
    written and ``(fig, axes)`` is returned.
    """

    # load the input velocity field
    data_dict = load_data(velocity_field)
    data = data_dict['data']

    # store the relevant metadata
    if pixelres is None:
        pixelres = data_dict['pixelres']
    if center is None:
        center = (data.shape[1] // 2, data.shape[0] // 2)
    if np.asarray(center).shape != (2,):
        raise ValueError('Provide the rotational center as a tuple (x,y) in pixel units')
    y0, x0 = center

    # convert the radii into array
    radii = np.asarray(radii)

    # get the number of radii
    nradii = len(radii)

    # extend the relevant quantities
    pa = extend(pa, nradii)
    inc = extend(inc, nradii)
    vrad = extend(vrad, nradii)
    vsys = extend(vsys, nradii)
    vrot = extend(vrot, nradii)

    # convert into radians
    pa = np.radians(pa)
    inc = np.radians(inc)

    # convert from data units to pixel units
    radii /= np.abs(pixelres)

    # initialize the result
    if extend_only:
        result = data.copy()
    else:
        result = data.copy() * np.nan

    # define the spatial range boundaries
    rmax = np.nanmax(radii)
    xmin = int(np.around(x0 - rmax, 0))
    xmax = int(np.around(x0 + rmax, 0))
    ymin = int(np.around(y0 - rmax, 0))
    ymax = int(np.around(y0 + rmax, 0))

    # clip the boundaries
    if xmin < 0:
        xmin = 0
    if xmax >= data.shape[1]:
        xmax = data.shape[1]
    if ymin < 0:
        ymin = 0
    if ymax >= data.shape[0]:
        ymax = data.shape[0]

    # define the spatial range
    xrange = np.arange(xmin, xmax)
    yrange = np.arange(ymin, ymax)

    # compute the velocity field extension
    with tqdm(desc='Pixel computed', total=len(xrange) * len(yrange), disable=not show_progress_bar) as pbar:
        for i in yrange:
            for j in xrange:
                # if the value in (i,j) is empty
                if np.isnan(result[i, j]):
                    # convert the pixel coordinates w.r.t. the center
                    x = j - x0
                    y = i - y0

                    # calculate the distance from the center
                    r = np.sqrt(x**2 + y**2)

                    # if the distance is less or equal the outer radius
                    if r <= radii[nradii - 1]:
                        # calculate the radius of (x,y) compared to the innermost radius
                        d1 = _radius(radii[0], x, y, pa[0], inc[0])

                        # calculate the radius of (x,y) compared to the outermost radius
                        d2 = _radius(radii[nradii - 1], x, y, pa[nradii - 1], inc[nradii - 1])

                        # If one of the two distances is negative, the point lies
                        # between the innermost and outermost radius, so the
                        # rotation velocity can be computed at that point.
                        if d1 * d2 <= 0:
                            # find the radius[i] corresponding to that point
                            for k in np.arange(1, nradii + 1, 1):
                                # keep computing the distance between (x, y) and the
                                # k-th radius
                                d2 = _radius(radii[k], x, y, pa[k], inc[k])

                                # once both distances are positive, the outermost
                                # radius has been reached
                                if d1 * d2 > 0:
                                    # set the two distances to be equal
                                    d1 = d2
                                else:
                                    break
                            # do the interpolation on adjacent radii to smooth the result

                            R = (radii[k] * d1 - radii[k - 1] * d2) / (d1 - d2)

                            # calculate the weight of the radius k-1
                            w1 = (radii[k - 1] - R) / (radii[k - 1] - radii[k])

                            # calculate the weight of the radius k
                            w2 = 1 - w1

                            # calculated the weighted quantities
                            posai = w1 * pa[k] + w2 * pa[k - 1]
                            incli = w1 * inc[k] + w2 * inc[k - 1]
                            vroti = w1 * vrot[k] + w2 * vrot[k - 1]
                            vradi = w1 * vrad[k] + w2 * vrad[k - 1]
                            vsysi = w1 * vsys[k] + w2 * vsys[k - 1]
                            if R == 0:
                                cost = 1
                                sint = 0
                            else:
                                cost = (-x * np.sin(posai) + y * np.cos(posai)) / R
                                sint = ((-x * np.cos(posai) - y * np.sin(posai)) / R) / np.cos(incli)

                            # calculate the rotation velocity
                            result[i, j] = vsysi + (vroti * cost + vradi * sint) * np.sin(incli)
                            pbar.update(1)

    if plot:
        fig, axes = make_figure(ncols=2, nrows=1, wcs=WCS(data_dict['header']), colorbar=True, **kwargs)
        fig, axes[0] = plot_mom1(data, cmap=cmap, fig=fig, ax=axes[0])
        fig, axes[1] = plot_mom1(result, cmap=cmap, fig=fig, ax=axes[1])
        axes[0].set_title('Original field')
        axes[1].set_title('Extended field')

    if output is not None:
        fits.PrimaryHDU(result, header=data_dict['header']).writeto(output, overwrite=True)

    if output is not None and not plot:
        return result
    if output is not None and plot:
        return fig, axes
    if plot:
        return result, fig, axes
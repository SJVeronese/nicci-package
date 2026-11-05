import matplotlib.colors as cl
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from astropy.wcs import WCS
from matplotlib.transforms import Bbox
from scipy.interpolate import interp1d
from scipy.special import erf
from tqdm.auto import tqdm

from nicci.io.fits import load_data
from nicci.utils.array import normalise, RMS
from nicci.utils.units import convert_from_HI, convert_to_HI


def chanmap(datacube, channel_step, *,
            first_channel=None, last_channel=None, n_channels=9,
            xmin=None, xmax=None, ymin=None, ymax=None,
            pixunits="deg", fluxunits='Jy/beam', specunits="m/s", pixelres=None,
            spectralres=None, z0=None, vsys=None,
            bmaj=None, bmin=None, bpa=None,
            use_column_density=True,
            fig_x_size=5, fig_y_size=5, wspace=0.01, hspace=0.01, dpi=300,
            wcs=None, cmap="Greys", vmin=None, vmax=None,
            norm="log", ctr="pow3", ctr_cmap="hot", ctr_width=1, plot_ctr=True,
            left_loc=0.05, right_loc=0.95, lower_loc=0.05, upper_loc=0.9,
            kpc=10, asectokpc=None, color="black", y_offset=0.02,
            colorbar_label=None,
            output=None):
    """Plot a grid of channel maps (2D slices) of a spectral data cube.

    Extracts a subcube in channel and spatial range, lays the selected
    channels out on a grid of panels, and overlays contours, a beam
    ellipse, a physical scale bar, and a colorbar.

    Parameters
    ----------
    datacube : str, ndarray, or tuple
        Input data cube, passed to `load_data`.
    channel_step : int
        Step between plotted channels.
    first_channel, last_channel : int, optional
        First and last channel of the plotted range. If not given, both
        are derived from the central channel and `n_channels`.
    n_channels : int, default 9
        Number of channels to plot when `first_channel`/`last_channel`
        are not given.
    xmin, xmax, ymin, ymax : int, optional
        Spatial subregion to plot, in pixel units.
    pixunits : str, default "deg"
        Units of the pixel scale ("deg" is converted to arcsec).
    fluxunits : str, default "Jy/beam"
        Flux units, used for the colorbar label.
    specunits : str, default "m/s"
        Spectral axis units ("m/s" is converted to km/s).
    pixelres, spectralres : float, optional
        Pixel scale and spectral resolution; read from the cube header
        if not given.
    z0, vsys : float, optional
        Reference (first-channel) velocity and systemic velocity.
    bmaj, bmin, bpa : float, optional
        Beam major axis, minor axis, and position angle; read from the
        cube header if not given.
    use_column_density : bool, default True
        If True (and the beam and spectral resolution are known), convert
        the contour levels to HI column density.
    fig_x_size, fig_y_size : float, default 5
        Size, in inches, of each panel.
    wspace, hspace : float, default 0.01
        Horizontal and vertical spacing between panels.
    dpi : int, default 300
        Resolution used when saving the figure.
    wcs : astropy.wcs.WCS, optional
        World coordinate system; read from the cube header if not given.
    cmap : str, default "Greys"
        Colormap used for the channel images.
    vmin, vmax : float, optional
        Normalization bounds; derived from the data if not given.
    norm : str, default "log"
        Normalization type, passed to `_norm`.
    ctr : str, array_like, or None, default "pow3"
        Contour levels, or a recipe passed to `_contours`.
    ctr_cmap : str, default "hot"
        Colormap used to color the contour levels.
    ctr_width : float, default 1
        Contour line width.
    plot_ctr : bool, default True
        Whether to overlay contours on each panel.
    left_loc, right_loc, lower_loc, upper_loc : float, default 0.05/0.95/0.05/0.9
        Axes-fraction anchor points for the velocity label, beam, and
        scale bar.
    kpc : float, default 10
        Length, in kpc, of the physical scale bar.
    asectokpc : float, optional
        Arcsec-to-kpc conversion factor; if given, a scale bar is drawn.
    color : str, default "black"
        Color used for text, beam, and scale bar annotations.
    y_offset : float, default 0.02
        Vertical offset of the scale bar label.
    colorbar_label : str, optional
        Colorbar label; a default based on `fluxunits` is used if not given.
    output : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axis_list : list of matplotlib.axes.Axes
        The panel axes, including any hidden unused ones.
    """

    # load the data
    data_dict = load_data(datacube)

    # store the relevant info
    data = data_dict['data']
    if wcs is None:
        wcs = data_dict['wcs']
    if wcs is not None and wcs.naxis == 3:
        wcs = wcs.dropaxis(2)
    if bmaj is None:
        bmaj = data_dict['bmaj']
    if bmin is None:
        bmin = data_dict['bmin']
    if bpa is None:
        bpa = data_dict['bpa']
    if fluxunits is None:
        fluxunits = data_dict['fluxunits']
    if pixunits is None:
        pixunits = data_dict['pixunits']
    if specunits is None:
        specunits = data_dict['specunits']
    if pixelres is None:
        pixelres = data_dict['pixelres']
    if spectralres is None:
        spectralres = data_dict['spectralres']
    if z0 is None:
        z0 = data_dict['z0']
    if colorbar_label is None:
        colorbar_label = f'Flux [{fluxunits}]'
    if any(x is None for x in (bmaj, bmin, spectralres)):
        use_column_density = False

    # convert the spatial units
    pixres_arcsec = pixelres
    bmaj_arcsec = bmaj
    bmin_arcsec = bmin
    if pixunits == "deg":
        if pixres_arcsec is not None:
            pixres_arcsec *= 3600
        if bmaj_arcsec is not None:
            bmaj_arcsec *= 3600
        if bmin_arcsec is not None:
            bmin_arcsec *= 3600

    # convert spectral units
    specres_kms = spectralres
    z0_kms = z0
    vsys_kms = vsys
    if specunits == "m/s":
        if specres_kms is not None:
            specres_kms /= 1000
        if z0_kms is not None:
            z0_kms /= 1000
        if vsys_kms is not None:
            vsys_kms /= 1000

    # create the velocity range
    v = (
        z0_kms + np.arange(data.shape[0]) * specres_kms
        if all(x is not None for x in (z0_kms, specres_kms))
        else None
    )

    # use full spatial range if not specified
    if xmin is None or xmin < 0:
        xmin = 0
    if xmax is None or xmax > data.shape[2]:
        xmax = data.shape[2]
    if ymin is None or ymin < 0:
        ymin = 0
    if ymax is None or ymax > data.shape[1]:
        ymax = data.shape[1]

    # determine the channel range
    if first_channel is not None and first_channel < 0:
        first_channel = 0
    if last_channel is not None and last_channel > data.shape[0]:
        last_channel = data.shape[0]
    if any(x is None for x in (first_channel, last_channel)):  # if no channel range is set, determine it from the central channel
        central_channel = (
            np.argmin(np.abs(v - vsys_kms))
            if all(x is not None for x in (v, vsys_kms))
            else data.shape[0] // 2
        )
        first_channel = central_channel - n_channels // 2 * channel_step
        last_channel = central_channel + n_channels // 2 * channel_step
    else:
        n_channels = (last_channel - first_channel) // channel_step

    # extract the subcube
    data = data[first_channel : last_channel + 1 : channel_step, ymin:ymax, xmin:xmax]
    if wcs is not None:
        wcs = wcs[ymin:ymax, xmin:xmax]
    if v is not None:
        v = v[first_channel : last_channel + 1 : channel_step]

    # determine number of channels and grid layout
    ncols = int(np.ceil(np.sqrt(n_channels)))
    nrows = int(np.ceil(n_channels / ncols))

    # define the normalization
    if vmin is None:
        vmin = 0.001 * np.nanmin(data[data != 0])
    if vmax is None:
        vmax = 0.9 * np.nanmax(data)
    norm = _norm(norm, vmin, vmax)

    # define the contours
    if isinstance(ctr, (list, np.ndarray)) and use_column_density:
        ctr = convert_from_HI(np.asarray(ctr) * 10.0**19,
                              beamarea=1.13 * bmaj_arcsec * bmin_arcsec,
                              spectralres=np.abs(specres_kms),
                              from_moment_map=False)  # convert back into Jy/beam
    else:
        ctr = _contours(ctr, data, vmax=vmax, norm=norm)

    # create the figure
    fig, axis_list = make_figure(nrows=nrows, ncols=ncols,
                                 fig_x_size=fig_x_size, fig_y_size=fig_y_size,
                                 wspace=wspace, hspace=hspace,
                                 wcs=wcs,
                                 colorbar=True)

    # hide unused axes
    for i in range(n_channels, len(axis_list)):
        axis_list[i].set_visible(False)

    # loop over channels and plot each panel
    for idx in range(n_channels):
        row = idx // ncols
        col = idx % ncols

        ax = axis_list[idx]
        im = ax.imshow(data[idx], origin='lower', cmap=cmap, norm=norm, aspect='equal')
        # add the contours
        if plot_ctr:
            ax.contour(data[idx], levels=ctr, cmap=ctr_cmap, linewidths=ctr_width, linestyles='solid')
            ax.contour(data[idx], levels=-np.flip(ctr), colors='gray', linewidths=ctr_width, linestyles='dashed')  # add the negative contours
        # add the channel velocity
        if vsys_kms is None:
            vsys_kms = 0
        if v is not None:
            ax.text(left_loc, lower_loc, f'v$_{{los}}$: {v[idx]-vsys_kms:.1f} km/s',
                    color=color, transform=ax.transAxes)
        # add the beam
        if all(x is not None for x in (pixres_arcsec, bmaj_arcsec, bmin_arcsec, bpa)):
            ax = _plot_beam(ax, pixres_arcsec,
                            bmaj_arcsec, bmin_arcsec, bpa,
                            xloc=right_loc, yloc=lower_loc, color=color)
        # add the kpc line
        if all(x is not None for x in (pixres_arcsec, asectokpc)):
            if col == 0:  # in the first column
                ax = _plot_kpc_scale(ax, pixres_arcsec, asectokpc, kpc=kpc,
                                     xloc=left_loc, yloc=upper_loc, y_offset=y_offset,
                                     color=color)

        # add the colorbar
        if row == 0:  # in the first row
            if use_column_density and all(
                x is not None for x in (bmaj_arcsec, bmin_arcsec, specres_kms)
            ):
                # convert the contour levels to column density in units of 10^19 cm^-2
                ctr_label = np.around(
                    convert_to_HI(ctr,
                                  beamarea=1.13 * bmaj_arcsec * bmin_arcsec,
                                  spectralres=np.abs(specres_kms),
                                  from_moment_map=False)
                                  / 10.0**19, 2)
                colorbar_label = r'HI column density [$10^{19}$ cm$^{-2}$]'
            else:
                ctr_label = None
            ax = _add_colorbar(fig, ax, im, norm, ctr,
                               label=colorbar_label, ctr_cmap=ctr_cmap, ctr_label=ctr_label)

    if output is not None:
        fig.savefig(output, bbox_inches='tight', dpi=dpi)  # save the figure
    return fig, axis_list  # return the figure and axis list
    
def make_figure(**kwargs):
    """Create a (optionally multi-panel) WCS-aware figure with labelled axes.

    A thin, keyword-driven wrapper around `matplotlib.pyplot.figure` that
    sets up one axis (``nrows == ncols == 1``) or a grid of axes sharing
    consistent axis labels, ticks, limits, and scales, with special
    handling of axis-label placement when a colorbar is present.

    Parameters
    ----------
    **kwargs
        fig_x_size, fig_y_size : float, default 7
            Size, in inches, of each panel.
        nrows, ncols : int, default 1
            Number of panel rows and columns.
        wcs : astropy.wcs.WCS, optional
            World coordinate system used as the axes projection.
        wspace, hspace : float, default 0.01
            Horizontal and vertical spacing between panels.
        xticks, yticks : array_like, optional
            Tick positions shared by all panels.
        xlabels, ylabels : str or list of str, optional
            Axis label(s). A single string is used for every panel; a
            list is matched to panels in reading order (row-major),
            repeating the last entry if shorter than the number of panels.
        xticklabels, yticklabels : array_like, optional
            Tick label text shared by all panels.
        xlim, ylim : tuple, optional
            Axis limits shared by all panels.
        xscale, yscale : str or list of str, optional
            Axis scale(s) (e.g. ``'log'``), matched to panels like
            `xlabels`/`ylabels`.
        colorbar : bool, default False
            Whether a colorbar will be added, which affects where axis
            labels are placed in the multi-panel case.
        Aliases ``figx``, ``figy``, ``nrow``, ``ncol``, ``xlabel``,
        ``ylabel``, ``xticklabel``, and ``yticklabel`` are also accepted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes or list of matplotlib.axes.Axes
        A single axis if ``nrows == ncols == 1``, otherwise a flat list
        of axes in row-major order. If neither `nrows` nor `ncols` is
        greater than 1 and both equal 1 is false (degenerate case), only
        the figure is returned.
    """

    # handle parameter
    aliases = {
        'figx': 'fig_x_size',
        'figy': 'fig_y_size',
        'nrow': 'nrows',
        'ncol': 'ncols',
        'xlabel': 'xlabels',
        'ylabel': 'ylabels',
        'xticklabel': 'xticklabels',
        'yticklabel': 'yticklabels',
    }

    # map aliases to canonical names
    for alias, canonical in aliases.items():
        if alias in kwargs and canonical not in kwargs:
            kwargs[canonical] = kwargs[alias]

    # extract parameters
    figx = kwargs.get("fig_x_size", 7)
    figy = kwargs.get("fig_y_size", 7)
    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", 1)
    wcs = kwargs.get("wcs", None)
    wspace = kwargs.get("wspace", 0.01)
    hspace = kwargs.get("hspace", 0.01)
    xticks = kwargs.get("xticks", None)
    yticks = kwargs.get("yticks", None)
    xlabel = kwargs.get("xlabels", None)
    ylabel = kwargs.get("ylabels", None)
    xticklabels = kwargs.get("xticklabels", None)
    yticklabels = kwargs.get("yticklabels", None)
    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    xscale = kwargs.get("xscale", None)
    yscale = kwargs.get("yscale", None)
    colorbar = kwargs.get("colorbar", False)

    # helper function to pick the appropriate label for each panel
    def pick(label, idx):
        """Select the label for panel `idx` from a scalar or per-panel list.

        Parameters
        ----------
        label : object or list
            A single value shared by all panels, or a list of per-panel
            values.
        idx : int
            Index of the panel whose label is requested.

        Returns
        -------
        object
            ``label[idx]`` if `label` is a list (clamped to the last entry
            if `idx` is out of range), otherwise `label` itself.
        """
        # if multiple labels are provided
        if isinstance(label, list):
            # return the corresponding label or the last one if idx exceeds the list length
            return label[idx] if idx < len(label) else label[-1]
        # if a single label is provided
        else:
            return label

    # create figure
    fig = plt.figure(figsize=(figx * ncols, figy * nrows))

    # single panel case
    if nrows == 1 and ncols == 1:
        # create the axis
        ax = fig.add_subplot(projection=wcs)

        # manage axis labels
        (
            ax.set_xlabel(xlabel)
            if xlabel is not None
            else ax.set_xlabel('RA [J2000]' if wcs is not None else 'X [px]')
        )
        (
            ax.set_ylabel(ylabel)
            if ylabel is not None
            else ax.set_ylabel('DEC [J2000]' if wcs is not None else 'Y [px]')
        )

        # manage ticks
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

        # manage ticklabels
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)

        # manage limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # manage the axis scale
        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)

        return fig, ax

    # multiple panel case
    elif nrows > 1 or ncols > 1:
        # create grid specification
        gs = fig.add_gridspec(nrows, ncols, wspace=wspace, hspace=hspace)

        # create the panels
        axis_list = []  # list to store the axes
        xlabel_idx, ylabel_idx = 0, 0  # indices to track xlabel and ylabel for multiple labels
        for idx in range(nrows * ncols):
            # determine row and column index
            row = idx // ncols
            col = idx % ncols

            # create the axis
            ax = fig.add_subplot(gs[row, col], projection=wcs)

            # manage ticks
            if xticks is not None:
                ax.set_xticks(xticks)
            if yticks is not None:
                ax.set_yticks(yticks)

            # manage ticklabels
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels)
            if yticklabels is not None:
                ax.set_yticklabels(yticklabels)

            # manage limits
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            # manage the axis scale
            if xscale is not None:
                ax.set_xscale(pick(xscale, xlabel_idx))
            if yscale is not None:
                ax.set_yscale(pick(yscale, ylabel_idx))

            # manage x-axis labels
            if nrows > 1:
                if (not colorbar and row == 0):  # if plotting the first row and a colorbar will not be added
                    (
                        ax.set_xlabel(pick(xlabel, xlabel_idx))
                        if xlabel is not None
                        else ax.set_xlabel('RA [J2000]' if wcs is not None else 'X [px]')
                    )
                    if wcs is not None:
                        ax.coords[0].set_ticklabel_position(
                            't'
                        )  # move the x-axis label to the top
                        ax.coords[0].set_axislabel_position(
                            't'
                        )  # move the x-axis tick labels to the top
                    else:
                        ax.xaxis.set_label_position('top')  # move the x-axis label to the top
                        ax.xaxis.tick_top()  # move the x-axis tick labels to the top
                elif row == nrows - 1:  # if plotting the last row
                    (
                        ax.set_xlabel(pick(xlabel, xlabel_idx))
                        if xlabel is not None
                        else ax.set_xlabel('RA [J2000]' if wcs is not None else 'X [px]')
                    )
                else:
                    # hide the x-axis ticklabels and labels
                    if wcs is not None:
                        ax.coords[0].set_ticklabel_visible(False)
                    else:
                        ax.set_xlabel('')
                        ax.set_xticklabels([])
            else:
                (
                    ax.set_xlabel(pick(xlabel, xlabel_idx))
                    if xlabel is not None
                    else ax.set_xlabel('RA [J2000]' if wcs is not None else 'X [px]')
                )
            xlabel_idx += 1

            # manage y-axis labels
            if ncols > 1:
                if col == 0:  # if plotting the first column
                    (
                        ax.set_ylabel(pick(ylabel, ylabel_idx))
                        if ylabel is not None
                        else ax.set_ylabel('DEC [J2000]' if wcs is not None else 'Y [px]')
                    )
                elif col == ncols - 1:  # if plotting the last column
                    (
                        ax.set_ylabel(pick(ylabel, ylabel_idx))
                        if ylabel is not None
                        else ax.set_ylabel('DEC [J2000]' if wcs is not None else 'Y [px]')
                    )
                    if wcs is not None:
                        ax.coords[1].set_ticklabel_position('r')  # move the y-axis label to the right
                        ax.coords[1].set_axislabel_position('r')  # move the y-axis tick labels to the right
                    else:
                        ax.yaxis.set_label_position('right')  # move the y-axis label to the right
                        ax.yaxis.tick_right()  # move the y-axis tick labels to the right
                else:
                    if wcs is not None:
                        ax.coords[1].set_ticklabel_visible(False)  # hide the y-axis ticklabels and labels
                    else:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])
            else:
                (
                    ax.set_ylabel(pick(ylabel, ylabel_idx))
                    if ylabel is not None
                    else ax.set_ylabel('DEC [J2000]' if wcs is not None else 'Y [px]')
                )
            ylabel_idx += 1

            if wcs is None:
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')

            # store the axis
            axis_list.append(ax)
        return fig, axis_list

    # just the figure case
    else:
        return fig
        
def plot_mom0(data, **kwargs):
    """Plot a moment-0 (integrated intensity) map. See `_plot_mom`."""

    return _plot_mom(data, which="mom0", **kwargs)


def plot_mom1(data, **kwargs):
    """Plot a moment-1 (velocity field) map. See `_plot_mom`."""

    return _plot_mom(data, which="mom1", **kwargs)


def plot_mom2(data, **kwargs):
    """Plot a moment-2 (velocity dispersion) map. See `_plot_mom`."""

    return _plot_mom(data, which="mom2", **kwargs)


def plot_noise(datacube, *,
               statistics='mad', flux_range='negative',
               mult_by=1, add_to=0,
               norm='linear', vmin=None, vmax=None,
               cmap='Greys', color='blue', colorbar=True, flux_units='',
               fig_x_size=7, fig_y_size=7,  dpi=300,
               show_progress_bar=True, output=None):
    """Compute and plot the spatial and per-channel noise of a data cube.

    Parameters
    ----------
    datacube : str, ndarray, or tuple
        Input data cube, passed to `load_data`.
    statistics : {'mad', 'std'}, default 'mad'
        Statistic used to estimate the noise, passed to `RMS`.
    flux_range : {'negative', 'positive', 'all'}, default 'negative'
        Flux range used to estimate the noise, passed to `RMS`.
    norm : str, default 'linear'
        Normalization type for the spatial noise map, passed to `_norm`.
    vmin, vmax : float, optional
        Normalization bounds for the spatial noise map; derived from the
        data if not given.
    cmap : str, default 'Greys'
        Colormap used for the spatial noise map.
    color : str, default 'blue'
        Line color for the per-channel noise plot.
    colorbar : bool, default True
        Whether to add a colorbar to the spatial noise map.
    flux_units : str, default ''
        Flux units, used for axis/colorbar labels. Read from the cube
        header if not given.
    fig_x_size, fig_y_size : float, default 7
        Figure size, in inches.
    mult_by : float, default 1
        Multiplicative scaling applied to the cube before computing noise.
    add_to : float, default 0
        Additive offset applied to the cube before computing noise.
    dpi : int, default 300
        Resolution used when saving the figure.
    output : str, optional
        If given, the figure is saved to this path.
    show_progress_bar : bool, default True
        Whether to display `tqdm` progress bars.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure, with the spatial noise map on top and the
        per-channel noise curve below.
    """

    # load the data
    data_dict = load_data(datacube)

    # store the relevant info
    data, header = data_dict['data'], data_dict['header']

    nchan, nx, ny = data.shape
    if flux_units == '' and header != {} and header is not None:
        flux_units = header['BUNIT']

    # apply optional scaling
    data = data * mult_by + add_to

    # compute the spectral noise
    spectral_rms = np.zeros(nchan)

    for i in tqdm(range(nchan), desc="Computing spectral RMS", disable=not show_progress_bar):
        spectral_rms[i] = RMS(data[i], mode=statistics, fluxrange=flux_range)

    # compute the spatial noise
    spatial_rms = np.zeros((nx, ny))
    flat_data = data.reshape(nchan, nx * ny)

    for idx in tqdm(range(nx * ny), desc="Computing spatial RMS", disable=not show_progress_bar):
        spatial_rms.flat[idx] = RMS(flat_data[:, idx], mode=statistics, fluxrange=flux_range)

    # clean up glitchy pixels
    spatial_rms[spatial_rms == 0] = np.nan

    # create the figure
    nrows, ncols = 2, 1
    fig = plt.figure(figsize=(fig_x_size * ncols, fig_y_size * nrows))
    gs = fig.add_gridspec(nrows + 1, ncols, hspace=0.1)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_xlabel('X [px]')
    ax1.set_ylabel('Y [px]')

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_xlabel('Channel')
    ax2.set_ylabel(f'Noise [{flux_units}]')

    # define the normalization
    if vmin is None:
        vmin = 0.001 * np.nanmin(data[data != 0])
    if vmax is None:
        vmax = 0.9 * np.nanmax(data)
    norm = _norm(norm, vmin, vmax)

    # plot the spatial rms
    im = ax1.imshow(spatial_rms, origin='lower', cmap=cmap)

    if colorbar:
        _add_colorbar(fig, ax1, im, norm, None, label=f'Noise [{flux_units}]')

    # plot the spectral rms
    ax2.plot(np.arange(nchan) + 1, spectral_rms, c=color)

    if output is not None:
        fig.savefig(output, bbox_inches='tight', dpi=dpi)  # save the figure
    return fig


def plot_pvslice(pv_data, pv_header, *,
                 mult_by=1, add_to=0,
                 pixunits="deg", fluxunits='Jy/beam',
                 pixelres=None, vsys=None,
                 fig=None, ax=None, fig_x_size=7, fig_y_size=7, dpi=300,
                 cmap="Greys", vmin=None, vmax=None, norm="log",
                 ctr="pow3", ctr_cmap="hot", ctr_width=1,
                 left_loc=0.05, upper_loc=0.9, kpc=10, asectokpc=None, color="black",
                 y_offset=0.02, colorbar=True, colorbar_label=None, output=None):
    """Plot a position-velocity (PV) slice with contours and annotations.

    Parameters
    ----------
    pv_data : ndarray
        2D PV slice data (distance along the path x velocity).
    pv_header : astropy.io.fits.Header
        Header describing the PV slice WCS.
    mult_by : float, default 1
        Multiplicative scaling applied to the data before plotting.
    add_to : float, default 0
        Additive offset applied to the data before plotting.
    pixunits : str, default 'deg'
        Spatial pixel units ('deg' is converted to arcsec).
    fluxunits : str, default 'Jy/beam'
        Flux units, used for the colorbar label. Read from the header if
        not given.
    pixelres : float, optional
        Spatial pixel resolution; derived from the WCS if not given.
    vsys : float, optional
        Systemic velocity used to recentre the velocity axis.
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes, optional
        Existing figure/axis to plot into. Created if not given.
    fig_x_size, fig_y_size : float, default 7
        Figure size, in inches, used if `fig` is not given.
    dpi : int, default 300
        Resolution used when saving the figure.
    cmap : str, default 'Greys'
        Colormap used for the image.
    vmin, vmax : float, optional
        Normalization bounds; derived from the data if not given.
    norm : str, default 'log'
        Normalization type, passed to `_norm`.
    ctr : str or array_like, default 'pow3'
        Contour levels, or a recipe passed to `_contours`.
    ctr_cmap : str, default 'hot'
        Colormap used to color the contour levels.
    ctr_width : float, default 1
        Contour line width.
    left_loc, upper_loc : float, default 0.05, 0.9
        Axes-fraction anchor point for the physical scale bar.
    kpc : float, default 10
        Length, in kpc, of the physical scale bar.
    asectokpc : float, optional
        Arcsec-to-kpc conversion factor; if given, a scale bar is drawn.
    color : str, default 'black'
        Color used for the scale bar annotation.
    y_offset : float, default 0.02
        Vertical offset of the scale bar label.
    colorbar : bool, default True
        Whether to add a colorbar.
    colorbar_label : str, optional
        Colorbar label; a default based on `fluxunits` is used if not
        given.
    output : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The plot axis.
    """

    # copy the header to avoid modifying the original
    header = pv_header.copy()

    # apply data scaling
    data = pv_data.copy()
    data = data * mult_by + add_to

    # store the flux units
    if fluxunits is None:
        fluxunits = header.get('BUNIT')
    if colorbar_label is None:
        colorbar_label = f'Flux [{fluxunits}]'

    # create the WCS object from the header
    wcs = WCS(header)

    # compute the pixel scale
    if pixelres is None:
        scales = np.diag(wcs.pixel_scale_matrix)  # extract the pixel scales
        pixelres = np.abs(scales[0])  # pixel resolution

    # set the pixel units
    if pixunits is None:
        pixunits = header.get('CUNIT1')

    # create the figure and axis if not provided
    if fig is None:
        fig = plt.figure(figsize=(fig_x_size, fig_y_size))
    if ax is None:
        if vsys is not None:
            header['CRVAL2'] -= vsys  # adjust the systemic velocity in the header
        ax = fig.add_subplot(projection=WCS(header))

    # define the normalization
    if vmin is None:
        vmin = 0.001 * np.nanmin(data[data != 0])
    if vmax is None:
        vmax = 0.9 * np.nanmax(data)
    norm = _norm(norm, vmin, vmax)

    # define the contours
    ctr = _contours(ctr, data, vmax=vmax, norm=norm)

    # plot the pv data
    im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm)

    # add the contours
    ax.contour(data, levels=ctr, cmap=ctr_cmap, linewidths=ctr_width, linestyles='solid')  # add the contours
    ax.contour(data, levels=-np.flip(ctr), colors='gray', linewidths=ctr_width, linestyles='dashed')  # add the negative contours

    # set x-axis labels
    if asectokpc is not None:
        ax.coords[0].set_format_unit((u.arcsec / asectokpc))  # convert x units to kpc
        ax.set_xlabel('Distance [kpc]')
    else:
        ax.coords[0].set_format_unit(u.arcsec)  # convert x units to arcsec
        ax.set_xlabel('Distance [arcsec]')

    # draw the path center line
    ax.axvline(x=header['CRPIX1'] - 1, ls='-.', c='black')  # -1 is due to python stupid 0-couting

    # set y-axis labels
    if header['CUNIT2'] == 'm/s':  # if the spectral units are m/s
        ax.coords[1].set_format_unit(u.km / u.s)  # convert to km/s
        ax.set_ylabel(r'LoS velocity [km s$^{-1}$]')
    if vsys is not None:  # if the systemic velocity is given
        ax.axhline(
            y=((0 - header['CRVAL2']) / header['CDELT2']) + header['CRPIX2'] - 1,
            ls='--',
            c='black',
        )  # draw the systemic velocity line
        ax.set_ylabel(r'LoS velocity w.r.t. v$_{sys}$ [km s$^{-1}$]')

    # add the kpc line
    pixres_arcsec = pixelres
    if pixunits == "deg":
        # convert spatial units
        if pixres_arcsec is not None:
            pixres_arcsec *= 3600
    if all(x is not None for x in (pixres_arcsec, asectokpc)):
        ax = _plot_kpc_scale(ax, pixres_arcsec, asectokpc, kpc=kpc,
                             xloc=left_loc, yloc=upper_loc, color=color, y_offset=y_offset)  # draw the kpc line

    # add the colorbar
    if colorbar:
        if (header != {} and header is not None) and colorbar_label is None:
            colorbar_label = header.get('BUNIT')
        ax = _add_colorbar(fig, ax, im, norm, ctr, label=colorbar_label, ctr_cmap=ctr_cmap)

    # save the figure
    if output is not None:
        fig.savefig(output, bbox_inches='tight', dpi=dpi)
    return fig, ax  # return the figure and axis list
    
def _add_colorbar(fig, ax, im, norm, ctr, *,
                  label=None, ctr_cmap=None, ctr_label=None, mode='colormap',
                  **kwargs):
    """Add a horizontal colorbar with contour-level tick marks to an axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure the colorbar is attached to.
    ax : matplotlib.axes.Axes
        Axis the colorbar is placed above.
    im : matplotlib.image.AxesImage
        Image (e.g. from `ax.imshow`) the colorbar maps.
    norm : matplotlib.colors.Normalize
        Normalization used to place contour-level ticks on the colorbar.
    ctr : array_like or None
        Contour levels to mark as ticks; no ticks are added if None.
    label : str, optional
        Colorbar label.
    ctr_cmap : str, optional
        Colormap used to color each contour-level tick (``mode='colormap'``).
    ctr_label : array_like, optional
        Custom tick labels for the contour levels; `ctr` rounded to two
        decimals is used if not given.
    mode : {'colormap', 'posneg'}, default 'colormap'
        How to mark contour levels: colored tick lines from `ctr_cmap`
        (``'colormap'``), or separate solid/dashed lines for positive and
        negative levels (``'posneg'``).
    **kwargs
        ``ctr_pos_color`` and ``ctr_neg_color`` (used only in
        ``mode='posneg'``): colors for positive and negative contour
        level lines, default ``'black'`` and ``'gray'``.

    Returns
    -------
    matplotlib.axes.Axes
        The input axis `ax`, unchanged (the colorbar is drawn as a side
        effect).
    """

    # create colorbar axis
    cax = ax.inset_axes([0, 1, 1, 0.05])

    # add the colorbar
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')

    # set the ticks position
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.tick_top()

    # add the title
    if label is not None:
        cb.set_label(label)

    # set the countours
    if ctr is not None:
        cb.set_ticks(ctr)
        cb.set_ticklabels(np.around(ctr, 2)) if ctr_label is None else cb.set_ticklabels(ctr_label)
        if mode == 'colormap' and ctr_cmap is not None:
            # assign the color of the countour levels to the correspondong tick
            for tick, tick_color in zip(ctr, plt.cm.get_cmap(ctr_cmap)(normalise(ctr, 0, 1))):
                # create the line
                cb.ax.plot([norm(tick), norm(tick)], [0, 1], c=tick_color, lw=1, transform=cb.ax.transAxes)
        elif mode == 'posneg':
            ctr_pos = ctr[ctr > 0]  # get the positive contours
            ctr_neg = ctr[ctr < 0]  # get the negative contours
            ctr_pos_color = kwargs.get('ctr_pos_color', 'black')  # color for positive contours
            ctr_neg_color = kwargs.get('ctr_neg_color', 'gray')  # color for negative contours
            # for each positive contour
            for tick in ctr_pos:
                # create the line for each contour level
                cb.ax.axvline(tick, c=ctr_pos_color, lw=1)
            # for each negative contour
            for tick in ctr_neg:
                # create the line for each contour level
                cb.ax.axvline(tick, c=ctr_neg_color, lw=1, ls='dashed')

    return ax


def _auto_fit_fontsize(text, *,
                       width=None, height=None, fig=None, ax=None, min_fontsize=1):
    """Shrink a text object's font size until it fits within given bounds.

    Parameters
    ----------
    text : matplotlib.text.Text
        Text object whose font size is adjusted in place.
    width, height : float, optional
        Maximum allowed width/height, in data coordinates. A bound is
        ignored if not given.
    fig : matplotlib.figure.Figure, optional
        Figure used to obtain a renderer. Defaults to the current figure.
    ax : matplotlib.axes.Axes, optional
        Axis used to convert display to data coordinates. Defaults to the
        current axis.
    min_fontsize : float, default 1
        Smallest font size to try before giving up.

    Returns
    -------
    float
        The font size at which the text fits within the given bounds.

    Raises
    ------
    ValueError
        If the text still does not fit at `min_fontsize`.
    """

    # get the figure and axes to use
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    # force a draw so that a renderer exists and text size can be measured reliably
    fig.canvas.draw()

    # obtain the renderer responsible for converting artists into pixel-based representations
    renderer = fig.canvas.get_renderer()

    # get the current font size
    fontsize = text.get_fontsize()

    # iteratively decrease the font size until the text fits within the constraints,
    # or until the minimum font size is reached
    while fontsize >= min_fontsize:
        # measure the text bounding box in display (pixel) coordinates
        bbox = text.get_window_extent(renderer=renderer)

        # convert the bounding box from display coordinates into the axes data coordinate system
        bbox = Bbox(ax.transData.inverted().transform(bbox))

        # determine whether the text width satisfies the constraint if one is specified
        fits_width = True if width is None else bbox.width <= width

        # determine whether the text height satisfies the constraint if one is specified
        fits_height = True if height is None else bbox.height <= height

        # exit successfully once the text fits within both width and height constraints
        if fits_width and fits_height:
            return fontsize

        # decrease the font size by one point and apply it to the text object
        fontsize -= 1
        text.set_fontsize(fontsize)

    # raise error if the text cannot fit within the constraints at any reasonable size
    raise ValueError("Text cannot fit within the given dimensions")


def _contours(ctr, data, *,
              vmax=None, norm=None):
    """Compute contour levels from a recipe, a normalization, or explicit values.

    Parameters
    ----------
    ctr : str or array_like
        Recipe for generating contour levels: a string of the form
        ``'pow<N>'`` for levels spaced as powers of ``N`` in units of the
        noise (see Notes), the string ``'even'`` for levels evenly spaced
        in the given normalization, or an explicit array of levels.
    data : ndarray
        Data used to estimate the noise (for the ``'pow<N>'`` recipe).
    vmax : float, optional
        Maximum data value, used to cap the ``'pow<N>'`` levels at the
        data's significance.
    norm : matplotlib.colors.Normalize, optional
        Normalization used to space levels evenly for the ``'even'``
        recipe.

    Returns
    -------
    ndarray
        The computed or passed-through contour levels.

    Notes
    -----
    For the ``'pow<N>'`` recipe, five levels are generated as
    ``N ** linspace(1, log(0.9 * max_sigma) / log(N), 5)`` times the noise,
    where ``max_sigma = round(vmax / rms)``.
    """

    if isinstance(ctr, str) and 'pow' in ctr:
        # calculate the rms
        rms = RMS(data)

        # calculate the maximum sigma in the data
        max_sigma = int(np.around(vmax / rms, 0))

        # determine the power
        power = int(ctr.split('pow')[-1])

        # five evenly distributed powers between 'power' and max_sigma
        ctr = np.around(power ** np.linspace(1, np.log(0.9 * max_sigma) / np.log(power), 5), 1)

        # contour levels in terms of rms
        ctr *= rms
    elif isinstance(ctr, str) and ctr == 'even':
        # define five evenly spaced contours in the used normalization
        ctr = norm.inverse(np.linspace(0, 1, 7))[1:-1]
    else:
        # convert into an array
        ctr = np.asarray(ctr)

    return ctr


def _covariance_to_error_ellipse(covariance_matrix, idx1, idx2):
    """Derive a 2D error ellipse from a covariance matrix.

    Parameters
    ----------
    covariance_matrix : ndarray, shape (N, N)
        Full covariance matrix.
    idx1, idx2 : int
        Indices of the two parameters defining the 2D subspace.

    Returns
    -------
    radius_maj : float
        Semi-axis length along the major eigenvector (1-sigma).
    radius_min : float
        Semi-axis length along the minor eigenvector (1-sigma).
    pa : float
        Position angle of the major axis, in degrees.
    """

    # extract the xx component of the covariance matrix
    cov_xx = covariance_matrix[idx1, idx1]

    # extract the yy component of the covariance matrix
    cov_yy = covariance_matrix[idx2, idx2]

    # extract the xy component of the covariance matrix
    cov_xy = covariance_matrix[idx1, idx2]

    # calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # standard deviation along one axis
    radius_maj = (
        np.sqrt(eigenvalues[0]) if eigenvalues[0] > eigenvalues[1] else np.sqrt(eigenvalues[1])
    )

    # standard deviation along the other axis
    radius_min = (
        np.sqrt(eigenvalues[1]) if eigenvalues[0] > eigenvalues[1] else np.sqrt(eigenvalues[0])
    )

    # if there is no covariance between x and y, the ellipse is aligned with the axes
    if cov_xy == 0:
        pa = 0.0
    else:
        # calculate the position angle (in degrees) of the major axis
        pa = np.degrees(np.arctan2(2 * cov_xy, cov_xx - cov_yy) / 2)

    return radius_maj, radius_min, pa


def _norm(norm, vmin, vmax):
    """Build a matplotlib normalization object.

    Parameters
    ----------
    norm : str
        ``'log'`` for a `matplotlib.colors.PowerNorm` with gamma=0.3
        (approximating a log stretch while tolerating non-positive
        values), any other value for a linear
        `matplotlib.colors.Normalize`.
    vmin, vmax : float
        Lower and upper normalization bounds.

    Returns
    -------
    matplotlib.colors.Normalize
        The constructed normalization object.
    """

    if norm == 'log':
        norm = cl.PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
    else:
        norm = cl.Normalize(vmin=vmin, vmax=vmax)
    return norm

def _plot_beam(ax, pixelres, bmaj, bmin, bpa, xloc=0.9, yloc=0.1, color='black'):
    """Draw a beam ellipse patch on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    pixelres : float
        Pixel resolution, in the same units as `bmaj`/`bmin`.
    bmaj, bmin : float
        Beam major and minor axis, in the same units as `pixelres`.
    bpa : float
        Beam position angle, in degrees (celestial convention).
    xloc, yloc : float, default 0.9, 0.1
        Beam centre, in axes fraction coordinates.
    color : str, default 'black'
        Beam patch color.

    Returns
    -------
    matplotlib.axes.Axes
        The input axis `ax`, with the beam patch added.
    """

    # convert from beam units to pixels
    width_px = bmaj / pixelres
    height_px = bmin / pixelres

    # convert from pixels to axis fractions
    xpix = np.diff(ax.get_xlim())[0]
    ypix = np.diff(ax.get_ylim())[0]

    width = width_px / xpix
    height = height_px / ypix

    # convert angle from celestial to python definition
    bpa += 90

    # define the beam patch
    beam = patch.Ellipse((xloc, yloc), width, height, angle=bpa, color=color, transform=ax.transAxes)

    # add the beam
    ax.add_patch(beam)

    return ax


def _plot_kpc_scale(ax, pixelres, asectokpc, *, xloc=0.05, yloc=0.95, kpc=10, color="black", y_offset=0.02):
    """Draw a physical (kpc) scale bar with a label on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    pixelres : float
        Pixel resolution, in arcsec/pixel.
    asectokpc : float
        Arcsec-to-kpc conversion factor at the source distance.
    xloc, yloc : float, default 0.05, 0.95
        Left end of the scale bar, in axes fraction coordinates.
    kpc : float, default 10
        Length of the scale bar, in kpc.
    color : str, default 'black'
        Line and label color.
    y_offset : float, default 0.02
        Vertical offset of the bar below `yloc`.

    Returns
    -------
    matplotlib.axes.Axes
        The input axis `ax`, with the scale bar added.
    """

    # convert kpc → pixels → axes fraction
    kpc_in_pixels = kpc / (asectokpc * pixelres)
    kpc_in_axes = kpc_in_pixels / np.diff(ax.get_xlim())[0]

    # draw line
    ax.hlines(yloc - y_offset, xloc, xloc + kpc_in_axes, color=color, linewidth=2, transform=ax.transAxes)

    # draw label
    ax.text(xloc + kpc_in_axes / 2, yloc,
            f"{kpc} kpc",
            ha="center", va="bottom", color=color, transform=ax.transAxes)

    return ax


def _plot_mom(data, *,
              which,
              mult_by=1, add_to=0,
              use_column_density=False,  # only used for mom0
              xmin=None, xmax=None, ymin=None, ymax=None,
              wcs=None, bmaj=None, bmin=None, bpa=None,
              pixunits='deg', fluxunits=None, pixelres=None,
              fig=None, ax=None, fig_x_size=7, fig_y_size=7, dpi=300,
              cmap=None, vmin=None, vmax=None, norm=None,
              plot_ctr=True, ctr=None, ctr_cmap=None, ctr_width=1, ctr_color_positive='black', ctr_color_negative='gray',  # only used for mom1
              left_loc=0.05, right_loc=0.95, lower_loc=0.05, upper_loc=0.9,
              kpc=10, asectokpc=None, vsys=None, color="black", y_offset=0.02,
              colorbar=True, colorbar_label=None,
              output=None):
    """Plot a moment-0, moment-1, or moment-2 map with contours and annotations.

    Shared implementation behind `plot_mom0`, `plot_mom1`, and
    `plot_mom2`.

    Parameters
    ----------
    data : str, ndarray, or tuple
        Input moment map, passed to `load_data`.
    which : {'mom0', 'mom1', 'mom2'}
        Which moment map is being plotted; controls the default
        colormap, normalization, contour recipe, and colorbar label.
    mult_by : float, default 1
        Multiplicative scaling applied to the data before plotting.
    add_to : float, default 0
        Additive offset applied to the data before plotting.
    use_column_density : bool, default False
        For ``which='mom0'`` only: convert the map (and its contours) to
        HI column density.
    xmin, xmax, ymin, ymax : int, optional
        Spatial subregion to plot, in pixel units. Defaults to the full
        range.
    wcs : astropy.wcs.WCS, optional
        World coordinate system; read from the data header if not given.
    bmaj, bmin, bpa : float, optional
        Beam major axis, minor axis, and position angle; read from the
        data header if not given.
    pixunits : str, default 'deg'
        Spatial pixel units ('deg' is converted to arcsec).
    fluxunits : str, optional
        Flux units, used for the colorbar label; read from the data
        header if not given.
    pixelres : float, optional
        Pixel resolution; read from the data header if not given.
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes, optional
        Existing figure/axis to plot into. Created via `make_figure` if
        not given.
    fig_x_size, fig_y_size : float, default 7
        Figure size, in inches, used if `fig` is not given.
    cmap : str, optional
        Colormap; a moment-specific default is used if not given.
    vmin, vmax : float, optional
        Normalization bounds (not used for ``which='mom1'``); derived
        from the data if not given.
    norm : str, optional
        Normalization type, passed to `_norm`; a moment-specific default
        is used if not given.
    ctr : str or array_like, optional
        Contour levels, or a recipe; a moment-specific default is used if
        not given.
    ctr_cmap : str, optional
        Colormap used to color the contour levels (``which != 'mom1'``);
        a moment-specific default is used if not given.
    ctr_width : float, default 1
        Contour line width.
    ctr_color_positive, ctr_color_negative : str, default 'black', 'gray'
        Colors for positive and negative contours (``which='mom1'`` only).
    plot_ctr : bool, default True
        Whether to overlay contours.
    left_loc, right_loc, lower_loc, upper_loc : float, default 0.05/0.95/0.05/0.9
        Axes-fraction anchor points for the beam and scale bar.
    kpc : float, default 10
        Length, in kpc, of the physical scale bar.
    asectokpc : float, optional
        Arcsec-to-kpc conversion factor; if given, a scale bar is drawn.
    vsys : float, optional
        Systemic velocity subtracted from the data (``which='mom1'``
        only); defaults to the data mean.
    color : str, default 'black'
        Color used for the beam and scale bar annotations.
    y_offset : float, default 0.02
        Vertical offset of the scale bar label.
    colorbar : bool, default True
        Whether to add a colorbar.
    colorbar_label : str, optional
        Colorbar label; a moment-specific default is used if not given.
    dpi : int, default 300
        Resolution used when saving the figure.
    output : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot, only if `fig` was not given.
    ax : matplotlib.axes.Axes
        The plot axis, only if `fig` or `ax` was not given.
    """

    # load the data
    data_dict = load_data(data)
    data = data_dict['data']

    # load the metadata
    if wcs is None:
        wcs = data_dict['wcs']
    if bmaj is None:
        bmaj = data_dict['bmaj']
    if bmin is None:
        bmin = data_dict['bmin']
    if bpa is None:
        bpa = data_dict['bpa']
    if pixelres is None:
        pixelres = data_dict['pixelres']
    if pixunits is None:
        pixunits = data_dict['pixunits']
    if fluxunits is None:
        fluxunits = data_dict['fluxunits']

    # scale the data
    data = data * mult_by + add_to

    # convert the spatial units
    pixres_arcsec = pixelres
    bmaj_arcsec = bmaj
    bmin_arcsec = bmin
    if pixunits == "deg":
        if pixres_arcsec is not None:
            pixres_arcsec *= 3600
        if bmaj_arcsec is not None:
            bmaj_arcsec *= 3600
        if bmin_arcsec is not None:
            bmin_arcsec *= 3600

    # moment-specific pre-processing
    if which == "mom0":
        # convert to column density
        if use_column_density and all(x is not None for x in (bmaj_arcsec, bmin_arcsec)):
            data[data < 0] = 0
            if 'km/s' not in fluxunits or 'km s-1' not in fluxunits:
                data /= 1000  # convert to km/s
    if which == "mom1":
        # subtract the systemic velocity
        if vsys is None:
            vsys = np.nanmean(data)
        data -= vsys

    # define the extraction region
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = data.shape[1]
    if ymin is None:
        ymin = 0
    if ymax is None:
        ymax = data.shape[0]

    # extract the subregion
    data = data[ymin:ymax, xmin:xmax]
    if wcs is not None:
        wcs = wcs[ymin:ymax, xmin:xmax]

    # plot defaults depending on the moment map
    if norm is None:
        norm = {"mom0": "log", "mom1": "", "mom2": "linear"}[which]
    if cmap is None:
        cmap = {"mom0": "Greys", "mom1": "mom1_cosmos", "mom2": "mom2"}[which]
    if ctr is None:
        ctr = {"mom0": "even", "mom1": None, "mom2": "even"}[which]
    if ctr_cmap is None:
        ctr_cmap = {"mom0": "hot", "mom1": "", "mom2": "mom2_ctr"}[which]
    if colorbar_label is None:
        colorbar_label = {
            "mom0": (
                f"Moment 0 [{fluxunits}]"
                if not use_column_density
                else r'HI column density [$10^{19}$ cm$^{-2}$]'
            ),
            "mom1": f"LoS velocity w.r.t systemic [{fluxunits}]",
            "mom2": f"Moment 2 [{fluxunits}]",
        }[which]

    # define the normalization
    if which == "mom1":
        norm_obj = cl.CenteredNorm()
    else:
        if vmin is None:
            vmin = 0.001 * np.nanmin(data[data != 0])
        if vmax is None:
            vmax = 0.9 * np.nanmax(data)
        norm_obj = _norm(norm, vmin, vmax)

    # define the contours
    if which == "mom1":
        if isinstance(ctr, (list, np.ndarray)):
            ctr = np.asarray(ctr)
            ctr_pos = ctr[ctr > 0]
            ctr_neg = ctr[ctr < 0]
        else:
            ctr_res = np.floor(0.85 * np.nanmax(np.abs(data)) / 15) * 5
            ctr_pos = np.arange(0, 0.85 * np.nanmax(np.abs(data)), ctr_res)[1:].astype(int)
            ctr_neg = -ctr_pos[::-1]
    else:
        if isinstance(ctr, (list, np.ndarray)):
            ctr = np.asarray(ctr)
            if which == "mom0" and use_column_density:
                ctr = convert_from_HI(ctr * 1e19, beamarea=1.13 * bmaj_arcsec * bmin_arcsec, from_moment_map=True)
        else:
            ctr = _contours(ctr, data, vmax=vmax, norm=norm_obj)

    # create the figure and axes
    return_fig = False
    if fig is None:
        fig, ax = make_figure(nrows=1, ncols=1,
                              fig_x_size=fig_x_size, fig_y_size=fig_y_size,
                              wcs=wcs, colorbar=colorbar)
        return_fig = True
    if ax is None:
        ax = fig.add_subplot(projection=wcs)
        return_fig = True

    # do the plot
    im = ax.imshow(data, cmap=cmap, norm=norm_obj, origin='lower' if ax.name == 'rectilinear' else None)

    # draw the contours
    if plot_ctr:
        if which == "mom1":
            ax.contour(data, levels=ctr_pos, colors=ctr_color_positive, linewidths=ctr_width, linestyles='solid')
            ax.contour(data, levels=ctr_neg, colors=ctr_color_negative, linewidths=ctr_width, linestyles='dashed')
        else:
            ax.contour(data, levels=ctr, cmap=ctr_cmap, linewidths=ctr_width, linestyles='solid')

    # draw the beam
    if all(x is not None for x in (pixres_arcsec, bmaj_arcsec, bmin_arcsec, bpa)):
        ax = _plot_beam(ax, pixres_arcsec,
                        bmaj_arcsec, bmin_arcsec, bpa,
                        xloc=right_loc, yloc=lower_loc, color=color)

    # draw the physical scale line
    if all(x is not None for x in (pixres_arcsec, asectokpc)):
        ax = _plot_kpc_scale(ax, pixres_arcsec, asectokpc, kpc=kpc,
                             xloc=left_loc, yloc=upper_loc, y_offset=y_offset,
                             color=color)

    # add the colorbar
    if colorbar:
        if which == "mom0":
            if use_column_density and all(x is not None for x in (bmaj_arcsec, bmin_arcsec)):
                # convert the contour levels to column density in units of 10^19 cm^-2
                ctr_label = np.around(
                    convert_to_HI(ctr, beamarea=1.13 * bmaj_arcsec * bmin_arcsec, from_moment_map=True)
                    / 10.0**19, 2)
            else:
                ctr_label = None
        else:
            ctr_label = None
        if which == "mom1":
            ax = _add_colorbar(fig, ax, im, norm_obj,
                               ctr=np.concatenate((ctr_neg, [0], ctr_pos)), ctr_pos_color=ctr_color_positive, ctr_neg_color=ctr_color_negative,
                               mode='posneg', label=colorbar_label)
        else:
            ax = _add_colorbar(fig, ax, im, norm_obj,
                               ctr, ctr_cmap=ctr_cmap, ctr_label=ctr_label,
                               label=colorbar_label)

    # save the figure
    if output is not None:
        fig.savefig(output, bbox_inches='tight', dpi=dpi)

    if return_fig:
        return fig, ax


def _plot_reliability(pos_params, neg_params, covariance, reliability, threshold, snr_min, output, *,
                      points_size=0.25, positive_color="royalblue", negative_color="hotpink", reliable_color="black",
                      ellipses_color="black", ellipses_width=1.5, xaxis_scale=0.05, yaxis_scale=0.05, dpi=300,
                      **kwargs):
    """Plot peak/sum/mean flux scatter diagrams for a reliability computation.

    Draws the three pairwise parameter-space panels (peak-vs-sum,
    peak-vs-mean, sum-vs-mean) used by `_reliability`, overlaying the
    negative-source covariance error ellipses, an S/N cut line, and the
    sources flagged as reliable.

    Parameters
    ----------
    pos_params, neg_params : ndarray, shape (3, N)
        log10(peak), log10(sum), log10(mean) flux for the positive and
        negative sources, as returned by `_extract_reliability_params`.
    covariance : ndarray, shape (3, 3)
        Covariance matrix of the negative-source parameters.
    reliability : ndarray, shape (N_pos,)
        Reliability value of each positive source.
    threshold : float
        Reliability threshold used to flag reliable sources.
    snr_min : float
        Minimum S/N used to draw the S/N cut line.
    output : str
        Base path the figure is saved to.
    points_size : float, default 0.25
        Scatter marker size, as a fraction of the default marker area.
    positive_color, negative_color : str, default 'royalblue', 'hotpink'
        Marker colors for positive and negative sources.
    reliable_color : str, default 'black'
        Marker color for sources flagged as reliable.
    ellipses_color : str, default 'black'
        Color of the covariance error ellipses.
    ellipses_width : float, default 1.5
        Line width of the covariance error ellipses.
    xaxis_scale, yaxis_scale : float, default 0.05
        Fractional padding added to the automatic axis limits.
    dpi : int, default 300
        Resolution used when saving the figure.
    **kwargs
        Additional keyword arguments forwarded to `make_figure`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The last plotted panel axis.
    """

    # define the parameter spaces (0=peak, 1=sum, 2=mean)
    param_space = [  # xaxis param, yaxis param, plot title , xlabel, ylabel
        (0, 1, "Peak vs sum", "Log(peak/rms)", "Log(sum/rms)"),
        (0, 2, "Peak vs mean", "Log(peak/rms)", "Log(mean/rms)"),
        (1, 2, "Sum vs mean", "Log(sum/rms)", "Log(mean/rms)"),
    ]

    # define the scatter point area
    points_area = (points_size * plt.rcParams["lines.markersize"]) ** 2

    # create the figure and the axes
    fig, axes = make_figure(ncols=2, nrows=2, hspace=0.1, wspace=0.1, **kwargs)

    # hide unused axes
    for i in range(3, len(axes)):
        axes[i].set_visible(False)

    for ax, (i, j, title, xlabel, ylabel) in zip(axes, param_space):
        # compute the covariance ellipse
        rmaj, rmin, pa = _covariance_to_error_ellipse(covariance, i, j)

        # determine the distribution center
        center = [np.nanmean(neg_params[i]), np.nanmean(neg_params[j])]

        # add the error ellipses
        for k, ls in zip([1, 2, 3], ["solid", "dashed", "-."]):
            ellipse = patch.Ellipse(center, width=k * rmaj, height=k * rmin, angle=pa,
                                    fill=False, ec=ellipses_color, ls=ls, lw=ellipses_width, zorder=100)
            ax.add_patch(ellipse)

        # scatter plot of the negative sources
        ax.scatter(neg_params[i], neg_params[j], s=points_area, c=negative_color, label="Negative")
        ax.scatter(pos_params[i], pos_params[j], s=points_area, c=positive_color, label="Positive")

        # set the title and the labels
        ax.set_title(title, pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # scale the axis for better visualisation
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = xaxis_scale * (xmax - xmin)
        dy = yaxis_scale * (ymax - ymin)
        ax.set_xlim(xmin - dx, xmax + dx)
        ax.set_ylim(ymin - dy, ymax + dy)

        # use a square aspect
        ax.set_aspect((xmax - xmin) / (ymax - ymin))

        # draw the S/N line in the sum vs mean panel
        if (i, j) == (1, 2):
            x = np.array([pos_params[i].min(), pos_params[i].max()])
            y = 2 * np.log10(snr_min) - x
            ax.plot(x, y, c="gray", ls="--", lw=1, zorder=0)
            ax.text(0.95, 0.05, f"SNR min = {snr_min}", ha="right", transform=ax.transAxes)

    # highlight reliable sources
    reliable = reliability >= threshold

    axes[0].scatter(
        pos_params[0][reliable],
        pos_params[1][reliable],
        s=8 * points_area,
        c=reliable_color,
        label="Reliable",
    )
    axes[1].scatter(
        pos_params[0][reliable],
        pos_params[2][reliable],
        s=8 * points_area,
        c=reliable_color,
        label="Reliable",
    )
    axes[2].scatter(
        pos_params[1][reliable],
        pos_params[2][reliable],
        s=8 * points_area,
        c=reliable_color,
        label="Reliable",
    )

    # add the legend
    for ax in axes:
        ax.legend(loc="best", ncol=2)

    # save the figure
    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
    return fig, ax


def _plot_skellam(P, N, kernel, output, *, dpi=300, **kwargs):
    """Plot a Skellam-distribution diagnostic for the reliability computation.

    Compares the cumulative distribution of the normalized
    ``(P - N) / sqrt(P + N)`` Skellam parameter to the expected standard
    Gaussian cumulative distribution.

    Parameters
    ----------
    P, N : ndarray
        Positive- and negative-source KDE density values.
    kernel : float
        Smoothing kernel scale used in the reliability computation
        (shown in the plot legend).
    output : str
        Path the figure is saved to.
    dpi : int, default 300
        Resolution used when saving the figure.
    **kwargs
        Additional keyword arguments forwarded to `make_figure`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The plot axis.
    """

    # compute the skellam paramenter (P-N)/sqrt(P+N)
    skellam = (P - N) / np.sqrt(P + N)

    # sort the array for the plot.
    # This step is necessary as we are going to normalize it
    skellam = np.sort(skellam)

    # normalize to sigma = 1.
    # This is necessary to ensure that the standard deviation of the Skellam parameter values is 1,
    # such that its distribution can be readily compared to a standard Gaussian
    skellam = (skellam - np.nanmedian(skellam)) / np.nanstd(skellam)
    skellam += np.nanmedian(skellam)

    # compute the cumulative distribution
    cum = np.arange(len(skellam)) / (len(skellam) - 1)

    # the Gaussian cumulative is given by the erf function
    x = np.linspace(-4, 4, 401)
    y = 0.5 - 0.5 * erf(-x / np.sqrt(2.0))

    # create the figure
    fig, ax = make_figure(ncols=1, nrows=1, **kwargs)

    # plot the skellam distribution
    ax.plot(skellam, cum, c="b", lw=1, label=fr"Data: $\mu={np.nanmedian(skellam):.2f}$" "\n" fr"kernel = {kernel}")

    # plot the expected gaussian distribution
    ax.plot(x, y, c="r", lw=1, label=r"Gaussian: $\mu=0$, $\sigma=1$")
    ax.axvline(np.nanmedian(skellam), c="b", ls="dashed", lw=1)
    ax.axvline(0, c="r", ls="dotted", lw=1)

    # set limits and labels
    ax.set_xlim([-4, 4])
    ax.set_ylim([0, 1])
    ax.set_xlabel(r"Skellam parameter normalized to $\sigma=1$")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="upper left")

    # save the figure
    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")

    return fig, ax


def _plot_spectrum(spectrum, v=None, *,
                   xunits=r'km s$^{-1}$', yunits=r'Jy beam$^{-1}$',
                   color='blue', linewidth=1, mask=None, masked_color='blue', masked_linewidth=1,
                   figx=6, figy=3,
                   **kwargs):
    """Plot a spectrum, optionally highlighting masked (detected) channels.

    Parameters
    ----------
    spectrum : ndarray
        Input spectrum.
    v : array_like, optional
        Velocity (or other spectral) axis; channel numbers are used if
        not given.
    xunits : str, default r'km s$^{-1}$'
        Spectral axis units label, used only if `v` is given.
    yunits : str, default r'Jy beam$^{-1}$'
        Flux axis units label.
    color : str, default 'blue'
        Line color for the (unmasked, or full) spectrum.
    linewidth : float, default 1
        Line width for the (unmasked, or full) spectrum.
    mask : ndarray, optional
        Detection mask; if given, masked (noise) channels are drawn
        separately from unmasked (source) channels.
    masked_color : str, default 'blue'
        Line color for the masked (noise) channels, used only if `mask`
        is given.
    masked_linewidth : float, default 1
        Line width for the masked (noise) channels, used only if `mask`
        is given.
    figx, figy : float, default 6, 3
        Default figure size, in inches (overridden by `fig_x_size`/
        `fig_y_size` in `**kwargs`, if given).
    **kwargs
        Additional keyword arguments forwarded to `make_figure`
        (``ylabel`` defaults to the flux axis label built from `yunits`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The plot axis.
    """

    figx = kwargs.get('figx', 8)
    figx = kwargs.get('fig_x_size', 8)
    figy = kwargs.get('figy', 4)
    figy = kwargs.get('fig_y_size', 4)
    ylabel = kwargs.get('ylabel', f'Flux [{yunits}]')

    # create the figure
    fig, ax = make_figure(ncols=1, nrows=1, figx=figx, figy=figy, ylabel=ylabel, **kwargs)

    # define the x-axes
    if v is None:
        v = np.arange(1, len(spectrum) + 1)
        ax.set_xlabel('Channels')
    else:
        ax.set_xlabel(f'Velocity [{xunits}]')

    # if a mask is not given
    if mask is None:
        ax.plot(v, spectrum, color=color, lw=linewidth)
    # if a mask is given
    else:
        # blank the noisy channels
        mask = mask.astype(np.float32)  # convert to float to assign nan values
        mask[mask == 0] = np.nan

        # ensure mask is binary
        mask[mask > 0] = 1

        # plot the spectrum
        ax.plot(v, spectrum, color=masked_color, lw=masked_linewidth, ls='dotted', label='Noise')

        # highlight the unmasked channels
        ax.plot(v, spectrum * mask, color=color, lw=linewidth, label='Source')

        # add the legend
        ax.legend(frameon=False)

    # save the axes limits to prevent the ancillary curves to affet them
    xlim = (np.nanmin(v), np.nanmax(v))
    ylim = ax.get_ylim()

    # draw the 0-flux line and 0-velocity
    ax.axhline(y=0, c='black', ls='dashed', lw=1)
    ax.axvline(x=0, c='black', ls='dotted', lw=1)

    # restore the axes limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return fig, ax


def _plot_vertical_structure(result):
    """Plot diagnostics for a `vertical_structure` solution.

    Produces three figures: the input and extended surface densities per
    component (log scale), a grid of summary profiles (surface density,
    velocity dispersion, scale height, vertical force, volume and number
    density), and mirrored 2D (R, z) volume density maps per component.

    Parameters
    ----------
    result : dict
        Output dictionary from `vertical_structure`.

    Returns
    -------
    fig1 : matplotlib.figure.Figure
        Extended surface density diagnostic figure.
    fig2 : matplotlib.figure.Figure
        Grid of summary profile figures.
    fig3 : matplotlib.figure.Figure
        2D volume density map figure, one panel per component (up to
        three).
    """
    # get the components
    components = result['components']

    # colours
    colors = ['blue', 'crimson', 'limegreen']

    # radial grid
    radii = result['radii']
    radii_grid = result['radii_grid']

    # vertical grid
    z_grid = result['z_grid']

    # convert vertical log-grid into vertical linear grid
    z_arr = np.linspace(z_grid[0], z_grid[-1], len(z_grid))

    # mirror to negative R and z
    R_full = np.concatenate([-radii[::-1], radii[1:]])
    z_full = np.concatenate([-z_arr[::-1], z_arr[1:]])

    # create the figures
    # extended surface densities diagnostic
    fig1, ax1 = make_figure(figx=8, figy=5,
                            xlabel='Radius [kpc]', ylabel=r'$\Sigma$ [M$_\odot$ kpc$^{-2}$]',
                            yscale='log')

    # summary plots
    fig2, ax2 = make_figure(ncols=2, nrows=3,
                            figx=6, figy=4,
                            xlabel='Radius [kpc]',
                            ylabel=[
                                r'$\Sigma$ [M$_\odot$ kpc$^{-2}$]',
                                r'$\sigma$ [km s$^{-1}$]',
                                r'$z_0$ [kpc]',
                                r'$\langle|K_z|\rangle$ [km$^{2}$ kpc s$^{-2}$]',
                                r'$\rho_0$ [M$_\odot$ kpc$^{-3}$]',
                                r'$n_0$ [cm$^{-3}$]',
                            ],
                            yscale=['log', 'linear', 'linear', 'linear', 'log', 'log'])

    # 2D density maps
    fig3, ax3 = make_figure(ncols=3, figx=5, figy=5, xlabel='R [kpc]', ylabel='z [kpc]')

    # for each components
    for component in components:
        # component index
        idx = components.index(component)

        # components color
        color = colors[idx]

        # component cmap
        cmap = cl.LinearSegmentedColormap.from_list(component, ['black', color])

        # surface density
        sdens = result[f'sdens_{component}']

        # extended surface density
        sdens_grid = result[f'sdens_grid_{component}']

        # velocity dispersion
        sigma = result[f'sigma_{component}']

        # analytical solutions
        z0_anal = result[f'z0_anal_{component}']

        # dark-matter only solution
        z0_dm = result[f'z0_dm_{component}']

        # full solution
        z0 = result[f'z0_{component}']
        z0_err = result.get(f'z0_{component}_mc_std', np.zeros_like(z0))

        # average vertical force
        Kz = np.abs(np.nanmean(result[f'Kz_2d_{component}'], axis=0))
        Kz_err = np.abs(np.nanmean(result.get(f'Kz_2d_{component}_mc_std', np.zeros_like(Kz)), axis=0))

        # midplane volume density
        rho0 = result[f'rho0_{component}']
        rho0_err = result.get(f'rho0_{component}_mc_std', np.zeros_like(rho0))

        # midplane number density
        n0 = result[f'n0_{component}']
        n0_err = result.get(f'n0_{component}_mc_std', np.zeros_like(n0))

        # 3D density
        rho = result[f'rho_{component}'] / 10.0**7

        # resample the density onto the linear z grid
        for R in range(len(radii)):
            rho[:, R] = interp1d(z_grid, rho[:, R], kind='linear', bounds_error=False, fill_value=0.0)(z_arr)

        # mirror onto the negative spatial coordinates
        rho = np.concatenate([rho[::-1], rho[1:]], axis=0)
        rho = np.concatenate([rho[:, :0:-1], rho], axis=1)

        # remove non-finite values
        rho[~np.isfinite(rho)] = 0.0

        # define the normalisation for the 2D density plot
        norm = cl.PowerNorm(gamma=0.2, vmin=np.nanmin(rho), vmax=np.nanmax(rho))

        # the colorbar labels
        ctr = norm.inverse(np.linspace(0, 1, 7))[1:-1]

        # the extended radii
        mask = radii_grid > np.nanmax(radii[np.isfinite(sdens)])

        # do the plots
        ax1.plot(radii, sdens, c=color, label=component.capitalize())
        ax1.plot(radii_grid[mask], sdens_grid[mask], c=color, ls='dashed')

        ax2[0].plot(radii, sdens, c=color, label=component.capitalize())  # input surface density
        ax2[1].plot(radii, sigma, c=color)  # input velocity dispersion
        ax2[2].plot(radii, z0, c=color)  # output scale height
        ax2[2].fill_between(radii, z0 - z0_err, z0 + z0_err, color=color, alpha=0.25, zorder=0)
        ax2[3].plot(radii, Kz, c=color)  # average vertical force
        ax2[4].plot(radii, rho0, c=color)  # midplane volume density
        ax2[5].plot(radii, n0, c=color)  # midplane number density

        im = ax3[idx].imshow(rho,
                             norm=norm, cmap='Greys',
                             aspect='auto', origin='lower',
                             extent=(R_full[0], R_full[-1], z_full[0], z_full[-1]))
        mins, maxs = np.nanmax((R_full[0], z_full[0])), np.nanmin((R_full[-1], z_full[-1]))
        ax3[idx].set_xlim(mins, maxs)
        ax3[idx].set_ylim(mins, maxs)

        # add the colorbar
        _add_colorbar(fig3, ax3[idx], im, norm, ctr, label=fr'$\rho_\mathrm{{{component}}}$ [10$^7$ M$_\odot$ kpc$^{{-3}}$]')

    # manage the y-axis limits
    ax1.set_ylim(1.01, None)

    # add the legends
    ax1.legend(prop={'size': 12}, frameon=False)
    ax2[0].legend(prop={'size': 12}, frameon=False)

    return fig1, fig2, fig3
from astropy.io import fits
from astropy.wcs import WCS
from pvextractor import Path
from pvextractor import extract_pv_slice
from tqdm.auto import tqdm

import numpy as np


_INPUT_TYPE_ERROR_MSG = (
    "Input must be a path to a FITS file (string), a NumPy array, or a "
    "(array, header) tuple."
)


def cutout(input_fits, *,
           xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
           output=None):
    """Extract a spatial/spectral cutout from a FITS file or array.

    Parameters
    ----------
    input_fits : str, ndarray, or tuple
        Input data, passed to `load_data`.
    xmin, xmax, ymin, ymax : int, optional
        Spatial cutout range, in pixel units. Defaults to the full range.
    zmin, zmax : int, optional
        Spectral (channel) cutout range, in pixel units, used only if the
        input data is 3D. Defaults to the full range.
    output : str, optional
        If given, the cutout is written to this path as a FITS file and
        nothing is returned.

    Returns
    -------
    data : ndarray
        The cutout data, only if `output` is None.
    header : astropy.io.fits.Header
        The updated header with a WCS matching the cutout, only if
        `output` is None.
    """

    # load the data
    data_dict = load_data(input_fits)

    # store the relevant info
    data, header = data_dict['data'], data_dict['header']

    # use full ranges if not specified
    if xmin is None or xmin < 0:
        xmin = 0
    if ymin is None or ymin < 0:
        ymin = 0
    if zmin is None or zmin < 0:
        zmin = 0

    if data.ndim == 3:
        if zmax is None or zmax > data.shape[0]:
            zmax = data.shape[0]
        if xmax is None or xmax > data.shape[2]:
            xmax = data.shape[2]
        if ymax is None or ymax > data.shape[1]:
            ymax = data.shape[1]
    elif data.ndim == 2:
        if xmax is None or xmax > data.shape[1]:
            xmax = data.shape[1]
        if ymax is None or ymax > data.shape[0]:
            ymax = data.shape[0]

    # extract the subdata
    if data.ndim == 3:
        data = data[zmin:zmax, ymin:ymax, xmin:xmax]
    elif data.ndim == 2:
        data = data[ymin:ymax, xmin:xmax]

    # extract the new wcs and update the header
    if header != {} and header is not None:
        if data.ndim == 3:
            wcs = WCS(header)[zmin:zmax, ymin:ymax, xmin:xmax]  # store the wcs
        elif data.ndim == 2:
            wcs = WCS(header)[ymin:ymax, xmin:xmax]  # store the wcs
        newheader = wcs.to_header()  # write the wcs into a header
        for key, value in newheader.items():
            header[key] = value

    # write the output
    if output is not None:
        fits.PrimaryHDU(data, header=header).writeto(output, overwrite=True)
    # return the data and header
    else:
        return data, header
        
        
def load_data(input_data, *, ext=None):
    """Load data and metadata from a FITS file, array, or (array, header) tuple.

    Normalizes the various accepted input forms into a single dictionary
    of data and derived metadata (WCS, pixel/spectral resolution, beam,
    units, and reference coordinates).

    Parameters
    ----------
    input_data : str, ndarray, or tuple of (ndarray, astropy.io.fits.Header)
        A path to a FITS file, a NumPy array, or an (array, header) tuple.
    ext : int or str, optional
        FITS extension to read, used only if `input_data` is a file path.
        If not given, the first extension containing array data is used.

    Returns
    -------
    dict
        Dictionary with keys ``'data'``, ``'header'``, ``'wcs'``,
        ``'pixelres'``, ``'spectralres'``, ``'x0'``, ``'y0'``, ``'z0'``,
        ``'fluxunits'``, ``'pixunits'``, ``'specunits'``, ``'bmaj'``,
        ``'bmin'``, and ``'bpa'``. Entries that cannot be derived (e.g. no
        header is available) are set to None.
    """

    # check if the input is a string (assume it's a path to a FITS file)
    if isinstance(input_data, str):
        # open the FITS file
        with fits.open(input_data) as hdul:
            # if the user provided an extension, use it
            if ext is not None:
                hdu = hdul[ext]
            else:
                # auto-detect first extension with data
                for h in hdul:
                    if isinstance(h.data, np.ndarray):
                        hdu = h
                        break
                else:
                    raise ValueError("No HDU with data found in FITS file.")
            # the data is usually in the first HDU (Header/Data Unit)
            data = hdu.data
            # the header contains metadata about the observation
            header = hdu.header
        del hdul

    # check if the input is a NumPy array
    elif isinstance(input_data, np.ndarray):
        # use the array as the data
        data = input_data.copy()
        # no header is available
        header = {}

    # check if the input is a tuple
    elif isinstance(input_data, tuple):
        if isinstance(input_data[0], np.ndarray):
            data = input_data[0].copy()
        else:
            raise TypeError(_INPUT_TYPE_ERROR_MSG)
        if isinstance(input_data[1], fits.header.Header):
            header = input_data[1].copy()
        else:
            raise TypeError(_INPUT_TYPE_ERROR_MSG)

    # if the input is neither a string, an array, nor a tuple, raise an error
    else:
        raise TypeError(_INPUT_TYPE_ERROR_MSG)

    # create the WCS object from the header
    wcs = WCS(header) if header != {} and header is not None else None

    # compute the pixel scale and spectral resolution
    if wcs is not None:
        scales = np.diag(wcs.pixel_scale_matrix)  # extract the pixel scales
        pixelres = np.abs(scales[0])  # pixel resolution
        if wcs.naxis == 3:
            spectralres = scales[2]  # spectral resolution
        else:
            spectralres = None
    else:
        pixelres, spectralres = None, None

    # create the data dictionary
    data_dict = {
        # data and metadata
        'data': data,
        'header': header,
        'wcs': wcs,
        # spatial and spectral resolution
        'pixelres': pixelres,
        'spectralres': spectralres,
        # image center and spectral reference
        'x0': (
            header.get('CRVAL1') + (header.get('CRPIX1') - 1) * scales[0]
            if header != {} and header is not None
            else None
        ),
        'y0': (
            header.get('CRVAL2') + (header.get('CRPIX2') - 1) * scales[1]
            if header != {} and header is not None
            else None
        ),
        'z0': (
            header.get('CRVAL3') + (header.get('CRPIX3') - 1) * spectralres
            if all(x is not None for x in (header, spectralres))
            else None
        ),
        # units
        'fluxunits': header.get('BUNIT') if header != {} and header is not None else '',
        'pixunits': header.get('CUNIT1') if header != {} and header is not None else '',
        'specunits': header.get('CUNIT3') if header != {} and header is not None else '',
        # beam parameters
        'bmaj': header.get('BMAJ') if header != {} and header is not None else None,
        'bmin': header.get('BMIN') if header != {} and header is not None else None,
        'bpa': header.get('BPA') if header != {} and header is not None else None,
    }

    return data_dict
    
    
def pvslice(datacube, *,
            width=3, start_point=None, end_point=None, length=None, angle=None, center=None,
            first_channel=None, last_channel=None, xmin=None, xmax=None, ymin=None, ymax=None,
            output=None, plot=False,
            **kwargs):
    """Extract (and optionally plot) a position-velocity slice from a cube.

    Parameters
    ----------
    datacube : str, ndarray, or tuple
        Input data cube, passed to `load_data`.
    width : float, default 3
        Width of the extraction path, in pixels.
    start_point, end_point : tuple of (float, float), optional
        Start and end points of the slice path, in pixel coordinates. If
        not given, they are computed from `length`, `angle`, and `center`.
    length : float, optional
        Length of the slice path, in pixels, used if `start_point`/
        `end_point` are not given.
    angle : float, optional
        Position angle of the slice path, in degrees, used if
        `start_point`/`end_point` are not given.
    center : tuple of (float, float), optional
        Centre of the slice path, in pixel coordinates. Defaults to the
        cube's spatial centre.
    first_channel, last_channel : int, optional
        Channel range to extract. Defaults to the full cube.
    xmin, xmax, ymin, ymax : int, optional
        Spatial subregion to extract from, in pixel units. Defaults to
        the full range.
    output : str, optional
        If given, the PV slice is written to this path as a FITS file.
    plot : bool, default False
        If True, also plot the PV slice via `plot_pvslice`.
    **kwargs
        Additional keyword arguments forwarded to `plot_pvslice` if
        `plot` is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure, only if `plot` is True.
    ax : matplotlib.axes.Axes
        The plot axis, only if `plot` is True.
    pv : astropy.io.fits.PrimaryHDU
        The extracted PV slice, always returned unless `output` is given
        and `plot` is False, in which case it is also returned.
    """

    # load the data
    data_dict = load_data(datacube)

    # store the relevant info
    data, header, wcs = data_dict['data'], data_dict['header'], data_dict['wcs']

    # define the channel range
    if first_channel is None or first_channel < 0:
        first_channel = 0
    if last_channel is None or last_channel > data.shape[0]:
        last_channel = data.shape[0]

    # use full ranges if not specified
    if xmin is None or xmin < 0:
        xmin = 0
    if ymin is None or ymin < 0:
        ymin = 0
    if xmax is None or xmax > data.shape[2]:
        xmax = data.shape[2]
    if ymax is None or ymax > data.shape[1]:
        ymax = data.shape[1]

    # extract the subcube
    data = data[first_channel : last_channel + 1, ymin:ymax, xmin:xmax]
    wcs = wcs[first_channel : last_channel + 1, ymin:ymax, xmin:xmax]

    # define the pv path
    if start_point is None or end_point is None:
        if length is None or angle is None:
            raise ValueError(
                "If start_point and end_point are not provided, length and angle must be provided."
            )

        # calculate the start and end points from length, angle, and center
        if center is None:
            x0, y0 = data.shape[2] // 2, data.shape[1] // 2
        else:
            x0, y0 = center
        theta = np.radians(angle + 90)  # +90 accounts for the pvextractor angle definition

        # calculate the start and end points
        x1 = x0 - (length / 2) * np.cos(theta)
        y1 = y0 - (length / 2) * np.sin(theta)
        x2 = x0 + (length / 2) * np.cos(theta)
        y2 = y0 + (length / 2) * np.sin(theta)

        start_point, end_point = (x1, y1), (x2, y2)

    # extract the pv slice
    pv_path = Path([start_point, end_point], width=width)
    pv = extract_pv_slice(data, pv_path, wcs)

    # fix the header in order to have the distance from the path center as spatial dimension
    pv.header['CRPIX1'] = round(pv.header['NAXIS1'] / 2) + 1

    # add the flux units
    pv.header['BUNIT'] = header.get('BUNIT')

    # write the pv slice to a fits file
    if output is not None:
        pv.writeto(output, overwrite=True)

    # plot the pv slice if requested
    if plot:
        from nicci.plotter.plotter import plot_pvslice

        fig, ax = plot_pvslice(pv.data, pv.header, **kwargs)
        return fig, ax, pv  # return the figure, axis, pv data and header
    else:
        return pv  # return the pv data and header
        
        
def shuffle(input_fits, velocity_field, *,
            output_fits=None, spectralres=None, v0=None, fill_value=np.nan, show_progress_bar=True):
    """Shift each spectrum of a cube so that a velocity field aligns to one channel.

    Rolls (circularly shifts) the spectrum at each spatial pixel so that
    the channel closest to the corresponding velocity-field value is
    moved to the central channel of the cube, following GIPSY's SHUFFLE
    task.

    Parameters
    ----------
    input_fits : str, ndarray, or tuple
        Input data cube, passed to `load_data`.
    velocity_field : str, ndarray, or tuple
        2D velocity field, passed to `load_data`; NaN pixels are skipped.
    output_fits : str, optional
        If given, the shuffled cube is written to this path as a FITS
        file and nothing is returned.
    spectralres : float, optional
        Spectral resolution (channel width). Derived from the cube
        metadata/header if not given.
    v0 : float, optional
        Velocity of the first channel. Derived from the header
        (``CRVAL3``, ``CRPIX3``) if not given.
    fill_value : float, default np.nan
        Value used to replace NaNs in the velocity field before shuffling.
    show_progress_bar : bool, default True
        Whether to display a `tqdm` progress bar.

    Returns
    -------
    shuffled : ndarray
        The shuffled cube, only if `output_fits` is None.
    header : astropy.io.fits.Header
        The (updated) header, only if `output_fits` is None.
    """

    # load the input data and velocity field
    data_dict = load_data(input_fits)
    vfield_dict = load_data(velocity_field)

    # store the relevant info
    data = data_dict['data']
    header = data_dict['header']
    vfield = vfield_dict['data']

    # determine spectral resolution
    if spectralres is None:
        spectralres = data_dict.get('spectralres') or header.get('CDELT3')
        if spectralres is None:
            raise ValueError("No spectral resolution found.")

    # determine starting velocity v0
    if v0 is None:
        if 'CRVAL3' in header and 'CRPIX3' in header:
            v0 = header['CRVAL3'] + (header['CRPIX3'] - 1) * spectralres
        else:
            raise ValueError("No starting velocity found.")

    # create the velocity array
    n_channel = data.shape[0]
    v = v0 + np.arange(n_channel) * spectralres
    mid_channel = n_channel // 2

    # initialize the shuffled data array
    shuffled = np.full_like(data, np.nan)

    # handle fill value for NaNs in velocity field
    if fill_value != np.nan:
        vfield[np.isnan(vfield)] = fill_value

    # find valid pixels in the velocity field
    valid_pixels = np.argwhere(~np.isnan(vfield))

    # loop over each pixel and shuffle the spectrum
    for y, x in tqdm(valid_pixels, desc='Spectra shuffled', disable=not show_progress_bar):
        spectrum = data[:, y, x]
        if np.all(np.isnan(spectrum)):
            continue
        # find the target channel, i.e. the channel whose velocity is closest to
        # the velocity-field value
        target_v = vfield[y, x]
        target_chan = np.argmin(np.abs(v - target_v))

        # compute the shift needed to align the target channel to the mid channel
        shift = mid_channel - target_chan

        # apply the shift using np.roll to preserve the noise properties
        shuffled[:, y, x] = np.roll(spectrum, shift)

    # update the header for the shuffled cube if a new FITS file is to be written
    if output_fits:
        header['CRPIX3'] = mid_channel + 1
        header['CRVAL3'] = 0.0
        fits.PrimaryHDU(shuffled, header=header).writeto(output_fits, overwrite=True)
    else:
        return shuffled, header
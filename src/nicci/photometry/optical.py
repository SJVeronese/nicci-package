from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import NDData
from astropy.table import Table
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars, EPSFBuilder, fit_2dgaussian

from nicci.io.fits import load_data


def download_gaia(radius=60, *, centre=None, ra=None, dec=None, output=None):
    """Query the Gaia archive for sources within a cone around a target.

    Parameters
    ----------
    radius : float, default 60
        Search cone radius, in arcsec.
    centre : astropy.coordinates.SkyCoord, optional
        Centre of the search cone. Built from `ra`/`dec` if not given.
    ra, dec : float, optional
        Right ascension and declination of the search cone centre, in
        degrees; used only if `centre` is not given.
    output : str, optional
        If given, the resulting catalogue is written to this path
        (format inferred from the extension).

    Returns
    -------
    astropy.table.Table
        The Gaia source catalogue found within `radius` of the cone
        centre.
    """

    # return every matching row instead of the default 50-row cap
    Gaia.ROW_LIMIT = -1

    # determine the centre of the search cone
    if centre is None:
        centre = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # run the cone search
    catalogue = Gaia.cone_search_async(centre, radius=u.Quantity(radius, u.arcsec)).get_results()

    # save the catalogue
    if output is not None:
        catalogue.write(output, overwrite=True)

    return catalogue


def epsf(data, *,
         threshold=10.0, fwhm=1.5, cutout_size=25.0, maxiters=100,
         progress_bar=True, **kwargs):
    """Build an effective PSF (ePSF) model from stars detected in an image.

    Parameters
    ----------
    data : str, ndarray, or tuple
        Input image, passed to `nicci.io.fits.load_data`.
    threshold : float, default 10.0
        Detection threshold, in units of the background rms, passed to
        `photutils.detection.DAOStarFinder`.
    fwhm : float, default 1.5
        Approximate FWHM of the sources to detect, in pixels, passed to
        `DAOStarFinder`.
    cutout_size : float, default 25.0
        Size, in pixels, of the square cutout extracted around each star.
    maxiters : int, default 100
        Maximum number of `EPSFBuilder` iterations.
    progress_bar : bool, default True
        Whether to display a progress bar while building the ePSF.
    **kwargs
        Additional keyword arguments forwarded to `DAOStarFinder` and
        `EPSFBuilder`. These two callees do not share a keyword
        interface, so a kwarg valid for one may raise a `TypeError` in
        the other.

    Returns
    -------
    dict
        Dictionary with keys ``'epsf'`` (the built ePSF, as an ndarray),
        ``'fwhm'`` (its fitted FWHM), ``'fit_output'`` (the raw
        `EPSFBuilder` output), ``'stars'`` (the detected source table),
        and ``'fitted_stars'`` (the star cutouts used in the fit).
    """

    # load the data
    data = load_data(data)['data']

    # define and run the stellar finder
    finder = DAOStarFinder(threshold=threshold, fwhm=fwhm, **kwargs)
    sources = finder(data)

    # get the star coordinates
    x = sources['x_centroid']
    y = sources['y_centroid']

    # keep only stars whose extraction cutout fits fully within the image
    hsize = (cutout_size - 1) / 2
    mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] - 1 - hsize)))

    # build the stellar catalogue
    stars_tbl = Table()
    stars_tbl['x'] = x[mask]
    stars_tbl['y'] = y[mask]

    # extract_stars requires an NDData input
    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=cutout_size)

    # build the ePSF
    epsf_builder = EPSFBuilder(oversampling=1, maxiters=maxiters, progress_bar=progress_bar, **kwargs)
    output = epsf_builder(stars)

    return {
        'epsf': output.epsf.data,
        'fwhm': fit_2dgaussian(output.epsf.data).psf_model.fwhm.value,
        'fit_output': output,
        'stars': sources,
        'fitted_stars': output.fitted_stars,
    }



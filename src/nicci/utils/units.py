import numpy as np

HI_CONSTANT_COLDEN = 1.25 * 10.0**24
HI_CONSTANT_MASS = 2.35 * 10.0**5

def convert_to_HI(data, beamarea, *, spectralres=None, from_moment_map=False):
    """Convert a flux density into an HI column density.

    Parameters
    ----------
    data : array_like
        Flux density values, in the same units implied by `beamarea`
        (typically Jy/beam, or Jy/beam * km/s for a moment-0 map).
    beamarea : float
        Beam area, in arcsec^2.
    spectralres : float, optional
        Spectral resolution (channel width), in km/s. Required unless
        `from_moment_map` is True.
    from_moment_map : bool, default False
        If True, `data` is already velocity-integrated (e.g. a moment-0
        map) and `spectralres` is not needed.

    Returns
    -------
    ndarray
        HI column density, in cm^-2.
    """

    if beamarea is None or beamarea <= 0:
        raise ValueError("Beam area must be a positive number.")

    if not from_moment_map:
        if spectralres is None:
            raise ValueError("spectralres must be provided when from_moment_map=False.")
        dv = np.abs(spectralres)
    else:
        dv = 1.0  # moment-0 maps are already integrated over velocity

    return np.asarray(data) * dv * HI_CONSTANT_COLDEN / beamarea


def convert_from_HI(data, beamarea, *, spectralres=None, from_moment_map=False):
    """Convert an HI column density back into a flux density.

    Inverse of `convert_to_HI`.

    Parameters
    ----------
    data : array_like
        HI column density values, in cm^-2.
    beamarea : float
        Beam area, in arcsec^2.
    spectralres : float, optional
        Spectral resolution (channel width), in km/s. Required unless
        `from_moment_map` is True.
    from_moment_map : bool, default False
        If True, the output is velocity-integrated (e.g. a moment-0 map)
        and `spectralres` is not needed.

    Returns
    -------
    ndarray
        Flux density, in Jy/beam (or Jy/beam * km/s if `from_moment_map`
        is True).
    """

    if beamarea is None or beamarea <= 0:
        raise ValueError("Beam area must be a positive number.")

    if not from_moment_map:
        if spectralres is None:
            raise ValueError("spectralres must be provided when from_moment_map=False.")
        dv = np.abs(spectralres)
    else:
        dv = 1.0  # moment-0 maps are already integrated over velocity

    return np.asarray(data) * beamarea / (dv * HI_CONSTANT_COLDEN)
    
    
def hi_mass(data,*,
            beamarea=None, distance=None, pixelres=None,
            xmin=None, xmax=None, ymin=None, ymax=None):
    """Compute the total HI mass from a flux map or flux value.

    Parameters
    ----------
    data : array_like
        Flux density map (or already-summed flux), in Jy/beam * km/s.
    beamarea : float
        Beam area, in arcsec^2.
    distance : float
        Source distance, in Mpc.
    pixelres : float
        Pixel resolution, in arcsec.
    xmin, xmax, ymin, ymax : int, optional
        Spatial subregion to sum over, in pixel units, used only if
        `data` is 2D. Defaults to the full range.

    Returns
    -------
    float
        Total HI mass, in solar masses.
    """

    # check the input
    if beamarea is None or beamarea == 0:
        raise ValueError("Provide the beam area in arcsec^2.")
    if beamarea < 0:
        beamarea = np.abs(beamarea)
    if distance is None or distance == 0:
        raise ValueError("Provide the distance in Mpc.")
    if pixelres is None or pixelres == 0:
        raise ValueError("Provide the pixel resolution in arcsec.")

    # convert the data into an array
    data = np.asarray(data)

    if data.ndim == 2:
        # use full spatial range if not specified
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

    # calculate the total flux
    flux = np.nansum(data)

    return HI_CONSTANT_MASS * distance**2 * pixelres**2 * flux / beamarea
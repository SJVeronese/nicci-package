import numpy as np

from nicci.utils.array import extend

def assign_to_cells(data, *, nregions):
    """Partition a 2D array into a near-square grid of rectangular regions.

    The data array is divided into ``nrows * ncols >= nregions`` equally
    sized cells (integer pixel sizes), the resulting grid is centred on
    the array, and the pixel coordinates of each of the first ``nregions``
    cells are returned.

    Parameters
    ----------
    data : ndarray, shape (height, width)
        The 2D array to partition. Only its shape is used.
    nregions : int
        Number of regions (grid cells) to generate. Must be positive.

    Returns
    -------
    xvalid : ndarray of ndarray
        Array of length ``nregions``; each element holds the flattened
        x pixel coordinates of the corresponding region.
    yvalid : ndarray of ndarray
        Array of length ``nregions``; each element holds the flattened
        y pixel coordinates of the corresponding region.
    """

    # check that input number of regions is valid
    if nregions <= 0:
        raise ValueError("nregions must be a positive integer")

    # determine data shape
    height, width = data.shape

    # choose a near-square grid such that nrows * ncols >= nregions
    nrows = int(
        np.floor(np.sqrt(nregions))
    )  # floor ensures that the number of rows is the largest integer not exceeding sqrt(nregions)
    ncols = int(
        np.ceil(nregions / nrows)
    )  # ceil gives the maximum number of columns to cover the entire data array

    # compute uniform cell size (integer division)
    cell_h = height // nrows
    cell_w = width // ncols

    # determine the portion of the array covered by the grid
    used_h = cell_h * nrows
    used_w = cell_w * ncols

    # center the grid within the array
    y0 = (height - used_h) // 2
    x0 = (width - used_w) // 2

    # initialize lists of valid pixel coordinates per region
    xvalid = []
    yvalid = []

    # track how many regions have been generated
    region_count = 0

    # iterate over the grid cells
    for row in range(nrows):
        for col in range(ncols):
            # stop once the requested number of regions is reached
            if region_count >= nregions:
                break

            # compute cell bounds in array coordinates
            ys = y0 + row * cell_h
            ye = ys + cell_h
            xs = x0 + col * cell_w
            xe = xs + cell_w

            # generate all pixel coordinates inside the cell
            y, x = np.meshgrid(np.arange(ys, ye), np.arange(xs, xe), indexing="ij")

            # store flattened coordinates for the current region
            xvalid.append(x.ravel())
            yvalid.append(y.ravel())

            # increment region counter
            region_count += 1

    return np.array(xvalid), np.array(yvalid)


def assign_to_rings(data, *, radii, pa, inc, xpos, ypos, between_angles=None):
    """Assign pixels of a 2D array to a set of concentric elliptical rings.

    Rings are defined by a sequence of ``nrings`` radii and, optionally,
    independent position angle, inclination, and centre per radius
    (tilted-ring geometry). Each pixel is tested against the inner and
    outer boundary of each ring, and optionally restricted to an angular
    sector.

    Parameters
    ----------
    data : ndarray, shape (height, width)
        The 2D array defining the pixel grid. Only its shape is used.
    radii : array_like
        Ring boundary radii, in pixel units, from innermost to outermost.
        There are ``len(radii) - 1`` rings.
    pa : float or array_like
        Position angle(s) of the rings, in degrees. If shorter than
        ``radii``, the last value is repeated to match its length.
    inc : float or array_like
        Inclination(s) of the rings, in degrees. Extended like `pa`.
    xpos, ypos : float or array_like
        Centre coordinates of the rings, in pixel units. Extended like `pa`.
    between_angles : array_like of shape (2,), optional
        Lower and upper position-angle boundary (in degrees, measured in
        the ring's own position-angle frame) restricting each ring to an
        angular sector. If None, the full ring is used.

    Returns
    -------
    xvalid : list of list of int
        For each ring, the x pixel coordinates of the pixels it contains.
    yvalid : list of list of int
        For each ring, the y pixel coordinates of the pixels it contains.
    """

    # ensure radii is an array
    radii = np.asarray(radii)

    # get the number of rings
    nrings = len(radii)

    # extend the relevant quantities
    pa = extend(pa, nrings)
    inc = extend(inc, nrings)
    xpos = extend(xpos, nrings)
    ypos = extend(ypos, nrings)

    # manage the angle boundaries
    if between_angles is not None:
        # convert angle boundaries to "position angle" frame.
        # +90 is needed to match python array orientation with position angle on sky convention
        angles = np.asarray(between_angles) + pa[:, None] + 90

        # convert to radians
        angles = np.radians(angles)

        # normalise angle in the [-np.pi,np.pi) range
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

    # convert to radians
    pa = np.radians(pa)
    inc = np.radians(inc)

    # determine the data shape
    xshape, yshape = data.shape

    # generate the pixel coordinates
    xx, yy = np.meshgrid(np.arange(yshape), np.arange(xshape), indexing="ij")

    # initialize the list of valid pixel coordinates per ring
    xvalid = [[] for _ in range(nrings - 1)]
    yvalid = [[] for _ in range(nrings - 1)]

    # for each ring
    for ring in range(nrings - 1):
        # determine the inner boundary radius, position angle and inclination
        r_min = radii[ring]
        pa_min = pa[ring]
        inc_min = inc[ring]

        # determine the outer boundary radius, position angle and inclination
        r_max = radii[ring + 1]
        pa_max = pa[ring + 1]
        inc_max = inc[ring + 1]

        # compute the pixel positions w.r.t. the ring centers
        dx = xx - xpos[ring]
        dy = yy - ypos[ring]

        # if angle boundaries are provided
        if between_angles is not None:
            # compute the point angle
            theta = np.arctan2(dy, dx)

            # get the angle boundaries
            min_angle, max_angle = angles[ring]

            # mask the pixels outside the angle range
            mask = (max_angle - min_angle) % (2 * np.pi) >= (theta - min_angle) % (2 * np.pi)
        else:
            # all pixels are valid
            mask = np.ones_like(dx, dtype=bool)

        # calculate the inner elliptical radius of the pixel
        d_min = _radius(r_min, dx, dy, pa_min, inc_min)

        # calculate the outer elliptical radius of the pixel
        d_max = _radius(r_max, dx, dy, pa_max, inc_max)

        # If one of the two distances is negative, the point lies between the inner
        # and outer radius and is therefore a valid pixel. Indeed, if a pixel is
        # inside rmin then d_min < 0 and d_max < 0, so d_min * d_max > 0; if the
        # pixel is outside rmax then d_min > 0 and d_max > 0, so d_min * d_max > 0.
        mask &= (np.round(d_min) * np.round(d_max)) <= 0

        # extract the pixels in the ring
        ys, xs = np.nonzero(mask)

        # store the pixel coordinates in the ring
        xvalid[ring] = xs.tolist()
        yvalid[ring] = ys.tolist()

    return xvalid, yvalid


def _radius(radius, x, y, pa, inc):
    """Signed difference between a ring radius and a pixel's elliptical radius.

    Used to test whether a point lies inside, on, or outside an
    elliptical ring of given position angle and inclination; the sign of
    the result indicates which side of the ring boundary the point falls
    on.

    Parameters
    ----------
    radius : float
        Ring (circular) radius to compare against.
    x, y : float or ndarray
        Pixel offset from the ring centre.
    pa : float
        Position angle, in radians.
    inc : float
        Inclination, in radians.

    Returns
    -------
    float or ndarray
        ``radius`` minus the point's elliptical radius; positive inside
        the ring, negative outside.
    """

    # x-radius is the same of a circle
    xr = -x * np.sin(pa) + y * np.cos(pa)

    # y-radius is the same of a circle but reduced by the inclination angle
    yr = (-x * np.cos(pa) - y * np.sin(pa)) / np.cos(inc)

    # return the distance between the radius[i] and the elliptical radius of the point
    return radius - np.sqrt(xr**2 + yr**2)
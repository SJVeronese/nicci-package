import numpy as np
import pandas as pd

from astropy.convolution import convolve, Box1DKernel
from scipy.stats import anderson
from photutils.background import Background2D
from photutils.segmentation import detect_sources, deblend_sources
from scipy.ndimage import binary_dilation
from tqdm.auto import tqdm

from nicci.io.fits import load_data
from nicci.photometry import epsf
from nicci.utils.array import extend, RMS


def lines_finder(spectrum, spectrum_rms=None, *,
                 smooth_kernel=1,
                 finder_threshold=4, finder_replace=None, finder_fluxrange='negative', finder_statistic='mad',
                 linker_kernel=1, linker_minsize=1,
                 reliability=False, reliability_threshold=0.8, reliability_snrmin=5, reliability_kernel=0.4,
                 reliability_tolerance=1e-12, reliability_plot=False, reliability_skellam=False,
                 gaussianity=False, gaussianity_fluxrange='negative', gaussianity_statistic='mad', gaussianity_tries=10000,
                 keep_negative=False, write_catalogue=False, output=None,
                 show_progress_bar=True,
                 **kwargs):
    """Detect and mask line-like sources in one or more spectra.

    For each input spectrum, applies iterative sigma-clipping (optionally
    after smoothing with one or more boxcar kernels) to flag candidate
    channels, links adjacent flagged channels into sources, and can
    estimate source reliability following the SoFiA-2 approach and test
    the noise for gaussianity.

    Parameters
    ----------
    spectrum : ndarray
        Input spectrum, or a 2D array/sequence of spectra (one per row).
    spectrum_rms : float or array_like, optional
        Per-spectrum noise estimate. Computed from each spectrum if not
        given.
    smooth_kernel : int or array_like, default 1
        Boxcar smoothing kernel size(s) used by the source finder, in
        channels. A kernel of 1 disables smoothing for that pass.
    finder_threshold : float, default 4
        Detection threshold, in units of the noise.
    finder_replace : float, optional
        Value (in units of the noise) used to replace already-detected
        channels before smoothing, to avoid source smearing. Defaults to
        `finder_threshold / 2`.
    finder_fluxrange : {'negative', 'positive', 'all'}, default 'negative'
        Flux range used when estimating the noise for the source finder.
    finder_statistic : {'mad', 'std'}, default 'mad'
        Statistic used to estimate the noise for the source finder.
    linker_kernel : int, default 1
        Maximum number of consecutive non-detected channels allowed
        within a single linked source.
    linker_minsize : int, default 1
        Minimum number of channels for a linked source to be kept.
    reliability : bool, default False
        If True, compute a reliability value for each positive source and
        discard unreliable ones.
    reliability_threshold : float, default 0.8
        Minimum reliability for a source to be kept.
    reliability_snrmin : float, default 5
        Minimum S/N for a positive source to be eligible as reliable.
    reliability_kernel : float, default 0.4
        Smoothing scale applied to the reliability covariance matrix.
    reliability_tolerance : float, default 1e-12
        Diagonal regularization added to the covariance matrix.
    reliability_plot : bool, default False
        If True (and `output` is given), save a reliability diagnostic plot.
    reliability_skellam : bool, default False
        If True (and `output` is given), save a Skellam-distribution
        diagnostic plot.
    gaussianity : bool, default False
        If True, test the noise of each spectrum for gaussianity using
        the Anderson-Darling and Kanekar statistics.
    gaussianity_fluxrange : {'negative', 'positive', 'all'}, default 'negative'
        Flux range used for the gaussianity noise estimate.
    gaussianity_statistic : {'mad', 'std'}, default 'mad'
        Statistic used for the gaussianity noise estimate.
    gaussianity_tries : int, default 10000
        Number of Monte Carlo realizations used to calibrate the expected
        range of the Kanekar parameter.
    keep_negative : bool, default False
        If True, negative sources are kept in the output mask (as -1)
        instead of being set to 0.
    write_catalogue : bool, default False
        If True (and `output` is given), write a source catalogue to CSV.
    output : str, optional
        Base output path/name used for diagnostic plots and catalogues.
    show_progress_bar : bool, default True
        Whether to display a `tqdm` progress bar.
    **kwargs
        Additional keyword arguments forwarded to the reliability
        computation and catalogue writer.

    Returns
    -------
    ndarray
        Binary (or ternary, if `keep_negative` is True) detection mask,
        with the same shape as the input spectrum/spectra.
    """

    # determine the number of spectra to process
    if isinstance(spectrum, np.ndarray):
        nspectra = len(spectrum) if spectrum.ndim > 1 or spectrum.dtype == object else 1
    elif isinstance(spectrum, (list, tuple)):
        nspectra = (
            len(spectrum) if any(isinstance(x, (list, tuple, np.ndarray)) for x in spectrum) else 1
        )

    # ensure input spectrum is an iterable
    if spectrum.ndim == 1:
        spectrum = np.array([spectrum])

    # ensure there is at least a noise value per spectrum
    noise = extend(spectrum_rms, nspectra)

    # ensure kernel list is an array
    kernels = np.atleast_1d(smooth_kernel)

    if finder_replace is None:
        finder_replace = finder_threshold / 2
    if output is None and reliability:
        reliability_plot = False
        reliability_skellam = False

    if gaussianity:
        # initialise the gaussianity checks
        gaussian_for_anderson = np.zeros(nspectra)
        gaussian_for_kanekar = np.zeros(nspectra)
        gaussianity_kernel = Box1DKernel(10)
        gaussianity_nchans = 0

    # initialize the list of mask and source IDs
    mask_list = []
    source_ids = []

    # do the source finding
    for i in tqdm(range(len(spectrum)),
                  desc='Spectra analysed', total=len(spectrum),
                  disable=not show_progress_bar):  # for each input spectrum
        # extract the spectrum
        spec = spectrum[i].copy()

        # extract the noise value
        sigma = noise[i]

        # if not given, calculate and store into the noise array for later use
        if sigma is None or np.isnan(sigma):
            sigma = RMS(spec, mode=finder_statistic, fluxrange=finder_fluxrange)
            noise[i] = sigma

        # run the source finder
        mask = _source_finding(spec, sigma,
                               smooth_kernel=kernels,
                               finder_threshold=finder_threshold,
                               finder_replace=finder_replace,
                               finder_fluxrange=finder_fluxrange,
                               finder_statistic=finder_statistic)

        # run the source linker
        mask = _source_linker(spec, mask, linker_kernel, linker_minsize)

        # store the current mask, which contains the source IDs, in the source IDs list
        source_ids.append(mask.copy())

        # perform the gaussianity tests
        if gaussianity:
            # The Kanekar parameter has an expected value of 1, but the finite number
            # of channels and the correlation introduced by smoothing produce a range
            # of values. Check what range of values is expected, which depends on the
            # number of channels.

            # if the spectrum has a different number of channels with respect to the others
            if len(spec) != gaussianity_nchans:
                # store the new number of channels
                gaussianity_nchans = len(spec)

                # initialize the test result list
                test = []
                for tries in range(gaussianity_tries):
                    # create a Gaussian spectrum
                    spec = np.random.normal(size=gaussianity_nchans)

                    # compute the unsmoothed spectrum
                    rms_0 = RMS(spec)

                    # Convolve the spectrum. The boundaries are wrapped to avoid
                    # convolution glitches at the edges and preserve noise properties.
                    spec = convolve(spec, gaussianity_kernel, boundary='wrap')

                    # compute the rms of the smoothed spectrum
                    rms_1 = RMS(spec)

                    # store the expected Kanekar parameter
                    test.append(rms_1 * np.sqrt(10) / rms_0)

                # compute the 95% range
                lower, upper = np.percentile(np.array(test), [2.5, 97.5])

            # test for Gaussianity for anderson-darlin and kanekar
            anders, kanekar = _gaussianity(spec, mask,
                                           statistic=gaussianity_statistic,
                                           fluxrange=gaussianity_fluxrange)

            # Gaussian for Kanekar if within the range.
            # anders is already boolean
            k = lower < kanekar and kanekar < upper

            # store the results
            gaussian_for_anderson[i] = anders
            gaussian_for_kanekar[i] = k

            # convert into array
            gaussian_for_anderson = np.array(gaussian_for_anderson, dtype=bool)
            gaussian_for_kanekar = np.array(gaussian_for_kanekar, dtype=bool)
        else:
            gaussian_for_anderson, gaussian_for_kanekar = None, None

        # make the mask binary
        mask[mask < 0] = -1 if keep_negative else 0
        mask[mask > 0] = 1

        # store the mask
        mask_list.append(mask)

    # calculate the reliability
    if reliability:
        # split the positive and negative sources for the reliability computation
        pos, neg = _split_sources(spectrum, noise, source_ids)

        # run the reliability computation
        reliability_values = _reliability(pos, neg, reliability_snrmin,
                                          kernel=reliability_kernel,
                                          eps=reliability_tolerance,
                                          threshold=reliability_threshold,
                                          plot=reliability_plot,
                                          skellam=reliability_skellam,
                                          output=output,
                                          **kwargs)

        # remove unreliable sources from the mask
        k = 0

        # run through the source_ids array.
        # Each element i contains the sources IDs for each channel of spectrum i
        for i, src_ids in enumerate(source_ids):
            # skip if no positive sources
            if np.all(src_ids <= 0):
                continue

            # positive source IDs
            positive_sources = src_ids[src_ids > 0]

            # for each positive source
            for j in positive_sources:
                # if unreliable
                if reliability_values[k + j - 1] < reliability_threshold:
                    # channels belonging to source j
                    mask = src_ids == j
                    idx = np.where(mask)[0]

                    # minimum, maximum and total channel
                    chanmin = idx[0]
                    chanmax = idx[-1]

                    # remove the unrealiable source j from the mask
                    mask_list[i][chanmin : chanmax + 1] = 0

            # update the reliability index
            k += np.nanmax(positive_sources)
    else:
        reliability_values = None

    # if a catalogue must be written
    if write_catalogue and output:
        catalogue = _write_catalogue(spectrum, source_ids, noise,
                                     gaussianity=gaussianity,
                                     gaussian_for_anderson=gaussian_for_anderson,
                                     gaussian_for_kanekar=gaussian_for_kanekar,
                                     reliability=reliability,
                                     reliability_values=reliability_values)

        # create a copy of the catalogue to retain only the reliable source
        filtered_catalogue = catalogue.copy()
        if reliability:
            # filter unreliable sources
            filtered_catalogue = filtered_catalogue.drop(
                filtered_catalogue[filtered_catalogue['Reliability'] < reliability_threshold].index
            )
        if gaussianity:
            # filter non-Gaussian spectra
            filtered_catalogue = filtered_catalogue.drop(
                filtered_catalogue[
                    (filtered_catalogue['Anderson test']) & (filtered_catalogue['Kanekar test'])
                ].index
            )

        # save the catalouges
        catalogue.to_csv(output + '_sources_catalogue.csv', index=False)
        filtered_catalogue.to_csv(output + '_sources_catalogue_filtered.csv', index=False)

    return np.array(mask_list)


def optical_finder(data, *,
                   sources_pos, gal_pos,
                   psf_fwhm=None,
                   box_size=None, filter_size=9,
                   nsigma=3.0, npixels=10,
                   match_radius=5.0, dilate=0.1,
                   **kwargs):
    """Detect and mask sources around a galaxy from an optical image.

    Detects sources above a local background via image segmentation,
    matches them against a list of candidate source positions,
    and masks either the matched segment's full footprint or,
    for unmatched/undetected candidates, a small fixed aperture.
    The galaxy's own segment is never masked wholesale.

    Parameters
    ----------
    data : str, ndarray, or tuple
        Input image, passed to `nicci.io.fits.load_data`.
    sources_pos : tuple of (array_like, array_like)
        ``(x, y)`` pixel coordinates of the candidate contaminating
        sources (e.g. from a Gaia query).
    gal_pos : tuple of (float, float)
        ``(x, y)`` pixel coordinates of the galaxy centre.
    psf_fwhm : float, optional
        Image PSF FWHM, in pixels, used to size the fallback aperture
        for undetected sources. Estimated via `build_epsf` if not given
        (falling back to 5.0 pixels if that estimate fails).
    box_size : tuple of (int, int), optional
        Background mesh size, in pixels, passed to
        `photutils.background.Background2D`. Defaults to roughly 1/10th
        of the image size.
    filter_size : int, default 9
        Background mesh median-filter size, passed to `Background2D`.
    nsigma : float, default 3.0
        Detection threshold, in units of the background rms.
    npixels : int, default 10
        Minimum number of connected pixels for a detection, passed to
        `detect_sources`/`deblend_sources`.
    match_radius : float, default 5.0
        Maximum distance, in pixels, between a candidate source and a
        detected segment's centroid for them to be considered a match.
    dilate : float, default 0.1
        Mask dilation radius, as a fraction of `match_radius`; no
        dilation is applied if 0.
    **kwargs
        Additional keyword arguments forwarded to `Background2D`,
        `detect_sources`, `deblend_sources`, and `build_epsf` (if
        `psf_fwhm` is not given).

    Returns
    -------
    mask : ndarray of int
        Binary mask, the same shape as `data`, with 1 marking pixels
        belonging to a masked source.
    segmentation_map : photutils.segmentation.SegmentationImage or None
        The deblended segmentation map, or None if no sources were
        detected above threshold.
    """

    # load the data
    data = load_data(data)['data']

    # get the x, y coordinates of the galaxy and the candidate sources
    gal_x, gal_y = gal_pos
    source_x, source_y = sources_pos

    # determine the box size for background estimation
    if box_size is None:
        box_size = (data.shape[0] // 10, data.shape[1] // 10)

    # determine the image PSF
    if psf_fwhm is None:
        try:
            psf_fwhm = epsf(data, **kwargs)['fwhm']
        except Exception as exc:
            print(f"WARNING: Cannot measure the PSF ({exc}). Falling back to a FWHM of 5 pixels.")
            psf_fwhm = 5.0

    # convert the PSF FWHM into a Gaussian sigma
    dispersion = psf_fwhm / (2 * np.sqrt(2 * np.log(2)))

    # build the local background model (segmentation needs a
    # background-subtracted image)
    bkg = Background2D(data, box_size=box_size, filter_size=filter_size, **kwargs)

    # subtract the background
    bkg_subtracted_data = data - bkg.background

    # detect sources
    threshold = nsigma * bkg.background_rms
    segmentation_map = detect_sources(bkg_subtracted_data, threshold, npixels=npixels, **kwargs)

    # deblend overlapping sources
    if segmentation_map is not None:
        segmentation_map = deblend_sources(bkg_subtracted_data, segmentation_map,
                                           npixels=npixels, nlevels=32, contrast=0.001, **kwargs)

    # initialise the mask
    mask = np.zeros_like(data, dtype=int)
    yy, xx = np.ogrid[: data.shape[0], : data.shape[1]]

    if segmentation_map is None or len(segmentation_map.labels) == 0:
        # nothing detected at all: mask every candidate source with the
        # fallback aperture
        for xs, ys in zip(source_x, source_y):
            source = (xx - xs) ** 2 + (yy - ys) ** 2 <= (dispersion / 2) ** 2
            mask[source] = 1
        return mask, segmentation_map

    # identify the galaxy segment. This segment is never allowed to be
    # masked wholesale, regardless of what candidate sources fall in it.
    gxi, gyi = int(np.around(gal_x)), int(np.around(gal_y))
    galaxy_segment = segmentation_map.data[gyi, gxi] if segmentation_map.data[gyi, gxi] != 0 else None

    # determine the centroid (x, y) of every detected segment
    cat_x, cat_y = [], []
    for segment in segmentation_map.labels:
        # y, x pixels covered by the segment
        ys_segment, xs_segment = np.where(segmentation_map.data == segment)

        # the mean position is the centroid
        cat_x.append(xs_segment.mean())
        cat_y.append(ys_segment.mean())

    cat_x, cat_y = np.asarray(cat_x), np.asarray(cat_y)

    # match every candidate source to its closest detected segment
    matched_sources = set()
    for xs, ys in zip(source_x, source_y):
        # distance from the candidate source to every segment centroid
        dist = np.hypot(cat_x - xs, cat_y - ys)

        # closest segment
        idx_min = np.argmin(dist)
        closest_segment = segmentation_map.labels[idx_min]

        # if close enough and not the galaxy itself, flag it as matched;
        # otherwise fall back to the small fixed aperture
        if dist[idx_min] <= match_radius and closest_segment != galaxy_segment:
            matched_sources.add(closest_segment)
        else:
            source = (xx - xs) ** 2 + (yy - ys) ** 2 <= (dispersion / 2) ** 2
            mask[source] = 1

    # mask the full footprint of every matched segment
    for source in matched_sources:
        mask[segmentation_map.data == source] = 1

    # grow the mask slightly to catch faint wings/diffraction spikes that
    # fall below the detection threshold
    if dilate > 0:
        dilate = int(np.around(match_radius * dilate))
        mask = binary_dilation(mask == 1, iterations=dilate).astype(int)

    return mask, segmentation_map


def _extract_reliability_params(sources, *, sign=1.0):
    """Compute log10 peak, sum, and mean flux for a set of sources.

    Parameters
    ----------
    sources : sequence of ndarray
        Flux values of each source (e.g. channels belonging to it).
    sign : float, default 1.0
        Sign convention: use 1.0 for positive sources (peak = max) or
        -1.0 for negative sources (peak = min); the output values are
        always expressed as positive log10 quantities via ``sign * value``.

    Returns
    -------
    ndarray, shape (3, N)
        Stacked array of log10(peak), log10(sum), and log10(mean) flux
        for the `N` input sources.
    """

    # initialize the parameter lists
    peak = []
    summ = []
    mean = []

    # for each source
    for src in sources:
        # calculate the peak flux (the max for positive sources and the min for negative sources)
        (
            peak.append(np.log10(sign * np.nanmax(src)))
            if sign > 0
            else peak.append(np.log10(sign * np.nanmin(src)))
        )

        # calculate the total flux
        summ.append(np.log10(sign * np.nansum(src)))

        # calculate the mean flux
        mean.append(np.log10(sign * np.nanmean(src)))

    return np.vstack([peak, summ, mean])


def _gaussianity(spectrum, mask, *, statistic, fluxrange):
    """Test a spectrum's noise for gaussianity (Anderson-Darling and Kanekar).

    Parameters
    ----------
    spectrum : ndarray
        Input spectrum.
    mask : ndarray
        Detection mask for `spectrum` (zero for noise channels).
    statistic : {'mad', 'std'}
        Statistic used to estimate the noise for the Kanekar test.
    fluxrange : {'negative', 'positive', 'all'}
        Flux range used to estimate the noise for the Kanekar test.

    Returns
    -------
    ander : bool
        True if the Anderson-Darling p-value exceeds 0.05 (consistent
        with gaussian noise).
    kanekar : float
        Kanekar parameter: the ratio of the smoothed to unsmoothed rms,
        rescaled by sqrt(10); expected to be close to 1 for gaussian
        noise with no channel-to-channel correlation.
    """

    # do the Andeson-Darling test for gaussianity.
    # The test is done on the noise channels (spec[mask == 0])
    anderson_test = anderson(spectrum[mask == 0])

    # compute the p-value depending on the test statistic
    # The p-value gives the probability that the spectrum is gaussian.
    if anderson_test.statistic >= 0.6:
        p = np.exp(1.2937 - 5.709 * anderson_test.statistic + 0.0186 * anderson_test.statistic**2)
    elif anderson_test.statistic >= 0.34:
        p = np.exp(0.9177 - 4.279 * anderson_test.statistic - 1.38 * anderson_test.statistic**2)
    elif anderson_test.statistic >= 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * anderson_test.statistic - 59.938 * anderson_test.statistic**2)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * anderson_test.statistic - 223.73 * anderson_test.statistic**2)

    # if p>0.05, the spectrum can be considered gaussian
    ander = p > 0.05

    # Do the Kanekar test for gaussianity. We expect the rms of the spectrum to be
    # reduced by a factor of sqrt(10) ~ 3.16 (see Effect of smoothing window length
    # on RMS EMG amplitude estimates,
    # https://www.researchgate.net/publication/3635815).

    # make a copy of the spectrum
    dummy = spectrum.copy()

    # calculate the rms before smmothing
    rms_0 = RMS(dummy, mode=statistic, fluxrange=fluxrange)

    # blank the detections as we want to test the noise
    dummy[mask != 0] = np.nan

    # define the smoothing boxcar kernel
    kernel = Box1DKernel(10)

    # convolve the spectrum
    dummy = convolve(dummy, kernel)

    # calculate the rms after smmothing
    rms_1 = RMS(dummy, mode=statistic, fluxrange=fluxrange)

    # calculate the Kanekar parameter
    kanekar = rms_1 * np.sqrt(10) / rms_0

    return ander, kanekar


def _kde_density(data, centers, inverse_covariance, *, block_size=512):
    """Evaluate a Gaussian kernel-density estimate at a set of points, in blocks.

    Parameters
    ----------
    data : ndarray, shape (N, D)
        Sample points defining the KDE.
    centers : ndarray, shape (M, D)
        Points at which the density is evaluated.
    inverse_covariance : ndarray, shape (D, D)
        Inverse of the kernel covariance matrix.
    block_size : int, default 512
        Number of evaluation points processed per block, to limit peak
        memory usage.

    Returns
    -------
    ndarray, shape (M,)
        Unnormalized KDE density at each of the `M` evaluation points.
    """

    # number of points to evaluate
    npoints = centers.shape[0]

    # initialise the KDE density array
    kde = np.zeros(npoints)

    for start in range(0, npoints, block_size):
        # determine the ending index pf the current block
        end = min(start + block_size, npoints)

        # determine the KDE evaluation points (block centers)
        cen = centers[start:end]

        # pairwise parameter offsets: data - centers
        diff = data[None, :, :] - cen[:, None, :]

        # compute the quadratic forms (x - mu)^T cov^-1 (x - mu)
        quad = np.einsum('ijk,kl,ijl->ij', diff, inverse_covariance, diff)
        kde[start:end] = np.exp(-0.5 * quad).sum(axis=1)

    return kde


def _reliability(pos_sources, neg_sources, snr_min, *,
                 kernel, eps, threshold, plot, skellam, output,
                 BLOCK=512,  # tune this (256–2048 depending on RAM)
                 **kwargs):
    """Compute source reliability from positive/negative source parameter KDEs.

    Implements the SoFiA-2 reliability measure: the KDE density of
    positive sources is compared to the KDE density of (mirrored)
    negative sources in peak/sum/mean flux space, assuming negative
    detections are pure noise.

    Parameters
    ----------
    pos_sources, neg_sources : sequence of ndarray
        Normalized (flux / rms) channel values of each positive and
        negative source.
    snr_min : float
        Minimum S/N for a positive source to be eligible as reliable.
    kernel : float
        Smoothing scale applied to the negative-source covariance matrix.
    eps : float
        Diagonal regularization added to the covariance matrix for
        numerical stability.
    threshold : float
        Reliability threshold, used only for the diagnostic plot.
    plot : bool
        If True, save a reliability diagnostic plot via `_plot_reliability`.
    skellam : bool
        If True, save a Skellam-distribution diagnostic plot via
        `_plot_skellam`.
    output : str
        Base path used for diagnostic plot output.
    BLOCK : int, default 512
        Block size used by `_kde_density` to limit peak memory usage.
    **kwargs
        Additional keyword arguments; ``plot_format`` (default 'pdf')
        selects the diagnostic plot file format, and the rest are
        forwarded to `_plot_reliability`/`_plot_skellam`.

    Returns
    -------
    ndarray, shape (N_pos,)
        Reliability value of each positive source, in [0, 1]; an empty
        array if there are no positive sources.

    Raises
    ------
    ValueError
        If fewer than two negative detections are available.
    RuntimeError
        If the negative-source covariance matrix cannot be inverted.
    """

    # get the plot format
    plot_format = kwargs.get('plot_format', 'pdf')

    # sanity chekcs
    if len(pos_sources) == 0:
        return np.array([])
    if len(neg_sources) < 2:
        raise ValueError(
            "At least two negative detections are required to estimate a covariance matrix."
        )

    # Compute the S/N of the positive sources. Following the SoFiA-2 source code,
    # the S/N is the summed flux (normalized by the rms) divided by the square root
    # of the number of channels contributing to the source. pos_sources already
    # holds the flux/rms value for each channel.
    snr = np.array([np.nansum(source) / np.sqrt(len(source)) for source in pos_sources])

    # determine which sources have enough S/N
    valid = snr > snr_min

    # Calculate the peak, sum and mean flux for each positive and negative source.
    # The outcome is a (3, N) array, where N is the number of sources and each row
    # holds the peak (1), sum (2) and mean (3) flux.
    pos_params = _extract_reliability_params(pos_sources, sign=1.0)
    neg_params = _extract_reliability_params(neg_sources, sign=-1.0)

    # Compute the covariance matrix for the negative detections. rowvar=True means
    # each row is a variable and each column an observation. bias=True is needed to
    # match the SoFiA-2 covariance matrix, which is normalised by N rather than
    # N - 1.
    cov = np.cov(neg_params, rowvar=True, bias=True)

    # scale the covariance matrix
    cov *= kernel**2

    # add diagonal regularization for numerical stability
    cov += np.eye(cov.shape[0]) * eps

    # invert the covariance matrix
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Covariance matrix inversion failed. " "Check the distribution of negative detections."
        ) from exc

    # transpose the reliability parameters for the KDE computatiom
    pos_params_T = pos_params.T.astype(np.float32)  # shape (Npos,3). Use float 32 to halve memory usage
    neg_params_T = neg_params.T.astype(np.float32)  # shape (Nneg,3). Use float 32 to halve memory usage

    # compute the gaussian KDE of the positive sources
    P = _kde_density(pos_params_T, pos_params_T, cov_inv, block_size=BLOCK)
    N = _kde_density(neg_params_T, pos_params_T, cov_inv, block_size=BLOCK)

    # initialize the reliability array
    reliability = np.zeros(len(pos_sources))

    # valid sources have enough S/N (valid) but also P > N
    mask = valid & (P > N)

    # compute the reliability
    reliability[mask] = (P[mask] - N[mask]) / P[mask]

    # plot the parameter space
    if plot:
        from nicci.plotter.plotter import _plot_reliability

        outname = output + '_rel.' + plot_format

        # do the reliability plot
        _plot_reliability(pos_params, neg_params, cov, reliability, threshold, snr_min, outname, **kwargs)

    # plot the skellam distribution
    if skellam:
        from nicci.plotter.plotter import _plot_skellam

        # we need to recompute the KDE, but now referring to the negative sources
        P = _kde_density(pos_params_T, neg_params_T, cov_inv, block_size=BLOCK)
        N = _kde_density(neg_params_T, neg_params_T, cov_inv, block_size=BLOCK)

        # do the skellam plot
        outname = output + '_skellam.' + plot_format
        _plot_skellam(P, N, kernel, outname, **kwargs)

    return reliability


def _source_finding(spectrum, rms, *,
                    smooth_kernel,
                    finder_threshold, finder_replace, finder_fluxrange, finder_statistic):
    """Flag candidate source channels in a spectrum via thresholding and smoothing.

    Applies sigma-clipping to the raw spectrum and, optionally, repeats
    the detection after smoothing with one or more boxcar kernels
    (replacing already-detected channels first, to avoid smearing real
    sources into the noise estimate).

    Parameters
    ----------
    spectrum : ndarray
        Input spectrum.
    rms : float
        Noise estimate of the spectrum.
    smooth_kernel : ndarray
        Boxcar smoothing kernel sizes to try, in channels (sizes of 1 or
        less are skipped).
    finder_threshold : float
        Detection threshold, in units of the noise.
    finder_replace : float
        Value, in units of the (unsmoothed) noise, used to replace
        already-detected channels before smoothing.
    finder_fluxrange : {'negative', 'positive', 'all'}
        Flux range used when re-estimating the noise after smoothing.
    finder_statistic : {'mad', 'std'}
        Statistic used when re-estimating the noise after smoothing.

    Returns
    -------
    ndarray
        Integer mask, the same shape as `spectrum`, counting how many of
        the detection passes (raw plus each smoothing kernel) flagged
        each channel.
    """

    # initialize the mask as zeros
    mask = np.zeros(spectrum.shape, dtype=int)

    # set to 1 the channels above the sigma threshold
    mask[np.abs(spectrum) > finder_threshold * rms] = 1

    # set the smoothed sigma to the one of the spectrum
    smooth_sigma = np.array(rms)

    # if smoothing kernels are applied
    if smooth_kernel.size > 0:
        # for each smoothing kernel
        for ker in smooth_kernel:
            # smooth only when input kernel is > 1
            if ker > 1:
                # create a copy of the original spectrum
                spec_tmp = spectrum.copy()

                # replace the already detected channels to avoid source smearing
                # find the detected channels
                detected_channels = mask > 0

                # replace while retaining the sign
                spec_tmp[detected_channels] = (finder_replace * rms * np.sign(spec_tmp[detected_channels]))

                # define the boxcar kernel
                kernel = Box1DKernel(ker)

                # smooth the spectrum
                smooth_spectrum = convolve(spec_tmp, kernel)

                # calculate the new rms
                smooth_sigma = RMS(smooth_spectrum, mode=finder_statistic, fluxrange=finder_fluxrange)

                # create a tempurary mask
                mask_tmp = np.zeros(smooth_spectrum.shape, dtype=int)

                # apply the sigma clipping
                mask_tmp[np.abs(smooth_spectrum) > finder_threshold * smooth_sigma] = 1

                # add to the mask
                mask = np.add(mask, mask_tmp)

    return mask


def _source_linker(data, mask, kernel, min_size):
    """Link adjacent detected channels into separate positive/negative sources.

    Scans a binary (or counted) detection mask and groups runs of
    detected channels, tolerating up to `kernel` consecutive
    non-detected channels within a single source, then assigns each
    resulting group a unique positive or negative integer source ID
    depending on the sign of its total flux.

    Parameters
    ----------
    data : ndarray
        Spectrum the mask was derived from, used to determine the sign
        (and total flux) of each candidate source.
    mask : ndarray
        Detection mask (nonzero where a channel is flagged).
    kernel : int
        Maximum number of consecutive non-detected channels allowed
        within a single linked source.
    min_size : int
        Minimum number of channels for a linked source to be kept.

    Returns
    -------
    ndarray
        Integer mask, the same shape as `mask`: 0 for unlinked channels,
        positive integers (1, 2, ...) for positive sources in order of
        detection, and negative integers (-1, -2, ...) for negative
        sources in order of detection.
    """

    # set the number of channels
    n_channels = mask.size

    # initialized the mask
    linked_mask = np.zeros(n_channels, dtype=int)

    # initialize the source indeces
    pos_idx = 1
    neg_idx = -1

    # initialize the channel counter
    i = 0

    # scan along the channels
    while i < n_channels:
        # skip non-detected channels and keep running
        if mask[i] == 0:
            i += 1
            continue

        # store the starting channel
        starting_channel = i

        # initialize the zeros counter
        zeros_found = 0

        # extend the source until kernel-consecutive zeros are found
        while i < n_channels:
            # if channel i is non-detected
            if mask[i] == 0:
                # increase the zeros counter
                zeros_found += 1

                # if the number of consecutive zeros exceed the linker kernel
                if zeros_found >= kernel:
                    break

            # if channel i is detected
            else:
                # reset the zeros counter
                zeros_found = 0

            # increase the channel counter
            i += 1

        # The source ends at the first zero found; +1 accounts for NumPy's
        # half-open slicing semantics.
        ending_channel = i - zeros_found + 1

        # extract the source channels
        source_mask = mask[starting_channel:ending_channel] != 0

        # indices of the channels relative to the starting channel of the source
        detected_indices = np.where(source_mask)[0]

        # determine the source size
        source_size = detected_indices.size

        # if of covering enough channels
        if source_size >= min_size:
            # calculate the total flux
            flux_sum = np.nansum(data[starting_channel:ending_channel][detected_indices])

            if flux_sum > 0:
                linked_mask[starting_channel:ending_channel][source_mask] = pos_idx
                pos_idx += 1
            elif flux_sum < 0:
                linked_mask[starting_channel:ending_channel][source_mask] = neg_idx
                neg_idx -= 1
            # zero-flux sources are discarded (remain zero)

        # continue scanning after the source
        i += 1

    return linked_mask


def _split_sources(spectrum, noise, source_ids):
    """Split per-spectrum source masks into lists of normalized positive/negative sources.

    Parameters
    ----------
    spectrum : sequence of ndarray
        Input spectra.
    noise : sequence of float
        Per-spectrum noise estimate, used to normalize the channel values.
    source_ids : sequence of ndarray
        Per-spectrum source ID mask, as produced by `_source_linker`
        (negative IDs for negative sources, positive IDs for positive
        sources).

    Returns
    -------
    pos : ndarray of object
        Array of 1D arrays, one per positive source, holding its
        noise-normalized channel values.
    neg : ndarray of object
        Array of 1D arrays, one per negative source, holding its
        noise-normalized channel values.
    """

    # initialize the containers for results
    pos = []
    neg = []

    # loop once over all spectra, noises, and source-ID masks together.
    # enumerate() gives us the spectrum index `i`, which we need to keep
    # track of where each positive source came from.
    for i, (spec, rms, mask) in enumerate(zip(spectrum, noise, source_ids)):

        # identify pixels that belong to any source.
        non_zero_mask = mask != 0

        # if this spectrum has no sources at all, skip it immediately.
        if not np.any(non_zero_mask):
            continue

        # extract the channels belonging to sources
        masked_spec = spec[non_zero_mask]
        masked_ids = mask[non_zero_mask]

        # normalize the spectrum by its noise.
        # This step is needed for the reliability computation
        normalized_spec = masked_spec / rms

        # group channels by source ID.
        # `ids` is the sorted array of unique source IDs present
        # `inverse` gives, for each channel, the index of its source ID in the `ids` array
        ids, inverse = np.unique(masked_ids, return_inverse=True)

        # loop over sources and split them
        for idx, source_id in enumerate(ids):

            # extract all channels belonging to the source idx
            source_values = normalized_spec[inverse == idx]

            # separate positive and negative sources
            if source_id < 0:
                neg.append(source_values)
            else:
                pos.append(source_values)

    # convert lists to NumPy arrays.
    # dtype=object is used because sources can have different lengths.
    pos = np.array(pos, dtype=object)
    neg = np.array(neg, dtype=object)

    return pos, neg


def _write_catalogue(spectrum, source_ids, noise, *,
                     gaussianity, gaussian_for_anderson, gaussian_for_kanekar,
                     reliability, reliability_values):
    """Build a per-source catalogue DataFrame from source-finding results.

    Parameters
    ----------
    spectrum : sequence of ndarray
        Input spectra.
    source_ids : sequence of ndarray
        Per-spectrum source ID mask, as produced by `_source_linker`.
    noise : sequence of float
        Per-spectrum noise estimate.
    gaussianity : bool
        Whether gaussianity test results are available and should be
        included as columns.
    gaussian_for_anderson, gaussian_for_kanekar : ndarray or None
        Per-spectrum Anderson-Darling and Kanekar gaussianity test
        results, used only if `gaussianity` is True.
    reliability : bool
        Whether reliability values are available and should be included
        as columns.
    reliability_values : ndarray or None
        Reliability value of each positive source, in detection order
        across all spectra, used only if `reliability` is True.

    Returns
    -------
    pandas.DataFrame
        One row per detected source, with columns ``'Spectrum ID'``,
        ``'Source ID'``, ``'Channel min'``, ``'Channel max'``,
        ``'Channels'``, ``'Flux'``, ``'Rms'``, ``'SNR'``, ``'Mean'``,
        ``'Median'``, and (if requested) ``'Anderson test'``,
        ``'Kanekar test'``, ``'Reliability'``, ``'Log(Sum)'``,
        ``'Log(Peak)'``, ``'Log(Mean)'``.
    """

    # To efficiently build the catalogue, each row is created as a list and appended
    # to a list of rows; the resulting list of lists (list[0] is row 0) is then
    # converted into a DataFrame.

    # initialise the catalogue
    catalogue = []

    # column names
    colnames = [
        'Spectrum ID',
        'Source ID',
        'Channel min',
        'Channel max',
        'Channels',
        'Flux',
        'Rms',
        'SNR',
        'Mean',
        'Median',
    ]

    if gaussianity:
        colnames += ['Anderson test', 'Kanekar test']

    if reliability:
        colnames += ['Reliability', 'Log(Sum)', 'Log(Peak)', 'Log(Mean)']
        k = 0  # reliability index offset

    # loop over spectra
    for i, (spec, src_ids) in enumerate(zip(spectrum, source_ids)):

        if np.all(src_ids == 0):
            continue

        # compute the mean
        spec_mean = np.nanmean(spec)

        # compute the median
        spec_median = np.nanmedian(spec)

        # store the noise
        rms = noise[i]

        # unique non-zero source IDs
        unique_sources = np.unique(src_ids[src_ids != 0])

        # for each unique source
        for j in unique_sources:
            # channels belonging to source j
            mask = src_ids == j
            idx = np.where(mask)[0]

            # minimum, maximum and total channel
            chanmin = idx[0]
            chanmax = idx[-1]
            nchan = idx.size

            # total flux
            flux = np.nansum(spec[mask])

            # S/N
            snr = flux / (np.sqrt(nchan) * rms)

            # build the row
            row = [
                i + 1,
                int(j),  # spectrum ID and source ID
                chanmin,
                chanmax,
                nchan,
                flux,
                rms,
                snr,
                spec_mean,
                spec_median,
            ]

            if gaussianity:
                row += [gaussian_for_anderson[i], gaussian_for_kanekar[i]]

            if reliability:
                # normalise the sources by the rms
                norm_spec = spec[mask] / rms

                # if a positive source
                if j > 0:
                    # get its reliability.
                    # -1 accounts for j being always > 1
                    r = reliability_values[k + int(j) - 1]

                    row += [
                        r,
                        np.log10(np.nansum(norm_spec)),
                        np.log10(np.nanmax(norm_spec)),
                        np.log10(np.nanmean(norm_spec)),
                    ]
                # if a negative source
                else:
                    # its reliability is 0
                    row += [
                        0,
                        np.log10(-np.nansum(norm_spec)),
                        np.log10(-np.nanmin(norm_spec)),
                        np.log10(-np.nanmean(norm_spec)),
                    ]

            # store the row
            catalogue.append(row)

        if reliability:
            # unique positive IDs in the spectrum i
            positive_sources = unique_sources[unique_sources > 0]
            # if there are positive sources
            if positive_sources.size:
                # update k so that the reliability values stay aligned across spectra
                k += int(np.max(positive_sources))

    return pd.DataFrame(catalogue, columns=colnames)
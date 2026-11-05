import numpy as np

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


def effective_radius(r, Sigma):
    """Compute the (projected) half-light radius of a surface density profile.

    Parameters
    ----------
    r : array_like
        Radii at which `Sigma` is sampled; must be strictly increasing.
    Sigma : array_like
        Surface density (or surface brightness) at `r`; must be
        non-negative.

    Returns
    -------
    float
        Radius enclosing half of the total projected mass/light.

    Raises
    ------
    ValueError
        If `r` and `Sigma` have different shapes, if `r` is not strictly
        increasing, if `Sigma` contains negative values, or if the total
        enclosed mass is zero or negative.
    """

    r = np.asarray(r, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    # basic sanity checks
    if r.shape != Sigma.shape:
        raise ValueError("r and Sigma must have the same shape.")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r must be strictly increasing.")
    if np.any(Sigma < 0):
        raise ValueError("Sigma must be non-negative.")

    # cumulative projected mass: M(<r) = integral_0^r 2*pi*r'*Sigma(r') dr'
    integrand = 2.0 * np.pi * r * Sigma  # [Msun / kpc]
    M_cumulative = np.concatenate(([0.0], cumulative_trapezoid(integrand, r)))
    M_total = M_cumulative[-1]

    if M_total <= 0:
        raise ValueError("Total mass is zero or negative; check your inputs.")

    # interpolate to find the radius enclosing half of the total mass
    interp = interp1d(M_cumulative, r, kind='linear')
    R_eff = float(interp(0.5 * M_total))

    return R_eff


def exponential(R, Sigma_0, R_d):
    """Exponential surface density profile.

    Parameters
    ----------
    R : array_like
        Radii.
    Sigma_0 : float
        Central surface density.
    R_d : float
        Scale length.

    Returns
    -------
    ndarray
        Surface density at `R`.
    """
    return Sigma_0 * np.exp(-R / R_d)


def fit_profile(radii, sdens, *, sdens_err=None, model='sersic', initial_guess=None, bounds=None):
    """Fit a composite surface density profile to radial data.

    Fits a sum of exponential, Sersic, and/or smooth broken-exponential
    components (see `exponential`, `sersic`, `smooth_broken_exponential`)
    to an observed surface density profile by least squares, with
    data-driven initial guesses, bounds, and Gaussian priors generated
    automatically for any parameter not explicitly provided.

    Parameters
    ----------
    radii : array_like
        Radii at which `sdens` is sampled.
    sdens : array_like
        Surface density values at `radii`.
    sdens_err : array_like, optional
        1-sigma uncertainty on `sdens`; defaults to 1 for every point.
    model : str, tuple, or list, default 'sersic'
        Model specification, normalized by `_normalize_model_format`
        (e.g. ``'exp'``, ``[('sersic', {}), ('exp', {})]``, or
        ``[('broken_exp', {'fix_alpha': True})]``).
    initial_guess : array_like, optional
        Initial parameter vector; derived from the data if not given.
    bounds : tuple of (array_like, array_like), optional
        Lower and upper parameter bounds; derived from the data if not
        given.

    Returns
    -------
    dict
        Dictionary with keys ``'parameters'`` (best-fit parameters, in
        linear units), ``'errors'``, ``'covariance'`` (both propagated to
        linear units), ``'chi2'``, ``'chi2_red'``, ``'dof'``, ``'aic'``,
        ``'bic'``, ``'model'`` (best-fit model evaluated at `radii`),
        ``'ls_output'`` (the raw `scipy.optimize.least_squares` result),
        and, if the model includes a ``'broken_exp'`` component,
        ``'knee_radius'``, ``'knee_dens'``, and ``'knee_dens_err'`` (the
        break radius and the total model surface density, with
        propagated uncertainty, evaluated there).

    Notes
    -----
    Internally, amplitude parameters (``Sigma_e``, ``Sigma_0``) are
    fitted in log10 space for numerical stability and transformed back to
    linear units for the returned parameters, errors, and covariance.
    """

    # internal function to build the model (r defaults to the fit radii,
    # but can be overridden e.g. to evaluate the model at a break radius)
    def build_model(params, r=None):
        """Evaluate the composite surface density model at given radii.

        Parameters
        ----------
        params : array_like
            Flat parameter vector for all model components, in the
            (partially log-space) fit parametrization.
        r : array_like, optional
            Radii to evaluate at; defaults to the fit radii `radii`.

        Returns
        -------
        ndarray
            Total model surface density at `r`, clipped to be positive
            (minimum 1e-30).
        """
        if r is None:
            r = radii
        # initialise the model array
        model_array = np.zeros_like(r, dtype=float)

        # initialise the parameter index
        i = 0
        for component, opts in model:
            # number of parameters used by that component
            npar = _n_params(component, opts)

            # get the component parameters
            comp_params = params[i : i + npar]

            # add the component to the total model
            if component == 'sersic':
                log_Sigma_e, R_e, n = comp_params
                model_array += sersic(r, 10**log_Sigma_e, R_e, n)

            elif component == 'exp':
                log_Sigma_0, R_d = comp_params
                model_array += exponential(r, 10**log_Sigma_0, R_d)

            elif component == 'broken_exp':
                if opts['fix_alpha']:
                    Sigma_0, beta, R_b, gamma = comp_params
                    alpha = 0.0
                else:
                    Sigma_0, alpha, beta, R_b, gamma = comp_params
                model_array += smooth_broken_exponential(r, Sigma_0, alpha, beta, R_b, gamma)

            # update the parameters index
            i += npar

        return np.clip(model_array, 1e-30, None)

    # internal function to compute the residuals
    def residuals(params):
        """Uncertainty-weighted residuals of the surface density fit, plus Gaussian priors.

        Parameters
        ----------
        params : array_like
            Trial parameter vector.

        Returns
        -------
        ndarray
            Concatenation of the data residuals ``(sdens - model) / sdens_err``
            and the Gaussian prior residuals ``(param - mu) / sigma``.
        """
        # build the model
        model_vals = build_model(params)

        # compute the uncertainty-weighted residuals
        res = (sdens - model_vals) / sdens_err

        # add the Gaussian prior penalties
        prior_res = [(p - mu) / sig for p, (mu, sig) in zip(params, priors)]

        return np.concatenate([res, prior_res])

    # internal function to convert the best-fit parameters from log to linear space
    def transform_parameters(params):
        """Convert the composite model's best-fit parameters from the fit to linear space.

        Parameters
        ----------
        params : array_like
            Flat parameter vector in the fit parametrization (amplitude
            parameters in log10 space).

        Returns
        -------
        ndarray
            Flat parameter vector with amplitude parameters converted to
            linear space.
        """
        linear = []

        # initialise the parameter index
        i = 0
        for component, opts in model:
            # number of parameters used by that component
            npar = _n_params(component, opts)

            # get the component parameters
            comp_params = params[i : i + npar]

            # transfor to linear
            if component == 'sersic':
                log_Sigma_e, R_e, n = comp_params
                linear += [10**log_Sigma_e, R_e, n]

            elif component == 'exp':
                log_Sigma_0, R_d = comp_params
                linear += [10**log_Sigma_0, R_d]

            elif component == 'broken_exp':
                if opts['fix_alpha']:
                    Sigma_0, beta, R_b, gamma = comp_params
                    linear += [10**Sigma_0, beta, R_b, gamma]
                else:
                    Sigma_0, alpha, beta, R_b, gamma = comp_params
                    linear += [10**Sigma_0, alpha, beta, R_b, gamma]

            # update the parameters index
            i += npar
        return np.array(linear)

    # ensure input model is in readable format
    model = _normalize_model_format(model)

    # ensure inputs are arrays
    radii = np.asarray(radii, dtype=float)
    sdens = np.asarray(sdens, dtype=float)
    sdens_err = (np.asarray(sdens_err, dtype=float) if sdens_err is not None else np.ones_like(sdens))

    # filter nans
    valid = np.isfinite(radii) & np.isfinite(sdens) & np.isfinite(sdens_err)
    radii = radii[valid]
    sdens = sdens[valid]
    sdens_err = sdens_err[valid]

    # ensure numerical stability for low surface density values
    sdens = np.clip(sdens, 1e-30, None)
    sdens_err = np.clip(sdens_err, 1e-30, None)

    # derive data-driven quantities
    log_sdens = np.log10(sdens)
    sdens_max = np.nanmax(log_sdens)
    R_med = np.nanmedian(radii)
    R_max = np.nanmax(radii)
    R_min = np.maximum(1e-2, np.nanmin(radii))

    # compute priors for Gaussian penalties
    priors = []
    for component, opts in model:
        if component == 'sersic':
            n_guess = 2.0
            b_n_guess = (
                2.0 * n_guess
                - 1 / 3
                + 4.0 / (405 * n_guess)
                + 46.0 / (25515 * n_guess**2)
                + 131.0 / (114875 * n_guess**3)
                - 2194697.0 / (30690717750 * n_guess**4)
            )
            log_Sigma_e_guess = sdens_max - b_n_guess / np.log(10)

            priors += [
                (log_Sigma_e_guess, 2.0),  # log Sigma_e, scatter
                (R_med, 0.5 * R_med),  # R_e, scatter
                (n_guess, 1.0),
            ]  # n, scatter

        elif component == 'exp':
            # crude slope evaluation for scale length
            slope = np.gradient(log_sdens) / np.gradient(radii)
            slope_outer = np.nanmedian(slope[-max(3, len(slope) // 3) :])
            R_d_guess = (np.minimum(-1 / (slope_outer * np.log(10)), R_max) if slope_outer < 0 else R_med)

            priors += [
                (sdens_max, 2.0),  # log Sigma_0, scatter
                (R_d_guess, 0.5 * R_d_guess),
            ]  # R_d, scatter

        elif component == 'broken_exp':
            # the knee occurs always around 5x10^20 because of the atomic-to-molecular transition
            dens_knee = np.log10(5 * 10.0**20)

            # find the radius where the profile reach this point
            knee_loc = np.argmin(np.abs(log_sdens - dens_knee))  # index of the knee

            # initial guess on the knee radius
            R_b_guess = radii[knee_loc]

            # split the profile in before and after the break
            inner_mask = radii <= R_b_guess
            outer_mask = radii >= R_b_guess

            # initial guess on the inner slope
            if inner_mask.sum() >= 2:
                alpha_guess = np.minimum(0.0,
                                         (log_sdens[inner_mask][-1] - log_sdens[inner_mask][0])
                                         / (radii[inner_mask][-1] - radii[inner_mask][0]))
            else:
                alpha_guess = 0.0

            # initial guess on the outer slope
            if outer_mask.sum() >= 2:
                beta_guess = (log_sdens[outer_mask][-1] - log_sdens[outer_mask][0]) / (radii[outer_mask][-1] - radii[outer_mask][0])
            else:
                beta_guess = slope_outer

            # initial guess on the central surface density
            Sigma_0_guess = log_sdens[0] - alpha_guess * radii[0]

            # initial guess on the smooth factor
            gamma_guess = 0.5

            base = [(Sigma_0_guess, 2.0)]  # Sigma_0, scatter
            if not opts['fix_alpha']:
                base += [(alpha_guess, 1.0)]  # alpha, scatter
            base += [
                (beta_guess, 1.0),  # beta, scatter
                (R_b_guess, 0.5 * R_b_guess),  # R_b, scatter
                (gamma_guess, 5.0),
            ]  # gamma, scatter
            priors += base

    # determine initial guesses if not given
    if initial_guess is None:
        initial_guess = []
        for component, opts in model:
            if component == 'sersic':
                initial_guess += [log_Sigma_e_guess, R_med, n_guess]

            elif component == 'exp':
                initial_guess += [sdens_max, R_d_guess]

            elif component == 'broken_exp':
                if opts['fix_alpha']:
                    initial_guess += [Sigma_0_guess, beta_guess, R_b_guess, gamma_guess]
                else:
                    initial_guess += [
                        Sigma_0_guess,
                        alpha_guess,
                        beta_guess,
                        R_b_guess,
                        gamma_guess,
                    ]

    # determine parameter boundaries if not given
    if bounds is None:
        lower, upper = [], []
        for component, opts in model:
            if component == 'sersic':
                lower += [sdens_max - 5, R_min, 0.3]
                upper += [sdens_max + 5, 10.0 * R_max, 8.0]

            elif component == 'exp':
                lower += [sdens_max - 5, R_min]
                upper += [sdens_max + 5, 10.0 * R_max]

            elif component == 'broken_exp':
                if opts['fix_alpha']:
                    lower += [-np.inf, -np.inf, 0, 0.01]
                    upper += [+np.inf, +np.inf, R_max, 50.0]
                else:
                    lower += [-np.inf, -np.inf, -np.inf, 0, 0.01]
                    upper += [+np.inf, +np.inf, +np.inf, R_max, 50.0]
        bounds = (lower, upper)

    # do the least squares fit
    result = least_squares(residuals, initial_guess, bounds=bounds)

    # compute the fit statistics
    dof = np.maximum(len(sdens) - len(result.x), 1)
    chi2 = np.nansum(result.fun[: len(sdens)] ** 2)
    chi2_red = chi2 / dof

    # compute the covariance matrix and parameter errors.
    # We need to account for the fact that amplitude parameters are fit in
    # log space, while shape parameters are in linear space.
    # Therefore we transform the covariance matrix into full linear space.
    cov = None
    errors = None
    if result.jac is not None:
        try:
            # consider only the data part of the jacobian (not the priors)
            J = result.jac[: len(sdens)]

            # covariance matrix in the (partially log) fit-parameter space
            cov_log = np.linalg.pinv(J.T @ J) * chi2_red

            # which parameters are log-amplitudes (need the 10**x transform)
            param_is_log = []
            for component, opts in model:
                npar = _n_params(component, opts)
                param_is_log += [True] + [False] * (npar - 1)

            # initialise the transform matrix with ones on the diagonal and zeros elsewhere
            transform = np.eye(len(result.x))
            for idx, is_log in enumerate(param_is_log):
                if is_log:
                    val = 10 ** result.x[idx]

                    # populate the diagonal with the linear parameter
                    transform[idx, idx] = np.log(10) * val

            # compute the covariance matrix
            cov = transform @ cov_log @ transform.T

            # compute the errors
            errors = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            cov = None
            errors = None

    # compute AIC and BIC
    k = len(result.x)
    n = len(sdens)
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)

    output = {
        'parameters': transform_parameters(result.x),
        'errors': errors,
        'covariance': cov,
        'chi2': chi2,
        'chi2_red': chi2_red,
        'dof': dof,
        'aic': aic,
        'bic': bic,
        'model': build_model(result.x),
        'ls_output': result,
    }

    # derived quantity: total model surface density (and its uncertainty)
    # at the break radius Rb of the first 'broken_exp' component, if any.
    # The uncertainty is propagated numerically (finite differences of
    # build_model with respect to all raw fit parameters, evaluated at
    # R=Rb) through the raw-parameter covariance cov_log, rather than via
    # a hardcoded analytic gradient -- this generalises automatically to
    # any combination of components.
    if cov is not None:
        i = 0
        for component, opts in model:
            npar = _n_params(component, opts)

            if component == 'broken_exp':
                Rb_local_idx = 2 if opts['fix_alpha'] else 3
                Rb_val = transform_parameters(result.x)[i + Rb_local_idx]

                eps = 1e-6
                grad = np.zeros(len(result.x))
                base_val = build_model(result.x, r=np.array([Rb_val]))[0]
                for j in range(len(result.x)):
                    step = eps * max(abs(result.x[j]), 1.0)
                    p_pert = result.x.copy()
                    p_pert[j] += step
                    grad[j] = (build_model(p_pert, r=np.array([Rb_val]))[0] - base_val) / step

                knee_dens_err = np.sqrt(grad @ cov_log @ grad)

                output['knee_radius'] = Rb_val
                output['knee_dens'] = base_val
                output['knee_dens_err'] = knee_dens_err
                break
            i += npar

    return output


def sersic(R, Sigma_e, R_e, n):
    """Sersic surface density profile.

    Parameters
    ----------
    R : array_like
        Radii.
    Sigma_e : float
        Surface density at the effective radius `R_e`.
    R_e : float
        Effective (half-light) radius.
    n : float
        Sersic index.

    Returns
    -------
    ndarray
        Surface density at `R`, the same shape as `R`, or an array of
        zeros if any of `Sigma_e`, `R_e`, `n` is zero.
    """
    # if at least one parameter is not given
    if np.any((Sigma_e, R_e, n)) == 0:
        return np.zeros_like(R)

    b_n = (
        2 * n
        - 1 / 3
        + 4 / (405 * n)
        + 46 / (25515 * n**2)
        + 131 / (114875 * n**3)
        - 2194697 / (30690717750 * n**4)
    )
    return Sigma_e * np.exp(-b_n * ((R / R_e) ** (1 / n) - 1))


def smooth_broken_exponential(R, Sigma_0, alpha, beta, R_b, gamma):
    """Smooth broken-exponential surface density profile (Erwin 2008, Eq. 5).

    Parameters
    ----------
    R : array_like
        Radii.
    Sigma_0 : float
        log10 of the extrapolated central surface density.
    alpha : float
        Inner, log-space slope.
    beta : float
        Outer, log-space slope.
    R_b : float
        Break radius.
    gamma : float
        Sharpness of the transition between the inner and outer slopes.

    Returns
    -------
    ndarray
        Surface density at `R`, in linear units.
    """
    # reference: https://ui.adsabs.harvard.edu/abs/2008AJ....135...20E/abstract
    return 10.0 ** (Sigma_0
        - (alpha / np.log(10)) * R
        - ((beta - alpha) * gamma / np.log(10)) * np.log(1 + np.exp((R - R_b) / gamma))
    )  # Eq.(5)


def _normalize_model_format(model):
    """Normalize a surface density model specification into a list of (name, opts) tuples.

    Parameters
    ----------
    model : str, tuple, or list
        One or more model components, each either a name (``'exp'``,
        ``'sersic'``, or ``'broken_exp'``) or a ``(name, opts)`` tuple
        with a dict of component-specific options.

    Returns
    -------
    list of tuple
        Normalized list of ``(name, opts)`` tuples, one per component,
        with ``opts['fix_alpha']`` defaulted to False for
        ``'broken_exp'`` components that did not specify it.

    Raises
    ------
    ValueError
        If a component name is not one of ``'exp'``, ``'sersic'``, or
        ``'broken_exp'``.
    """

    # ensure input model is a list
    if not isinstance(model, list):
        model = [model]

    normalised = []
    for component in model:
        # if the component is a tuple, the first element is the name and the second the options
        if isinstance(component, tuple):
            # get the name of the component and the options
            name, opts = component

            # ensure the options are a dictionary, even if not given
            opts = dict(opts) if opts else {}
        else:
            # get the name and initialise the options as a dictionary
            name, opts = component, {}

        # if the name is not supported
        if name not in ('exp', 'sersic', 'broken_exp'):
            raise ValueError("Model must be 'exp', 'sersic' and/or 'broken_exp'")

        # if the component is a smooth broken exponential, than set the default options
        if name == 'broken_exp':
            opts.setdefault('fix_alpha', False)

        # store the normalised model
        normalised.append((name, opts))
    return normalised


def _n_params(model, opts):
    """Number of free parameters for a surface density model component.

    Parameters
    ----------
    model : {'exp', 'sersic', 'broken_exp'}
        Component name.
    opts : dict
        Component options; only ``opts['fix_alpha']`` is used, for the
        ``'broken_exp'`` component.

    Returns
    -------
    int
        Number of free parameters: 2 for ``'exp'``, 3 for ``'sersic'``,
        and 4 or 5 for ``'broken_exp'`` depending on `opts['fix_alpha']`.
    """
    if model == 'sersic':
        return 3
    if model == 'exp':
        return 2
    if model == 'broken_exp':
        return 4 if opts['fix_alpha'] else 5
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from mcfit import Hankel
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq, least_squares
from scipy.special import erfcx
from tqdm.auto import tqdm

from nicci.utils import constants
from nicci.plotter.plotter import make_figure
from nicci.utils.array import extend


def circular_velocity(radii, sdens, *, sigma_r, sigma_phi, **radial_force_kwargs):
    """Compute the circular velocity of a disc from its surface density profile.

    Combines the radial gravitational force (via `_radial_force`) with an
    optional asymmetric drift correction.

    Parameters
    ----------
    radii : ndarray
        Galactocentric radii, in kpc.
    sdens : ndarray
        Surface density at `radii`, in Msun/kpc^2.
    sigma_r : ndarray or None
        Radial velocity dispersion at `radii`, in km/s; the asymmetric
        drift correction is skipped if None or identically zero.
    sigma_phi : ndarray or None
        Azimuthal velocity dispersion at `radii`, in km/s.
    **radial_force_kwargs
        Additional keyword arguments forwarded to `_radial_force`.

    Returns
    -------
    ndarray
        Circular velocity at `radii`, in km/s, clipped to be non-negative.
    """

    # compute the radial force
    radial_force = -_radial_force(radii, sdens, **radial_force_kwargs)

    # compute the circular velocity
    velocity_squared = radii * (-radial_force)

    # compute the asymmetric drift
    if sigma_r is None or np.all(sigma_r == 0.0):
        pass
    else:
        velocity_squared += _asymmetric_drift(radii, sdens, sigma_r, sigma_phi)

    # numerical safety (avoid tiny negative values)
    velocity_squared = np.maximum(velocity_squared, 0.0)

    return np.sqrt(velocity_squared)


def circular_velocity_dm(radii, halo, **params):
    """Circular velocity of a dark matter halo.

    Parameters
    ----------
    radii : array_like
        Galactocentric radii, in kpc.
    halo : {'iso', 'burkert', 'nfw', 'dc14'}
        Dark matter halo density profile.
    **params
        Halo-specific parameters:
        - 'iso', 'burkert' : core_density [Msun/kpc^3], core_radius [kpc]
        - 'nfw'            : M200 [Msun], concentration
        - 'dc14'           : M200 [Msun], concentration, Mstar [Msun, optional]

    Returns
    -------
    ndarray
        Circular velocity, in km/s.

    Notes
    -----
    'nfw' and 'dc14' assume a flat LambdaCDM cosmology with
    H0 = 70 km/s/Mpc and a 200x-the-critical-density halo definition.
    """

    # import physical constants
    G = constants.G
    H0 = constants.H0 / 1000.0  # km/s/kpc
    rho_crit = 3 * H0**2 / (8 * np.pi * G)  # Msun/kpc^3

    halo = halo.lower().strip()

    # ensure input radii are array
    radii = np.asarray(radii, dtype=float)

    # avoid division by zero at r = 0
    radii_safe = np.where(radii == 0, 1e-8, radii)

    if halo == 'iso':
        core_density, core_radius = params['core_density'], params['core_radius']

        # dimensionless geometric term
        geometric_term = 1 - (core_radius / radii_safe) * np.arctan(radii_safe / core_radius)

        # compute the square velocities
        velocity_squared = 4 * np.pi * G * core_density * core_radius**2 * geometric_term

    elif halo == 'burkert':
        core_density, core_radius = params['core_density'], params['core_radius']

        # normalise the radii for the scale radius
        x = radii_safe / core_radius

        # dimensionless geometric term
        geometric_term = 0.5 * np.log(1 + x**2) + np.log(1 + x) - np.arctan(x)

        # compute the square velocities
        velocity_squared = 4 * np.pi * G * core_density * core_radius**3 * geometric_term / radii_safe

    elif halo == 'nfw':
        M200, concentration = params['M200'], params['concentration']

        # compute R200
        R200 = (3 * M200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)

        # compute the scale radius
        scale_radius = R200 / concentration

        # compute the characteristic overdensity factor
        f_c = np.log(1 + concentration) - concentration / (1 + concentration)

        # compute the scale density
        scale_density = (200.0 / 3.0) * rho_crit * concentration**3 / f_c

        # normalise the radii for the scale radius
        x = radii_safe / scale_radius

        # enclosed mass function term
        f_x = np.log(1 + x) - x / (1 + x)

        # compute the square velocities
        velocity_squared = 4 * np.pi * G * scale_density * scale_radius**3 * f_x / radii_safe

    elif halo == 'dc14':
        M200, concentration = params['M200'], params['concentration']

        # compute the DC14 halo parameters
        Mstar = params.get('Mstar', None)
        scale_density, scale_radius, alpha, beta, gamma, R200 = _dc14_params(M200, concentration, Mstar)

        # build the logarithmic grid until R200, which sets the value of M200.
        # We use a logarithmic grid for numerical stability given the large dynamic range in radii
        radii_grid = np.logspace(-4, np.log10(R200), 2000)

        # normalise the radii for the scale radius
        x = radii_grid / scale_radius

        # unnormalized density (eq. 1 of Di Cinto+14)
        rho_grid = scale_density / (x**gamma * (1 + x**alpha) ** ((beta - gamma) / alpha))

        # compute the integral (eq. 5 of Di Cinto+14)
        M_grid = 4 * np.pi * cumulative_trapezoid(rho_grid * radii_grid**2, radii_grid, initial=0)

        # interpolate back to input radii
        M_interp = np.interp(radii_safe, radii_grid, M_grid)

        # compute the square velocities
        velocity_squared = G * M_interp / radii_safe

    else:
        raise ValueError(f"Unknown halo model '{halo}'. Must be 'iso', 'burkert', 'nfw' or 'dc14'.")

    return np.sqrt(np.maximum(velocity_squared, 0))


def cored_halo_params(M200, halo='burkert'):
    """Solve for the core density and radius of a cored halo matching a given M200.

    Finds the (core_density, core_radius) pair of a Burkert or
    pseudo-isothermal halo whose enclosed mass at R200 equals `M200`,
    using an empirical core-radius-density scaling relation together with
    a root-finding step on the core radius.

    Parameters
    ----------
    M200 : float
        Halo mass within R200, in Msun.
    halo : {'burkert', 'iso'}, default 'burkert'
        Halo density profile.

    Returns
    -------
    rho0_solution : float
        Core density, in Msun/kpc^3.
    r0_solution : float
        Core radius, in kpc.
    R200 : float
        Virial radius, in kpc.

    Raises
    ------
    ValueError
        If `halo` is not ``'burkert'`` or ``'iso'``.
    """

    # import physical constants
    G = constants.G
    H0 = constants.H0 / 1000.0  # km/s/kpc
    rho_crit = 3 * H0**2 / (8 * np.pi * G)  # Msun/kpc^3

    # virial radius from M200 (same convention as v_nfw / _dc14_params)
    R200 = (3 * M200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)

    # scaling relation: rho0 in Msun/pc^3 -> Msun/kpc^3 (x1e9)
    def get_rho0(r0):
        """Empirical core density, in Msun/kpc^3, for a given core radius `r0` (kpc)."""
        return 4.5e-2 * 1e9 * r0 ** (-2.0 / 3.0)

    if halo == 'burkert':

        def mass_enclosed(rho0, r0, r):
            """Enclosed mass of a Burkert halo of core density `rho0` and radius `r0` at `r`."""
            x = r / r0
            geometric_term = 0.5 * np.log(1 + x**2) + np.log(1 + x) - np.arctan(x)
            return 4 * np.pi * rho0 * r0**3 * geometric_term

    elif halo == 'iso':

        def mass_enclosed(rho0, r0, r):
            """Enclosed mass at `r` of a pseudo-isothermal halo (density `rho0`, radius `r0`)."""
            x = r / r0
            geometric_term = 1 - (1.0 / x) * np.arctan(x)
            return 4 * np.pi * rho0 * r0**2 * r * geometric_term

    else:
        raise ValueError("halo must be 'burkert' or 'iso'")

    def residual(r0):
        """Difference between the Burkert/iso enclosed mass at R200 and the target M200.

        Parameters
        ----------
        r0 : float
            Trial core radius, in kpc.

        Returns
        -------
        float
            ``mass_enclosed(get_rho0(r0), r0, R200) - M200``, used as the
            root-finding residual for `scipy.optimize.brentq`.
        """
        rho0 = get_rho0(r0)
        return mass_enclosed(rho0, r0, R200) - M200

    # bracket the root; widen if needed for extreme masses
    r0_solution = brentq(residual, 1e-3, 500.0)
    rho0_solution = get_rho0(r0_solution)

    return rho0_solution, r0_solution, R200


def fit_dark_matter(radii, *, vobs, vstars, vgas, vobs_err=None, halo='iso', fit_mass_to_light=False, Mstar=None):
    """Fit a dark matter halo model to an observed rotation curve.

    Decomposes the observed rotation curve into stellar, gas, and dark
    matter contributions added in quadrature, and fits the dark matter
    halo parameters (and, optionally, the stellar mass-to-light ratio) by
    least squares.

    Parameters
    ----------
    radii : array_like
        Galactocentric radii, in kpc.
    vobs : array_like
        Observed rotation velocity at `radii`, in km/s.
    vstars : array_like
        Stellar contribution to the rotation velocity at `radii`, in km/s
        (for a mass-to-light ratio of 1).
    vgas : array_like
        Gas contribution to the rotation velocity at `radii`, in km/s.
    vobs_err : array_like, optional
        1-sigma uncertainty on `vobs`; defaults to 1 for every point.
    halo : {'iso', 'nfw', 'burkert', 'dc14'}, default 'iso'
        Dark matter halo density profile to fit.
    fit_mass_to_light : bool, default False
        If True, also fit a multiplicative stellar mass-to-light ratio.
    Mstar : float, optional
        Stellar mass, in Msun, used by the ``'dc14'`` halo model (see
        `v_dc14`).

    Returns
    -------
    dict
        Dictionary with keys ``'parameters'`` (best-fit halo parameters,
        and mass-to-light ratio if fitted, in linear units),
        ``'errors'`` (1-sigma parameter uncertainties),
        ``'chi2'``, ``'dof'``, ``'aic'``, ``'bic'``,
        ``'covariance'`` (parameter covariance matrix), and
        ``'ls_output'`` (the raw `scipy.optimize.least_squares` result).

    Raises
    ------
    ValueError
        If `halo` is not a recognized model, or if no valid (finite) data
        points remain after masking.

    Notes
    -----
    For the ``'nfw'`` and ``'dc14'`` halos, the mass and concentration
    are fitted in log10 space, and a Gaussian prior on the concentration
    is applied using the Dutton & Maccio (2014) mass-concentration
    relation (scatter of 0.15 dex).
    """

    # import physical constants
    G = constants.G
    H0 = constants.H0 / 1000.0  # km/s/kpc

    # internal function to compute the residual
    def residuals(fit_params):
        """Residuals (and, for NFW/DC14, concentration prior) of the rotation-curve fit.

        Parameters
        ----------
        fit_params : array_like
            Trial parameter vector: ``(params_1, params_2)`` or
            ``(params_1, params_2, mass_to_light_ratio)`` if
            `fit_mass_to_light` is True, where `params_1`/`params_2` are the
            halo's two free parameters (linear for iso/burkert, log10 for
            nfw/dc14).

        Returns
        -------
        ndarray
            Uncertainty-weighted residuals between the model and `vobs`,
            with a concentration-prior residual appended for the 'nfw' and
            'dc14' halos.
        """
        # store the best-fit parameters
        if fit_mass_to_light:
            params_1, params_2, mass_to_light_ratio = fit_params
        else:
            params_1, params_2 = fit_params
            mass_to_light_ratio = 1.0

        if halo in ('iso', 'burkert'):
            halo_kwargs = dict(core_density=params_1, core_radius=params_2)
        else:  # nfw, dc14
            # fit M200/concentration in log space to remove scaling asymmetry
            halo_kwargs = dict(M200=10.0**params_1, concentration=10.0**params_2)
            if halo == 'dc14':
                halo_kwargs['Mstar'] = Mstar

        # compute the circular velocity
        vhalo = circular_velocity_dm(radii, halo, **halo_kwargs)

        # build the final model rotation curve
        vmodel = np.sqrt(mass_to_light_ratio * vstars**2 + vgas**2 + vhalo**2)

        # compute the residuals
        res = (vmodel - vobs) / vobs_err

        if halo in ['nfw', 'dc14']:
            # use concentration-mass relation for Gaussian penalty on c
            logc_expected = 0.905 - 0.101 * (params_1 - 12.0 - np.log10(H0))
            sigma_c = 0.15  # scatter in the relation of 0.15 dex

            # compute the prior
            prior_c = (params_2 - logc_expected) / sigma_c

            return np.concatenate([res, [prior_c]])

        return res

    # ensure input halo is valid
    if halo not in ['iso', 'nfw', 'burkert', 'dc14']:
        raise ValueError(
            f"Input halo model '{halo}' is invalid. Please choose between "
            "'iso', 'nfw', 'burkert' or 'dc14'."
        )

    # ensure inputs are arrays
    radii = np.asarray(radii)
    vobs = np.asarray(vobs)
    vstars = np.asarray(vstars)
    vgas = np.asarray(vgas)

    # initialise the errors if not given
    if vobs_err is None:
        vobs_err = np.ones(vobs.shape)

    # clip errors for numerical stability
    vobs_err = np.clip(vobs_err, 1e-6, None)

    # mask invalid points
    mask = (
        (np.isfinite(radii))
        & (np.isfinite(vobs))
        & (np.isfinite(vgas))
        & (np.isfinite(vstars))
        & (np.isfinite(vobs_err))
    )
    radii = radii[mask]
    vobs = vobs[mask]
    vgas = vgas[mask]
    vstars = vstars[mask]
    vobs_err = vobs_err[mask]

    # if no valid points
    if np.all(~mask):
        raise ValueError('No points left for fit. Check input arrays for nan and inf.')

    # compute data-driven properties
    vmax = np.nanmax(vobs)  # peak velocity
    rmax = radii[np.argmax(vobs)]  # radius at peak velocity
    if halo in ['iso', 'burkert']:
        rc_guess = rmax / 2.0  # the core radius is at about half the radius of the peak velocity
        rho0_guess = vmax**2 / (4 * np.pi * G * rc_guess**2)  # density of the core from the peak velocity
    elif halo in ['nfw', 'dc14']:
        M200_guess = vmax**3 / (10 * G * H0)  # assuming vmax=v200
        c_guess = 10.0 ** (0.905 - 0.101 * np.log10(M200_guess / (H0 * 10.0**12)))  # from Dutton+14

        # convert to log scale
        logM200_guess = np.log10(M200_guess)
        logc_guess = np.log10(c_guess)

    # define the initial conditions
    if halo in ['iso', 'burkert']:
        initial_guess = [rho0_guess, rc_guess]  # [core density, core radius]
    elif halo in ['nfw', 'dc14']:
        initial_guess = [logM200_guess, logc_guess]  # [Log(M200), Log(concentration)]

    # define the boundaries
    if halo in ['iso', 'burkert']:
        bounds = [
            [rho0_guess / 100, rc_guess / 10],  # min values
            [rho0_guess * 100, rc_guess * 10],
        ]  # max values
    elif halo in ['nfw', 'dc14']:
        bounds = [
            [logM200_guess - 5, 0.0],  # min values
            [logM200_guess + 5, np.log10(40.0)],
        ]  # max values

    # if also the mass-to-light ratio is to be fitted
    if fit_mass_to_light:
        # initial condition
        initial_guess += 0.5

        # lower boundary
        bounds[0] += 0.01

        # upper boundary
        bounds[1] += 5.0

    # do the fit
    result = least_squares(residuals, initial_guess, bounds=bounds)

    # compute degrees of freedom
    dof = len(result.fun) - len(result.x)

    # compute the chi^2
    chi2 = np.sum(result.fun**2)

    # compute the reduced chi^2
    chi2_red = chi2 / dof

    # compute the fit statistics
    dof = np.maximum(len(result.fun) - len(result.x), 1)
    chi2 = np.sum(result.fun[: len(vobs)] ** 2)
    chi2_red = chi2 / dof

    # estimate covariance matrix
    jac = result.jac
    if jac is not None:
        try:
            cov = np.linalg.pinv(jac.T @ jac) * chi2_red
        except np.linalg.LinAlgError:
            cov = None

    # ensure the covariance matrix is reliable
    if np.linalg.cond(jac.T @ jac) > 1e12:
        warnings.warn('Covariance condition number too high. Errors might be unreliable.')

    # estimate the errors
    errors = np.sqrt(np.diag(cov)) if cov is not None else np.zeros(len(initial_guess))

    # revert to linear space
    if halo in ['nfw', 'dc14']:
        errors *= np.log(10) * 10.0**result.x
        result.x = 10.0**result.x

    # compute AIC and BIC
    k = len(result.x)
    n = len(vobs)
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)

    # build the output dictionary
    output = {
        'parameters': result.x,
        'errors': errors,
        'chi2': chi2,
        'dof': dof,
        'aic': aic,
        'bic': bic,
        'covariance': cov,
        'ls_output': result,
    }

    return output


def sdens_dm(radii, halo, z_max_factor=500.0, **params):
    """Surface density radial profile of a dark matter halo.

    'iso' and 'nfw' use the exact analytic Abel projection; 'burkert' and
    'dc14' have no closed form and are projected by numerical
    line-of-sight integration (one `scipy.integrate.quad` call per radius).

    Parameters
    ----------
    radii : array_like
        Projected (2D) galactocentric radii, in kpc.
    halo : {'iso', 'burkert', 'nfw', 'dc14'}
        Dark matter halo density profile.
    z_max_factor : float, default 500.0
        Vertical integration limit, as a multiple of the halo's core/scale
        radius. Only used for 'burkert' and 'dc14'.
    **params
        Halo-specific parameters (see `circular_velocity_dm`).

    Returns
    -------
    ndarray
        Projected surface density, in Msun/kpc^2.
    """

    # import physical constants
    G = constants.G
    H0 = constants.H0 / 1000.0  # km/s/kpc
    rho_crit = 3 * H0**2 / (8 * np.pi * G)  # Msun/kpc^3

    halo = halo.lower().strip()

    # ensure input radii are array
    radii = np.asarray(radii, dtype=float)

    # avoid division by zero at r = 0
    radii_safe = np.where(radii == 0, 1e-8, radii)

    if halo == 'iso':
        # exact analytic Abel projection of rho0/(1+(r/rc)^2)
        # sdens(R) = rho0 * rc * pi / sqrt(1 + (R/rc)^2)  ... derived via substitution z=rc*tan(theta)
        # full derivation: integral_-inf^inf rho0/(1+(R^2+z^2)/rc^2) dz = pi*rho0*rc/sqrt(1+(R/rc)^2)
        core_density, core_radius = params['core_density'], params['core_radius']

        # compute the mass surface density
        sdens = np.pi * core_density * core_radius / np.sqrt(1.0 + (radii_safe / core_radius) ** 2)

    elif halo == 'nfw':
        M200, concentration = params['M200'], params['concentration']

        # compute R200
        R200 = (3 * M200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)

        # compute the scale radius
        scale_radius = R200 / concentration

        # compute the characteristic overdensity factor
        f_c = np.log(1 + concentration) - concentration / (1 + concentration)

        # compute the scale density
        scale_density = (200.0 / 3.0) * rho_crit * concentration**3 / f_c
        x = radii_safe / scale_radius

        # analytic kernel f(x) from Wright & Brainerd (2000), Eq. 11
        f = np.where(
            x < 1,
            1.0 / (x**2 - 1) * (1 - 2.0 / np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))),
            np.where(
                x > 1,
                1.0 / (x**2 - 1) * (1 - 2.0 / np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x - 1) / (x + 1)))),
                1.0 / 3.0,  # x == 1 limit
            ),
        )

        # compute the mass surface density
        sdens = 2.0 * scale_density * scale_radius * f

    elif halo in ('burkert', 'dc14'):
        if halo == 'burkert':
            core_density, core_radius = params['core_density'], params['core_radius']

            # maximum vertical integration distance
            z_max = z_max_factor * core_radius

            def rho_3d(r):
                """Volume density at 3D radius `r`.

                Parameters
                ----------
                r : float or ndarray
                    3D galactocentric radius, in kpc.

                Returns
                -------
                float or ndarray
                    Volume density, in the same (unnormalized scale-density) units
                    as `scale_density` in the enclosing scope.
                """
                x = r / core_radius
                return core_density / ((1.0 + x) * (1.0 + x**2))
        else:  # dc14
            M200, concentration = params['M200'], params['concentration']

            # compute the DC14 halo parameters
            Mstar = params.get('Mstar', None)
            scale_density, scale_radius, alpha, beta, gamma, _ = _dc14_params(M200, concentration, Mstar)
            
            # maximum vertical integration distance
            z_max = z_max_factor * scale_radius

            def rho_3d(r):
                """Volume density at 3D radius `r`.

                Parameters
                ----------
                r : float or ndarray
                    3D galactocentric radius, in kpc.

                Returns
                -------
                float or ndarray
                    Volume density, in the same (unnormalized scale-density) units
                    as `scale_density` in the enclosing scope.
                """
                r = np.maximum(r, 1e-8)
                x = r / scale_radius
                return scale_density / (x**gamma * (1 + x**alpha) ** ((beta - gamma) / alpha))

        def integrand(z, R_val):
            """Volume density along the line of sight at projected radius `R_val`.

            Parameters
            ----------
            z : float
                Line-of-sight (vertical) offset, in kpc.
            R_val : float
                Projected (2D) radius, in kpc.

            Returns
            -------
            float
                Volume density at 3D radius ``sqrt(R_val**2 + z**2)``, in Msun/kpc^3.
            """
            return rho_3d(np.sqrt(R_val**2 + z**2))

        # compute the mass surface density
        sdens = np.array([2.0 * quad(integrand, 0, z_max, args=(Ri,), limit=200)[0] for Ri in radii])

    else:
        raise ValueError(f"Unknown halo model '{halo}'. Must be 'iso', 'burkert', 'nfw' or 'dc14'.")

    return sdens


def rho_dm(radii, halo, **params):
    """Volume density radial profile of a dark matter halo.

    Parameters
    ----------
    radii : array_like
        Galactocentric radii, in kpc.
    halo : {'iso', 'burkert', 'nfw', 'dc14'}
        Dark matter halo density profile.
    **params
        Halo-specific parameters (see `circular_velocity_dm`).

    Returns
    -------
    ndarray
        Volume density, in Msun/kpc^3.
    """

    # import physical constants
    G = constants.G
    H0 = constants.H0 / 1000.0  # km/s/kpc
    rho_crit = 3 * H0**2 / (8 * np.pi * G)  # Msun/kpc^3

    halo = halo.lower().strip()

    # ensure input radii are array
    radii = np.asarray(radii, dtype=float)

    # avoid division by zero at r = 0
    radii_safe = np.where(radii == 0, 1e-8, radii)

    if halo == 'iso':
        core_density, core_radius = params['core_density'], params['core_radius']

        # compute the density
        rho = core_density / (1.0 + (radii / core_radius) ** 2)

    elif halo == 'burkert':
        core_density, core_radius = params['core_density'], params['core_radius']

        # normalise the radii for the core radius
        x = radii / core_radius

        # compute the density
        rho = core_density / ((1.0 + x) * (1.0 + x**2))

    elif halo == 'nfw':
        M200, concentration = params['M200'], params['concentration']

        # compute R200
        R200 = (3 * M200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)

        # compute the scale radius
        scale_radius = R200 / concentration

        # compute the characteristic overdensity factor
        f_c = np.log(1 + concentration) - concentration / (1 + concentration)

        # compute the scale density
        scale_density = (200.0 / 3.0) * rho_crit * concentration**3 / f_c

        # normalise the radii for the scale radius
        x = radii_safe / scale_radius

        # compute the density
        rho = scale_density / (x * (1 + x) ** 2)

    elif halo == 'dc14':
        M200, concentration = params['M200'], params['concentration']

        # compute the DC14 halo parameters
        Mstar = params.get('Mstar', None)
        scale_density, scale_radius, alpha, beta, gamma, _ = _dc14_params(M200, concentration, Mstar)

        # normalise the radii for the scale radius
        x = radii_safe / scale_radius

        # compute the density
        rho = scale_density / (x**gamma * (1 + x**alpha) ** ((beta - gamma) / alpha))

    else:
        raise ValueError(f"Unknown halo model '{halo}'. Must be 'iso', 'burkert', 'nfw' or 'dc14'.")

    return rho


def rotation_curve(radii, sdens, *,
                   scale_height=0, z_layer='exp',
                   sigma_r=0, sigma_phi=0,
                   sdens_err=None, scale_height_err=None, sigma_r_err=None, sigma_phi_err=None,
                   grid_points=2000, extend_radii=True, extend_factor=10, points_for_fit=5, extend_model=None, extend_model_args=None,
                   mc_iter=1, show_progress_bar=True,
                   debug=False, debug_output=None):
    """Compute the circular velocity of a disc, with optional Monte Carlo error propagation.

    Public wrapper around `circular_velocity` that normalizes the input
    shapes and, if `mc_iter > 1`, propagates input uncertainties by
    repeated random sampling.

    Parameters
    ----------
    radii : array_like
        Galactocentric radii, in kpc.
    sdens : array_like
        Surface density at `radii`, in Msun/kpc^2.
    scale_height : float or array_like, default 0
        Vertical scale height at `radii`, in kpc.
    z_layer : {'exp', 'sech', 'sech2', 'gauss'}, default 'exp'
        Vertical density profile shape.
    sigma_r, sigma_phi : float or array_like, default 0
        Radial and azimuthal velocity dispersion at `radii`, in km/s.
    sdens_err, scale_height_err, sigma_r_err, sigma_phi_err : array_like, optional
        1-sigma uncertainties on the corresponding inputs, used only if
        `mc_iter > 1`.
    grid_points : int, default 2000
        Number of points in the internal logarithmic radial grid.
    extend_radii : bool, default True
        Whether to extrapolate the surface density beyond ``max(radii)``.
    extend_factor : float, default 10
        Factor by which the radial grid is extended if `extend_radii` is True;
        the default of 10 has been found to recover the rotation curve
        of THINGS galaxies (de Blok et al. 2008).
    points_for_fit : int, default 5
        Number of outermost points used for the power-law extrapolation.
    extend_model : callable, optional
        Model used to extrapolate the surface density beyond
        ``max(radii)``.
    extend_model_args : tuple, optional
        Positional arguments for `extend_model`.
    mc_iter : int, default 1
        Number of Monte Carlo realizations. If 1, no error propagation is
        performed and a single velocity curve is returned.
    show_progress_bar : bool, default True
        Whether to display a `tqdm` progress bar (used only if
        `mc_iter > 1`).
    debug : bool, default False
        If True, save a diagnostic plot of the surface density/scale
        height extrapolation (only on the first Monte Carlo iteration).
    debug_output : str, optional
        Path the debug plot is saved to.

    Returns
    -------
    ndarray
        Circular velocity, in km/s, if `mc_iter == 1`.
    mean : ndarray
        Mean circular velocity across Monte Carlo realizations, in km/s,
        if `mc_iter > 1`.
    std : ndarray
        Standard deviation of the circular velocity across Monte Carlo
        realizations, in km/s, if `mc_iter > 1`.
    """

    # extend_model is a callable of the form fun(radii, *args, **kwargs)
    # extend_factor=10 is a brute-force extension that has proven to recover the
    # rotation curve of THINGS galaxies (de Blok+08)

    # ensure inputs are arrays
    radii = np.asarray(radii)
    sdens = np.asarray(sdens)
    scale_height = (np.asarray([scale_height]) if isinstance(scale_height, (int, float)) else np.asarray(scale_height))
    sigma_r = np.asarray([sigma_r]) if isinstance(sigma_r, (int, float)) else np.asarray(sigma_r)
    sigma_phi = (np.asarray([sigma_phi]) if isinstance(sigma_phi, (int, float)) else np.asarray(sigma_phi))

    # extend the inputs assuming they stay constant
    scale_height = extend(scale_height, len(radii))
    sigma_r = extend(sigma_r, len(radii))
    sigma_phi = extend(sigma_phi, len(radii))

    # build the kwargs variable to pass to the circular velocity computer
    kwargs = {
        'scale_height': scale_height,
        'z_layer': z_layer,
        'sigma_r': sigma_r,
        'sigma_phi': sigma_phi,
        'grid_points': grid_points,
        'extend_radii': extend_radii,
        'extend_factor': extend_factor,
        'points_for_fit': points_for_fit,
        'extend_model': extend_model,
        'extend_model_args': extend_model_args,
    }

    # if no mc, simply return the circular velocity
    if mc_iter == 1:
        return circular_velocity(radii, sdens, debug=debug, debug_output=debug_output, **kwargs)
    else:
        # initialise the list where the circular velocities will be stored
        v_samples = []

        # compute the surface density samples.
        # We use log sampling do to the high dynamic range
        if sdens_err is not None:
            sdens_samples = np.exp(np.random.normal(np.log(sdens)[None, :],
                                                    (sdens_err / sdens)[None, :],
                                                    size=(mc_iter, len(sdens))))
        else:
            sdens_samples = np.repeat(sdens[None, :], mc_iter, axis=0)

        # compute the scale height samples
        if scale_height_err is not None:
            scale_height_samples = np.random.normal(scale_height[None, :],
                                                    scale_height_err[None, :],
                                                    size=(mc_iter, len(scale_height)))
        else:
            scale_height_samples = np.repeat(scale_height[None, :], mc_iter, axis=0)

        # compute the radial velocity dispersion samples
        if sigma_r_err is not None:
            sigma_r_samples = np.random.normal(sigma_r[None, :], sigma_r_err[None, :], size=(mc_iter, len(sigma_r)))
        else:
            sigma_r_samples = np.repeat(sigma_r[None, :], mc_iter, axis=0)

        # compute the azimuthal velocity dispersion samples
        if sigma_phi_err is not None:
            sigma_phi_samples = np.random.normal(sigma_phi[None, :], sigma_phi_err[None, :], size=(mc_iter, len(sigma_phi)))
        else:
            sigma_phi_samples = np.repeat(sigma_phi[None, :], mc_iter, axis=0)

        # enforce physical constraints
        sdens_samples = np.maximum(sdens_samples, 1e-10)
        scale_height_samples = np.maximum(scale_height_samples, 0)
        sigma_r_samples = np.maximum(sigma_r_samples, 0)
        sigma_phi_samples = np.maximum(sigma_phi_samples, 0)

        # perform the mc simulations
        for i in tqdm(
            range(mc_iter), total=mc_iter, desc='Simulations done', disable=not show_progress_bar
        ):
            # update the kwargs
            kwargs['scale_height'] = scale_height_samples[i]
            kwargs['sigma_r'] = sigma_r_samples[i]
            kwargs['sigma_phi'] = sigma_phi_samples[i]

            # store the circular velocity
            v_samples.append(circular_velocity(radii, sdens_samples[i],
                                                debug=debug, debug_output=debug_output,
                                                **kwargs))

            # do the debug plot only on the first iteration
            debug = False

        return np.nanmean(v_samples, axis=0), np.nanstd(v_samples, axis=0)


def vertical_structure(radii, components, *,
                       sdens=[None], sigma=[None],
                       z_layer=['gauss'], z0=[None],
                       solve=[True], cgm_confined=[False],
                       halo_model=None, halo_params=None,
                       extend_R=True, extend_R_factor=10, extend_models=[None], extend_models_args=[None], extend_fit_points=[5],
                       extend_z=True, extend_z_factor=5,
                       grid_points_R=2000, grid_points_z=200,
                       tol=1e-3, max_iter=50,
                       plot=False, plot_name=None, plot_show=False,
                       verbose=False,
                       **kwargs):
    """Self-consistently solve the vertical structure of a multi-component disc galaxy.

    Iteratively solves for the vertical scale height of one or more disc
    components (stars, gas, ...) embedded in the combined gravitational
    potential of all components plus an optional dark matter halo
    (and, optionally, confined by circumgalactic medium pressure), given
    their surface density and velocity dispersion profiles.

    Parameters
    ----------
    radii : array_like
        Radii at which the input profiles are sampled, in kpc.
    components : list of str
        Names of the disc components to model (e.g. ``['stars', 'gas']``).
    sdens : list of array_like, default [None]
        Surface density profile per component, in Msun/kpc^2. Must have
        one entry per component; a component with ``None`` is skipped.
    sigma : list of array_like, default [None]
        Vertical velocity dispersion profile per component, in km/s.
    z_layer : list of str, default ['gauss']
        Vertical density profile shape per component (one of ``'exp'``,
        ``'sech'``, ``'sech2'``, ``'gauss'``); extended to match
        `components` if shorter.
    z0 : list of array_like, default [None]
        Initial scale-height guess per component, in kpc; computed from
        self-gravity if not given.
    solve : list of bool, default [True]
        Whether each component's scale height is solved for iteratively;
        extended to match `components` if shorter.
    cgm_confined : list of bool, default [False]
        Whether each component is also constrained by CGM pressure
        confinement; extended to match `components` if shorter.
    halo_model : {'iso', 'burkert', 'nfw', 'dc14'}, optional
        Dark matter halo density profile; no halo is included if None.
    halo_params : dict, optional
        Parameters of the chosen halo model; required if `halo_model` is
        given.
    extend_R : bool, default True
        Whether to extend the radial grid beyond ``max(radii)``.
    extend_R_factor : float, default 10
        Factor by which the radial grid is extended if `extend_R` is True.
    extend_models : list of callable, default [None]
        Per-component model used to extrapolate the surface density
        beyond ``max(radii)``, passed to `_build_sdens_grid`.
    extend_models_args : list of tuple, default [None]
        Positional arguments for each entry of `extend_models`.
    extend_fit_points : list of int, default [5]
        Number of outermost points used for the power-law extrapolation,
        for components without an explicit `extend_models` entry.
    extend_z : bool, default True
        Whether to extend the vertical grid beyond the current maximum
        scale height each iteration.
    extend_z_factor : float, default 5
        Factor by which the vertical grid is extended if `extend_z` is True.
    grid_points_R, grid_points_z : int, default 2000, 200
        Number of points in the internal logarithmic radial and vertical
        grids.
    tol : float, default 1e-3
        Convergence tolerance on the maximum fractional change in scale
        height between iterations.
    max_iter : int, default 50
        Maximum number of self-consistency iterations.
    plot : bool, default False
        If True, produce diagnostic plots via `_plot_vertical_structure`.
    plot_name : str, optional
        Base path used to save diagnostic plots, used only if `plot` is
        True.
    plot_show : bool, default False
        If True (and `plot` is True), display the diagnostic plots
        instead of closing them.
    verbose : bool, default False
        If True, print convergence progress.
    **kwargs
        Additional keyword arguments forwarded to `_cgm_pressure` for
        CGM-confined components.

    Returns
    -------
    dict
        Dictionary with keys ``'radii'``, ``'radii_grid'``, ``'z_grid'``,
        ``'components'`` (list of component names, including ``'DM'`` if
        a halo was given), and, for each component name ``<name>``:
        ``'sdens_<name>'``, ``'sdens_grid_<name>'``, ``'sigma_<name>'``,
        ``'z0_<name>'``, ``'z0_anal_<name>'`` (pure self-gravity
        solution), ``'z0_dm_<name>'`` (dark-matter-only solution),
        ``'rho0_<name>'``, ``'n0_<name>'``, ``'rho_<name>'`` (3D volume
        density), and ``'Kz_2d_<name>'`` (2D vertical force); the halo
        component additionally provides ``'Kz_DM'``.
    """

    # import physical constants
    G = constants.G

    # ensure input lists have all the same length
    target_size = len(components)
    z_layer = extend(z_layer, target_size)
    solve = extend(solve, target_size)
    cgm_confined = extend(cgm_confined, target_size)
    sdens = sdens[:target_size]
    sigma = sigma[:target_size]
    z0 = z0[:target_size]

    # validate the galaxy components
    components = [
        _validate_disc_component(radii, sdens[c], sigma[c],
                                 z0[c], z_layer[c],
                                 component,
                                 solve=solve[c], cgm_confined=cgm_confined[c])
        for c, component in enumerate(components)
    ]

    # filter invalid components
    components = [c for c in components if c is not None]

    # validate the dark matter halo
    if halo_params is not None and halo_model is not None:
        halo_model = halo_model.lower().strip()
        if halo_model not in ['iso', 'burkert', 'nfw', 'dc14']:
            raise ValueError(
                f"Unknown halo model '{halo_model}'. Must be 'iso', 'burkert', 'nfw' or 'dc14'."
            )

    # midplane definitions
    z0_target = {
        'gauss': -0.5,
        'exp': -1.0,
        'sech': -np.log(np.cosh(0.5)),
        'sech2': -2.0 * np.log(np.cosh(0.5)),
    }

    # clip small radii for numerical stabilities
    radii = np.maximum(1e-3, radii)

    # create the logarithmic radial grid
    rmin = np.nanmin(radii)
    rmax = np.nanmax(radii)
    if extend_R:
        rmax *= extend_R_factor
    radii_grid = np.logspace(np.log10(rmin), np.log10(rmax), grid_points_R)

    # interpolate the surface density of each component onto the log grid
    for c, component in enumerate(components):
        model = extend_models[c]
        if not isinstance(model, type(None)):
            args = extend_models_args[c]
            points_to_fit = None
        else:
            args = None
            points_to_fit = extend_fit_points[c]

        component['sdens_grid'] = _build_sdens_grid(radii, component['sdens'], radii_grid,
                                                    extend=extend_R, points_to_fit=points_to_fit,
                                                    extend_model=model, extend_model_args=args)

        # compute the CGM pressure if the component is affected by the CGM
        if component.get('cgm_confined', False):
            # component['sigma_grid'] = np.interp(radii_grid, radii, component['sigma'])
            component['P_cgm'] = _cgm_pressure(radii, **kwargs)

    # Self-consistency loop. We first compute the vertical force with the initial
    # scale height z0(R). From the vertical force and the velocity dispersion we
    # compute the density, then use the resulting density distribution to get the
    # new z0(R). We repeat, feeding back the new z0(R) each time, until z0_old(R)
    # and z0_new(R) converge.
    for iteration in range(max_iter):
        # build the logarithmic vertical grid
        zmin = 1e-4
        zmax = max(np.nanmax(c['z0']) for c in components)
        if extend_z:  # extend input z to account for mass at larger z
            zmax *= extend_z_factor
        z_grid = np.logspace(np.log10(zmin), np.log10(zmax), grid_points_z)

        for c in components:
            # snapshot z0 for convergence checking before update
            if c['solve']:
                c['z0_old'] = c['z0'].copy()

            # mask non-finite values
            valid = np.isfinite(c['z0'])

            # interpolate scale height onto log grid using a constant value
            interp = interp1d(radii[valid], c['z0'][valid],
                              kind='linear', bounds_error=False,
                              fill_value=c['z0'][valid][-1], assume_sorted=False)
            c['z0_grid'] = interp(radii_grid)

        # pre-compute all vertical force components (and total) on the 2D grid
        shape = (len(z_grid), len(radii))
        Kz_tot = np.zeros(shape)
        Kz_per_component = {}

        for c in components:
            Kz = _vertical_force_disc(radii, radii_grid, z_grid, c['sdens_grid'], c['z0_grid'], c['z_layer'])
            Kz_per_component[c['name']] = Kz
            Kz_tot += Kz

        if halo_params is not None:
            Kz = _vertical_force_dm(radii, z_grid, halo_model, halo_params)
            Kz_per_component['DM'] = Kz
            Kz_tot += Kz

        # update z0 for every component being solved
        change = []
        for c in components:
            if not c['solve']:
                continue

            # integrate d(ln rho)/dz = Kz/sigma^2 for all radii at once.
            # This gives the 3D logarithm of the density: ln(rho(z,R))
            dz = np.diff(z_grid)
            integrand = (Kz_tot[:-1] + Kz_tot[1:]) / c['sigma'][np.newaxis, :] ** 2

            # trapezoidal cumulative integral along z-axis per radius
            ln_rho = np.zeros_like(Kz_tot)
            ln_rho[1:] = np.cumsum(0.5 * integrand * dz[:, np.newaxis], axis=0)  # true trapezoidal

            # derive the new scale height z0
            # z0 is defined as the height at which the vertical profile drops ln(rho(z0)/rho(0))
            target = z0_target[c['z_layer']]
            z0_new = np.empty(len(radii))

            for R in range(len(radii)):
                # get the vertical profile at radius R
                profile = ln_rho[:, R]
                # find z0 (crossing of ln_rho(R) = -0.5)
                crossing = np.where(profile <= target)[0]
                if len(crossing) == 0:
                    # profile never drops to e^{-1/2}: use the last z as lower bound
                    z0_new[R] = z_grid[-1]
                else:
                    # linear interpolation between the two bracketing points
                    j1 = crossing[0]
                    j0 = j1 - 1
                    if j0 < 0:
                        z0_new[R] = z_grid[0]
                    else:
                        # interpolate linearly
                        slope = (profile[j1] - profile[j0]) / (z_grid[j1] - z_grid[j0])
                        z0_new[R] = z_grid[j0] + (target - profile[j0]) / slope

            # store the updated scale height
            c['z0'] = np.clip(z0_new, 0.01, 50.0)

            # compute the CGM pressure
            if c['cgm_confined']:
                # mass-weighted <|Kz_tot|> using the ln_rho shape already computed this iteration
                w = np.exp(ln_rho - np.nanmax(ln_rho, axis=0, keepdims=True))  # avoid overflow
                Kz_abs = np.abs(Kz_tot)
                Kz_mean = np.trapezoid(w * Kz_abs, z_grid, axis=0) / np.trapezoid(w, z_grid, axis=0)  # shape (n_R,)

                norm_layer = {
                    'gauss': np.sqrt(2 * np.pi),
                    'exp': 2.0,
                    'sech': np.pi,
                    'sech2': 4.0,
                }[c['z_layer']]

                z0_cgm = (c['sdens'] * c['sigma'] ** 2) / (norm_layer * (c['P_cgm'] + 0.5 * c['sdens'] * Kz_mean))

                # confinement can only compress the layer relative to the self-gravity+DM solution
                c['z0'] = np.clip(np.minimum(c['z0'], z0_cgm), 0.01, 50.0)

            # compute the change w.r.t. the previous iteration
            change.append(np.max(np.abs(c['z0'] - c['z0_old']) / (c['z0_old'] + 1e-10)))

        max_change = max(change) if change else 0.0
        if verbose:
            print(f"Iteration {iteration+1:2d} - max fractional change in z0: {max_change:.4f}")

        if max_change < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations.")
            break
    else:
        if verbose:
            print(
                f"Warning: did not converge after {max_iter} iterations. "
                f"Final max fractional change: {max_change:.4f}"
            )

    # re-compute the three force components using the final scale-height
    Kz_tot = np.zeros(shape)
    Kz_per_component = {}

    # re-compute the vertical grid using the final scale-height
    for c in components:
        interp = interp1d(radii, c['z0'],
                          kind='linear', bounds_error=False, fill_value=c['z0'][-1], assume_sorted=False)
        c['z0_grid'] = interp(radii_grid)

        Kz = _vertical_force_disc(radii, radii_grid, z_grid, c['sdens_grid'], c['z0_grid'], c['z_layer'])
        Kz_per_component[c['name']] = Kz
        Kz_tot += Kz

    if halo_params is not None:
        Kz = _vertical_force_dm(radii, z_grid, halo_model, halo_params)
        Kz_per_component['DM'] = Kz
        Kz_tot += Kz

    # build the output dictionary
    result = dict(radii=radii, radii_grid=radii_grid, z_grid=z_grid, components=[])

    for c in components:
        name = c['name']
        z0 = c['z0']
        sdens = c['sdens']
        sdens_grid = c['sdens_grid']
        sigma = c['sigma']
        z_layer = c['z_layer']
        Kz_2d = Kz_per_component[name]  # shape (n_z, n_R)

        # volume density
        rho0, n0, rho = volume_density(z_grid, z0, sdens, z_layer)

        # pure self-gravity analytical solution
        z0_anal = sigma**2 / (np.pi * G * sdens)

        # compute the scale height due to dark matter only
        if 'DM' in Kz_per_component:
            # compute the circular velocity
            vc = circular_velocity_dm(radii, halo_model, **halo_params)

            # compute the scale height due to dark matter only
            z0_dm = sigma * radii / vc
        else:
            z0_dm = np.zeros_like(radii)

        result['components'].append(name)
        result[f'sdens_{name}'] = sdens
        result[f'sdens_grid_{name}'] = sdens_grid
        result[f'sigma_{name}'] = sigma
        result[f'z0_{name}'] = z0
        result[f'z0_anal_{name}'] = z0_anal
        result[f'z0_dm_{name}'] = z0_dm
        result[f'rho0_{name}'] = rho0
        result[f'n0_{name}'] = n0
        result[f'rho_{name}'] = rho
        result[f'Kz_2d_{name}'] = Kz_2d

    # dark matter force if present
    if 'DM' in Kz_per_component:
        result['components'].append('DM')
        result['sdens_DM'] = sdens_dm(radii, halo_model, **halo_params)
        result['sdens_grid_DM'] = sdens_dm(radii_grid, halo_model, **halo_params)
        result['sigma_DM'] = np.empty(radii.shape) * np.nan
        result['z0_DM'] = np.empty(radii.shape) * np.nan
        result['z0_anal_DM'] = np.empty(radii.shape) * np.nan
        result['z0_dm_DM'] = np.empty(radii.shape) * np.nan
        result['rho0_DM'] = rho_dm(radii, halo_model, **halo_params)
        result['n0_DM'] = np.empty(radii.shape) * np.nan
        result['rho_DM'] = volume_density_dm(radii, z_grid, halo_model, halo_params)
        result['Kz_2d_DM'] = Kz_per_component['DM']
        result['Kz_DM'] = Kz_per_component['DM'][0, :]

    # plot the solution
    if plot:
        from nicci.plotter.plotter import _plot_vertical_structure

        dpi = kwargs.get('dpi', 300)
        fig1, fig2, fig3 = _plot_vertical_structure(result)
        if plot_name is not None:
            fig1.savefig(plot_name + '_sdens.pdf', dpi=dpi, bbox_inches='tight')
            fig2.savefig(plot_name + '_summary.pdf', dpi=dpi, bbox_inches='tight')
            fig3.savefig(plot_name + '_density.pdf', dpi=dpi, bbox_inches='tight')
        plt.show() if plot_show else plt.close('all')

    return result


def virial_temperature(M200, mu=0.6):
    """Compute the virial temperature of a dark matter halo.

    Parameters
    ----------
    M200 : float
        Halo mass within R200, in Msun.
    mu : float, default 0.6
        Mean molecular weight.

    Returns
    -------
    float
        Virial temperature, in K.
    """

    # import physical constants
    kB = constants.kB
    m_p = constants.Mp
    G = constants.G
    H0 = constants.H0 / 1000.0  # km/s/kpc
    rho_crit = 3 * H0**2 / (8 * np.pi * G)  # Msun/kpc^3

    # virial radius and circular velocity squared at R200
    R200 = (3 * M200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)  # kpc
    V200_sq = G * M200 / R200  # (km/s)^2

    # virial temperature; 1e10 converts (km/s)^2 to (cm/s)^2
    T_vir = mu * m_p * (V200_sq * 1e10) / (2 * kB)

    return T_vir


def volume_density(z_grid, z0, sdens, z_layer):
    """Compute midplane and 3D volume density from a surface density and scale height.

    Parameters
    ----------
    z_grid : ndarray
        Heights above the midplane, in kpc.
    z0 : ndarray
        Vertical scale height at each radius, in kpc.
    sdens : ndarray
        Surface density at each radius, in Msun/kpc^2.
    z_layer : {'exp', 'sech', 'sech2', 'gauss'}
        Vertical density profile shape.

    Returns
    -------
    rho0 : ndarray
        Midplane volume density at each radius, in Msun/kpc^3.
    n0 : ndarray
        Midplane number density at each radius, in cm^-3.
    rho : ndarray, shape (len(z_grid), len(z0))
        3D volume density on the (z, R) grid, in Msun/kpc^3.
    """

    # unit conversion constants
    Msun_to_MH = constants.Msun_to_MH
    kpc_cm = constants.kpc_cm
    rho_to_n = Msun_to_MH / kpc_cm**3  # [cm^-3 / (Msun/kpc^3)]

    # normalisation factors: Sigma = 2 * integral_0^inf rho(z) dz = norm * rho0 * z0
    # so rho0 = Sigma / (norm * z0)
    norm = {'gauss': np.sqrt(2.0 * np.pi), 'exp': 2.0, 'sech': np.pi, 'sech2': 4.0}[z_layer]

    # midplane densities
    rho0 = sdens / (norm * z0)  # [Msun/kpc^3]
    n0 = rho0 * rho_to_n  # [cm^-3]

    # 3D profile on the (z,R) grid
    z_2d = z_grid[:, np.newaxis]  # (n_z,1)
    z0_2d = z0[np.newaxis, :]  # (1, n_R)

    if z_layer == 'gauss':
        rho = rho0 * np.exp(-0.5 * (z_2d / z0_2d) ** 2)
    elif z_layer == 'exp':
        rho = rho0 * np.exp(-np.abs(z_2d) / z0_2d)
    elif z_layer == 'sech':
        rho = rho0 / np.cosh(z_2d / (2.0 * z0_2d))
    elif z_layer == 'sech2':
        rho = rho0 / np.cosh(z_2d / (2.0 * z0_2d)) ** 2

    return rho0, n0, rho


def volume_density_dm(radii, z, halo_model, halo_params):
    """Compute a dark matter halo's 3D volume density on a (z, R) grid from its rotation curve.

    Solves the spherical Poisson equation for the enclosed-mass proxy
    ``Vc^2 * r`` to obtain the radial density profile, then evaluates it
    at the 3D radius ``r = sqrt(R^2 + z^2)`` for every (z, R) pair.

    Parameters
    ----------
    radii : ndarray
        Radii (R) at which the density is evaluated, in kpc.
    z : array_like
        Heights above the midplane at which the density is evaluated,
        in kpc.
    halo_model : {'iso', 'burkert', 'nfw', 'dc14'}
        Dark matter halo density profile.
    halo_params : dict
        Parameters of the chosen halo model (as expected by the
        corresponding ``v_*`` circular-velocity function).

    Returns
    -------
    ndarray, shape (len(z), len(radii))
        3D volume density, in Msun/kpc^3.
    """

    # import physical constants
    G = constants.G

    # compute the circular velocity
    vc = circular_velocity_dm(radii, halo_model, **halo_params)

    # compute the mass proxy (Vc^2 * r proportional to M(<r))
    mass_proxy = vc**2 * radii

    # compute the mass derivative (d(Vc^2 * r)/dr)
    dm = np.gradient(mass_proxy, radii)

    # compute the density rho(R) solving the Poisson equation
    rho = np.maximum(dm / (4.0 * np.pi * G * radii**2), 0.0)

    # interpolate in log-log space for accuracy across dynamic range
    log_rho_interp = interp1d(np.log(radii), np.log(np.maximum(rho, 1e-30)),
                              kind='linear', bounds_error=False, fill_value='extrapolate')

    # evaluate at r = sqrt(R^2 + z^2) for every (z, R) pair
    R_2d = radii[np.newaxis, :]  # (1,n_R)
    z_2d = np.array(z)[:, np.newaxis]  # (n_z,1)
    r_2d = np.sqrt(R_2d**2 + z_2d**2)  # (n_z,n_R)

    rho = np.exp(log_rho_interp(np.log(r_2d)))

    return rho  # shape (n_z,n_R) [Msun/kpc^3]


def _asymmetric_drift(radii, sdens, sigma_r, sigma_phi):
    """Compute the asymmetric drift correction to the circular velocity.

    Parameters
    ----------
    radii : ndarray
        Galactocentric radii, in kpc.
    sdens : ndarray
        Surface density at `radii`, in Msun/kpc^2.
    sigma_r : ndarray
        Radial velocity dispersion at `radii`, in km/s.
    sigma_phi : ndarray or None
        Azimuthal velocity dispersion at `radii`, in km/s. If None or
        identically zero, the epicyclic approximation
        ``sigma_phi = sqrt(0.5) * sigma_r`` is used.

    Returns
    -------
    ndarray
        Asymmetric drift term (``Vc^2 - Vphi^2``), in (km/s)^2, clipped to
        be non-negative.
    """

    if sigma_phi is None or np.all(sigma_phi == 0):
        sigma_phi = np.sqrt(0.5) * sigma_r
    else:
        sigma_phi = extend(sigma_phi, len(radii))
    sigma_ratio = np.divide(sigma_phi**2, sigma_r**2, where=sigma_r > 0)
    dlnSigma = np.gradient(np.log(sdens), np.log(radii))
    dlnSigR2 = np.gradient(np.log(sigma_r**2), np.log(radii))

    correction = sigma_r**2 * (dlnSigma + dlnSigR2 + 1 - sigma_ratio)

    return np.maximum(correction, 0.0)


def _build_sdens_grid(radii, sdens, radii_grid, *,
                      extend_radii=False, points_to_fit=5, extend_model=None, extend_model_args=None):
    """Interpolate (and optionally extrapolate) a surface density profile onto a grid.

    Parameters
    ----------
    radii : ndarray
        Radii at which `sdens` is sampled.
    sdens : ndarray
        Surface density values at `radii`.
    radii_grid : ndarray
        Target radial grid to interpolate/extrapolate onto.
    extend_radii : bool, default False
        If True, radii beyond ``max(radii)`` are extrapolated (via
        `extend_model` if given, otherwise a power law fit to the
        outermost points); otherwise the log-log interpolation's own
        extrapolation is used unchanged.
    points_to_fit : int, default 5
        Number of outermost points used to fit the power-law
        extrapolation, used only if `extend_model` is not given.
    extend_model : callable, optional
        Model of the form ``extend_model(radii, *extend_model_args)``
        used to extrapolate beyond ``max(radii)``.
    extend_model_args : tuple, optional
        Positional arguments for `extend_model`.

    Returns
    -------
    ndarray
        Surface density interpolated (and, if `extend_radii` is True,
        extrapolated) onto `radii_grid`.
    """

    # mask non-finite values
    valid = (np.isfinite(sdens)) & (np.isfinite(radii)) & (sdens > 0) & (radii > 0)
    sdens_safe = sdens[valid]
    radii_safe = radii[valid]

    # interpolate log-log onto the full grid (extrapolation here is temporary)
    interp = interp1d(np.log(radii_safe), np.log(sdens_safe),
                      kind='linear', bounds_error=False, fill_value='extrapolate', assume_sorted=False)
    sdens_grid = np.exp(interp(np.log(radii_grid)))

    # overwrite the extrapolated region if extend is requested
    if extend_radii:
        # mask for extrapolation region
        larger_radii = radii_grid > np.nanmax(radii_safe)

        # if extend with the user-supplied model
        if not isinstance(extend_model, type(None)) and not isinstance(extend_model_args, type(None)):
            sdens_grid[larger_radii] = extend_model(radii_grid[larger_radii], *extend_model_args)

        # extend using power-law extrapolation
        else:
            # fit a power law to the outermost points_for_fit points
            coeffs = np.polyfit(np.log(radii_safe[-points_to_fit:]), np.log(sdens[-points_to_fit:]), 1)
            alpha = max(-coeffs[0], 0.0)  # clamp: no rising extrapolation
            sdens_grid[larger_radii] = (sdens[-1] * (radii_grid[larger_radii] / radii_safe[-1]) ** -alpha)

    return sdens_grid


def _cgm_pressure(radii, *, n0=0.46 * 1.92, rc=0.35, beta=0.43, T_cgm=2e6):
    """Compute the thermal pressure profile of a beta-model hot circumgalactic medium.

    Parameters
    ----------
    radii : array_like
        Galactocentric radii, in kpc.
    n0 : float, default 0.46 * 1.92
        Central electron number density scaled to a total particle number
        density, in cm^-3 (Milky Way value from Miller & Bregman 2013).
    rc : float, default 0.35
        Core radius, in kpc (Milky Way value from Miller & Bregman 2013).
    beta : float, default 0.43
        Beta-model slope (Milky Way-like value from Zhang et al. 2024).
    T_cgm : float, default 2e6
        CGM temperature, in K (Milky Way-like value from Zhang et al. 2024).

    Returns
    -------
    ndarray
        CGM pressure, in Msun (km/s)^2 kpc^-3.
    """
    # n0=0.46 is n_e of the Milky Way from Miller & Bregman (2013), multiplied by
    # 1.92 to convert it into a number density.
    # rc=0.35 is of the Milky Way from Miller & Bregman (2013).
    # beta=0.43 is of MW-like halos from Zhang+24
    # T_cgm=2e6 is average MW-like virial temperature from Zhang+24

    # import physical constants
    kB = constants.kB

    units_conversion = 6.769e-22  # Msun (km/s)^2 kpc^-3 -> erg/cm^3

    n = n0 * (1.0 + (radii / rc) ** 2) ** (-1.5 * beta)  # [cm^-3], 3D density
    P_cgs = n * kB * T_cgm  # [erg/cm^3]
    return P_cgs / units_conversion  # [Msun (km/s)^2 kpc^-3]
    

def _dc14_params(M200, concentration, Mstar=None):
    """Derive the DC14 halo profile parameters for a given mass and concentration.

    Parameters
    ----------
    M200 : float
        Halo mass within R200, in Msun.
    concentration : float
        Halo concentration, ``c = R200 / scale_radius``.
    Mstar : float, optional
        Stellar mass, in Msun; estimated from `M200` via the Girelli et
        al. (2020) relation if not given (or 0).

    Returns
    -------
    scale_density : float
        Normalization of the DC14 density profile such that the enclosed
        mass at R200 equals `M200`.
    scale_radius : float
        DC14 scale radius, in kpc.
    alpha, beta, gamma : float
        DC14 inner, transition, and outer density-profile slopes
        (Di Cintio et al. 2014, Eq. 3).
    R200 : float
        Virial radius, in kpc.
    """

    # import physical constants
    G = constants.G
    H0 = constants.H0 / 1000.0  # km/s/kpc
    rho_crit = 3 * H0**2 / (8 * np.pi * G)  # Msun/kpc^3

    # compute stellar mass from Girelli+20 Mhalo-Mstar relation (see Eq. 6 and Table 1, z=0)
    if Mstar is None or Mstar == 0:
        A = 0.0465
        beta = 1.00
        gamma = 0.702
        M_A = 10.0**11.77
        Mstar = M200 * 2 * A / ((M200 / M_A) ** (-beta) + (M200 / M_A) ** gamma)

    # compute R200
    R200 = (3 * M200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)

    # compute the scale radius
    scale_radius = R200 / concentration

    # define the halo slopes (eq. 3 of Di Cinto+14)
    X = np.log10(np.maximum(Mstar / M200, 1e-12))

    alpha = 2.94 - np.log10((10 ** (X + 2.33)) ** (-1.08) + (10 ** (X + 2.33)) ** (2.29))
    beta = 4.23 + 1.34 * X + 0.26 * X**2
    gamma = -0.06 + np.log10((10 ** (X + 2.56)) ** (-0.68) + 10 ** (X + 2.56))

    # build the logarithmic grid until R200, which sets the value of M200.
    # We use a logarithmic grid for numerical stability given the large dynamic range in radii
    radii_grid = np.logspace(-4, np.log10(R200), 2000)

    # define the normalised radius grid
    x = radii_grid / scale_radius

    # unnormalized density (eq. 1 of Di Cinto+14)
    rho = 1 / (x**gamma * (1 + x**alpha) ** ((beta - gamma) / alpha))

    # compute the integral (eq. 5 of Di Cinto+14)
    M_grid = 4 * np.pi * cumulative_trapezoid(rho * radii_grid**2, radii_grid, initial=0)

    # compute the scale density
    scale_density = M200 / M_grid[-1]

    return scale_density, scale_radius, alpha, beta, gamma, R200


def _radial_force(radii, sdens, *,
                  scale_height, z_layer,
                  grid_points, extend_radii, extend_factor, points_for_fit, extend_model, extend_model_args,
                  debug, debug_output):
    """Compute the radial gravitational force of an axisymmetric disc.

    Interpolates the surface density onto a logarithmic radial grid
    (optionally extrapolating beyond the observed range), applies the
    vertical thickness correction in Fourier space, and uses a pair of
    Hankel transforms (order 0 then order 1) to obtain the radial force.

    Parameters
    ----------
    radii : ndarray
        Radii at which the force is evaluated, in kpc.
    sdens : ndarray
        Surface density at `radii`, in Msun/kpc^2.
    scale_height : ndarray
        Vertical scale height at `radii`, in kpc.
    z_layer : {'exp', 'sech', 'sech2', 'gauss'}
        Vertical density profile shape.
    grid_points : int
        Number of points in the internal logarithmic radial grid.
    extend_radii : bool
        Whether to extrapolate the surface density beyond ``max(radii)``.
    extend_factor : float
        Factor by which the radial grid is extended if `extend_radii` is True.
    points_for_fit : int
        Number of outermost points used for the power-law extrapolation,
        used only if `extend_model` is not given.
    extend_model : callable or None
        Model of the form ``extend_model(radii, *extend_model_args)``
        used to extrapolate beyond ``max(radii)``.
    extend_model_args : tuple or None
        Positional arguments for `extend_model`.
    debug : bool
        If True, save a diagnostic plot comparing the input and
        extrapolated surface density and scale height.
    debug_output : str or None
        Path the debug plot is saved to; defaults to
        ``'rotcurve_debug.pdf'`` in the current working directory.

    Returns
    -------
    ndarray
        Radial force at `radii`, in (km/s)^2/kpc (negative, pointing
        inward, for a physical disc).
    """

    # import physical constants
    G = constants.G

    # determine radii boundaries
    rmin = np.nanmin(radii)
    rmax = np.nanmax(radii)

    # filter for numerical stability
    rmin = np.maximum(rmin, 1e-4)

    # extend the radius to properly take into account the force at radii larger than observed
    if extend_radii:
        rmax *= extend_factor

    # build the logarithmic grid.
    # We use a logarithmic grid for numerical stability given the large dynamic range in radii
    radii_grid = np.logspace(np.log10(rmin), np.log10(rmax), grid_points)

    # interpolate the surface density onto log grid
    interp = interp1d(np.log(radii), np.log(sdens),
                      kind='linear', bounds_error=False, fill_value='extrapolate', assume_sorted=False)
    sdens_grid = np.exp(interp(np.log(radii_grid)))

    # interpolate the scale-height onto log grid assuming it stays constant at larger radii
    interp = interp1d(radii, scale_height,
                      kind='linear', bounds_error=False, fill_value=scale_height[-1], assume_sorted=False)
    z_grid = interp(radii_grid)

    if extend_radii:
        # define the larger radii indeces
        larger_radii = radii_grid > radii[-1]

        # if no model is provided for the extension
        if extend_model is None or extend_model_args is None:
            # The extrapolation to larger radii is sensitive to the shape of the
            # outermost points; since those oscillate close to the noise level,
            # this can lead to incorrect values. We therefore manually extrapolate
            # assuming a power law for the mass surface density.

            # compute the spectral index by doing a linear fit in log space
            coeffs = np.polyfit(np.log(radii[-points_for_fit:]), np.log(sdens[-points_for_fit:]), 1)

            # extract the slope, i.e., -spectral index
            alpha = coeffs[0]

            # extrapolate the mass surface density at larger radii
            sdens_grid[larger_radii] = sdens[-1] * (radii_grid[larger_radii] / radii[-1]) ** alpha
        else:
            sdens_grid[larger_radii] = extend_model(radii_grid[larger_radii], *extend_model_args)

    # do debug plots to check the quality of the extrapolation
    if debug:
        if debug_output is None:
            debug_output = os.getcwd() + '\rotcurve_debug.pdf'
        fig, ax = make_figure(nrows=2, figx=6, figy=4,
                              xlabel='Radius [kpc]', ylabel=[r'$\Sigma$ [M$_\odot$ kpc$^{-2}$]', 'z [kpc]'])
        ax[0].plot(radii, sdens, c='blue', lw=2, label='Input')
        ax[0].plot(radii_grid, sdens_grid, c='deeppink', lw=1, ls='dashed', label='Extrapolation')
        ax[0].set_yscale('log')
        ax[0].legend(prop={'size': 12}, frameon=False)
        ax[1].plot(radii, scale_height, c='blue', label='Input')
        ax[1].plot(radii_grid, z_grid, c='deeppink', lw=1, ls='dashed', label='Extrapolation')
        ax[1].legend(prop={'size': 12}, frameon=False)
        fig.savefig(debug_output, dpi=300, bbox_inches='tight')
        plt.close()

    # Hankel transform (J0 kernel) the densities to the Fourier–Bessel space
    h0 = Hankel(radii_grid, nu=0, q=1)
    k, sdens_k = h0(sdens_grid)

    # apply vertical thickness correction
    sdens_k *= _vertical_correction(k, z_grid, z_layer)

    # compute the radial force
    # This is another Hankel transform of order 1
    h1 = Hankel(k, nu=1, q=1)
    force_radii, force_kernel = h1(sdens_k)

    radial_force = -2.0 * np.pi * G * force_kernel

    # interpolate back to physical radii
    interp = interp1d(force_radii, radial_force, bounds_error=False, fill_value='extrapolate')
    radial_force = interp(radii)

    return radial_force


def _validate_disc_component(radii, sdens, sigma, z0, z_layer, name, solve, cgm_confined):
    """Validate and initialize one disc component for `vertical_structure`.

    Parameters
    ----------
    radii : ndarray
        Radii at which `sdens`/`sigma` are sampled, in kpc.
    sdens : array_like or None
        Surface density profile, in Msun/kpc^2. If None, the component is
        skipped (returns None).
    sigma : array_like
        Vertical velocity dispersion profile, in km/s.
    z0 : array_like or None
        Initial guess for the scale-height profile, in kpc. Computed from
        self-gravity (`sigma**2 / (pi * G * sdens)`) if None or if its
        length does not match `radii`.
    z_layer : {'exp', 'sech', 'sech2', 'gauss'}
        Vertical density profile shape.
    name : str
        Component name, used in error messages and as an output key.
    solve : bool
        Whether this component's scale height is solved for iteratively
        (True) or kept fixed (False).
    cgm_confined : bool
        Whether this component's scale height is also constrained by CGM
        pressure confinement.

    Returns
    -------
    dict or None
        None if `sdens` is None; otherwise a dictionary with keys
        ``'name'``, ``'sdens'``, ``'sigma'``, ``'z0'`` (clipped to
        [0.01, 50] kpc), ``'z_layer'``, ``'solve'``, ``'cgm_confined'``,
        and placeholders ``'sdens_grid'``/``'z0_grid'`` (filled in later).

    Raises
    ------
    ValueError
        If `z_layer` is not a recognized vertical profile, or if `sdens`
        or `sigma` does not match the length of `radii`.
    """

    # import physical constants
    G = constants.G

    if sdens is None:
        return None

    if z_layer not in ['exp', 'sech', 'sech2', 'gauss']:
        raise ValueError(f"Unknown vertical layer '{z_layer}' for {name}.")

    # ensure inputs are arrays
    sdens = np.atleast_1d(np.asarray(sdens, dtype=float))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    z0 = np.atleast_1d(np.asarray(z0, dtype=float))

    # ensure same length and valid values
    for arr, label in [(sdens, 'surface density'), (sigma, 'velocity dispersion')]:
        if len(arr) != len(radii):
            raise ValueError(f"{name} {label} length {len(arr)} != radii length {len(radii)}")

    # ensure input scale-height is valid
    if z0 is None or len(z0) != len(radii):
        # compute the initial stellar/gas scale height, assuming a
        # self-gravitating disc, if not given
        z0 = sigma**2 / (np.pi * G * np.maximum(sdens, 1e-10))

    return dict(
        name=name,
        sdens=sdens,
        sigma=sigma,
        z0=np.clip(z0, 0.01, 50.0),  # enforce physical bounds [kpc]
        z_layer=z_layer,
        solve=solve,  # whether z0 is iterated or fixed
        cgm_confined=cgm_confined,  # if the CGM pressure affect the component
        sdens_grid=None,  # filled later
        z0_grid=None,
    )  # filled later


def _vertical_correction(k, scale_height, profile):
    """Vertical thickness correction factor for a disc's Fourier-space surface density.

    Parameters
    ----------
    k : ndarray
        Radial wavenumbers (conjugate to radius under the Hankel
        transform).
    scale_height : ndarray
        Vertical scale height at each radius sampled by `k`.
    profile : {'exp', 'sech', 'sech2', 'gauss'}
        Vertical density profile shape.

    Returns
    -------
    ndarray
        Multiplicative correction factor, the same shape as `k`, equal to
        1 everywhere if `scale_height` is identically zero (infinitely
        thin disc).
    """

    # if infinitely thin vertical layer
    if np.all(scale_height == 0.0):
        return np.ones_like(k)

    # if exponetially declining vertical layer
    if profile == 'exp':
        return 1.0 / (1.0 + k * scale_height)

    # if sech^2 vertical layer
    elif profile == 'sech2':
        x = np.minimum(np.pi * k * scale_height, 1.4e3)  # ensure numerical stability
        return x / (2 * np.sinh(x / 2))

    # if sech vertical layer
    elif profile == 'sech':
        x = np.pi * k * scale_height
        return 1 / np.cosh(x / 2)

    # if gaussian vertical layer
    elif profile == 'gauss':
        x = k * scale_height / np.sqrt(2.0)
        return erfcx(k * scale_height / np.sqrt(2.0))


def _vertical_force_disc(radii, radii_grid, z_grid, sdens, scale_height, z_layer):
    """Compute the vertical gravitational force of a disc component on a (z, R) grid.

    Uses a zeroth-order Hankel transform of the surface density
    (corrected for finite thickness), applies the vertical damping factor
    ``exp(-k*z)`` at each height, and inverse-transforms back to real
    space.

    Parameters
    ----------
    radii : ndarray
        Radii at which the force is evaluated, in kpc.
    radii_grid : ndarray
        Radial grid the surface density is sampled on (for the Hankel
        transform), in kpc.
    z_grid : ndarray
        Heights above the midplane at which the force is evaluated, in kpc.
    sdens : ndarray
        Surface density sampled on `radii_grid`, in Msun/kpc^2.
    scale_height : ndarray
        Vertical scale height sampled on `radii_grid`, in kpc.
    z_layer : {'exp', 'sech', 'sech2', 'gauss'}
        Vertical density profile shape.

    Returns
    -------
    ndarray, shape (len(z_grid), len(radii))
        Vertical force ``K_z(z, R)``, in (km/s)^2/kpc.
    """

    # import physical constants
    G = constants.G

    # Hankel transform of order 0: Sigma(R) -> Sigma_tilde(k) ---
    h0 = Hankel(radii_grid, nu=0, q=1)
    k, sdens_k0 = h0(sdens)

    # apply vertical thickness correction C(k,z0)
    sdens_k0 *= _vertical_correction(k, scale_height, z_layer)  # shape (n_k,)

    # initialise the vertical force 2D grid (z,R)
    vertical_force = np.empty((len(z_grid), len(radii)))  # shape (n_z,n_R)

    # inverse Hankel transform of order 0: back to real space
    # Note: radial force uses H1 (J1 kernel); vertical force uses H0 (J0 kernel)
    h0_inv = Hankel(k, nu=0, q=1)

    # compute the vertical force F(z,R) at radius R
    for iz, z_R in enumerate(z_grid):
        # apply the e^{-kz} damping factor for height z above midplane
        # This is the key difference from the radial force:
        # it encodes how the potential of each Fourier mode decays with z
        sdens_k = sdens_k0 * np.exp(-k * z_R)

        # multiply by k to get the z-derivative of the potential
        # K_z = -dPhi/dz = -2piG * integral[ k * Sigma_k * C(k,z0) * e^{-kz} * J0(kR) dk ]
        sdens_k *= k

        # get the radii and the force amplitude
        force_radii, force_kernel = h0_inv(sdens_k)

        # compute the vertical force
        vertical_force_k = -2.0 * np.pi * G * force_kernel

        # interpolate back to input radii
        interp = interp1d(force_radii, vertical_force_k, bounds_error=False, fill_value='extrapolate')
        vertical_force[iz] = interp(radii)  # vertical force F(z,R) at R (f(z))

    return vertical_force  # shape (n_z,n_R)


def _vertical_force_dm(radii, z, halo_model, halo_params):
    """Compute the vertical gravitational force of a dark matter halo on a (z, R) grid.

    Parameters
    ----------
    radii : ndarray
        Radii at which the force is evaluated, in kpc.
    z : ndarray
        Heights above the midplane at which the force is evaluated, in kpc.
    halo_model : {'iso', 'burkert', 'nfw', 'dc14'}
        Dark matter halo density profile.
    halo_params : dict
        Parameters of the chosen halo model (as expected by the
        corresponding ``v_*`` circular-velocity function).

    Returns
    -------
    ndarray, shape (len(z), len(radii))
        Vertical force ``K_z(z, R) = -Vc(r)^2 * z / r^2``, where
        ``r = sqrt(R^2 + z^2)``, in (km/s)^2/kpc.
    """

    # 3D radius at height z
    # broadcast: r[j,i] = sqrt(R[i]^2 + z[j]^2)
    R3d = np.sqrt(radii[np.newaxis, :] ** 2 + z[:, np.newaxis] ** 2)  # shape (n_z,n_R)

    # dispatch to the appropriate halo circular velocity function
    # evaluated at R, i.e. the 3D distance from the centre
    Vc = circular_velocity_dm(R3d, halo_model, **halo_params)

    # compute the vertical force in the (z,R) space
    # Kz = -(Vc^2/R) * (z/R) = -Vc^2 * z / R^2
    # safe division: if z=0 the force is identically zero anyway
    with np.errstate(invalid='ignore', divide='ignore'):
        vertical_force = -np.where(R3d > 0, Vc**2 * z[:, np.newaxis] / R3d**2, 0.0)  # shape (n_z,n_R)
    return vertical_force
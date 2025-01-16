\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Galaxy models.

This module provides routines for computing the evolution of SMBH pair
orbits in different separation regimes.

Routines
--------
TODO: list routines

See Also
--------
TODO: list other relevant modules

Notes
-----
TODO: Add any notes

References
----------
TODO: Add relevant references, including to Binney & Tremaine (2008), Merritt (2013), and Sesana & Khan (2015)

Examples
--------
TODO: add usage examples

"""
# -----------------------------------------------------------------------------
# Copyright (c) 2015, the IPython Development Team and José Fonseca.
#
# Distributed under the terms of the Creative Commons License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
#
#
# REFERENCES:
# http://ipython.org/ipython-doc/rel-0.13.2/development/coding_guide.html
# https://www.python.org/dev/peps/pep-0008/
# -----------------------------------------------------------------------------
'''
OPTIONS ------------------------------------------------------------------
A description of each option that can be passed to this script

ARGUMENTS -------------------------------------------------------------
A description of each argument that can or must be passed to this script
'''

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# stdlib imports -------------------------------------------------------

# Third-party imports -----------------------------------------------
import numpy as np
import astropy.constants as const
import astropy.units as u
from scipy.special import hyp2f1
from scipy.stats import truncnorm

# Our own imports ---------------------------------------------------
from .config import COSMOLOGY as cosmo
from .smbhb_models import differential_pair_number


# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# LOCAL UTILITIES
# -----------------------------------------------------------------------------
# Changes the default string encoding to UTF-8


# -----------------------------------------------------------------------------
# CLASSES
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# POPULATIONS -----------------------------------------
def differential_dagn_mass(
    log10_mbh,
    log10_mbulge,
    log10_mstellar,
    redz, 
    mratq,
    a_sep,
    phi0, 
    phi1, 
    log10_mbreak, 
    alpha0, 
    alpha1,
    f0, 
    alpha_f, 
    beta_f, 
    gamma_f,
    tau0, 
    alpha_t, 
    beta_t, 
    gamma_t,
    alpha_mbh,
    beta_mbh,
    eps_mbh,
    dagn_frac=.03,
    cosmo=cosmo,
    redz_pair_grid=np.append([0], np.geomspace(1e-4, 100, num=1000)),
    fet0=.587,
    zet0=2.808,
    ket=-3.775,
    mbulge_disp=.2
):
    """Compute the dual agn number density as a function of luminosity.

    Parameters
    ----------
    log10_mbh : float or shape (H,) array_like of float
        Base-10 logarithm of black hole mass
        Units: dex(Msun)
    log10_mbulge : float or shape (B,) array_like of float
        Base-10 logarithm of galaxy bulge mass
        Units: dex(Msun)
    log10_mstellar : float or shape (M,) array_like of float
        Base-10 logarithm of galaxy stellar mass
        Units: dex(Msun
    redz : float or shape (Z,) array_like of float
        Redshift
    mratq : float or shape (Q,) array_like of float
        Galaxy mass ratio
    a_sep : float or array_like of float
        Binary separation, i.e., semi-major axis assuming circular
        orbits
    phi0 : float or shape (P,) array_like of float
        Intercept of the redshift-dependent normalization.
    phi1 : float or shape (P,) array_like of float
        Slope of the redshift-dependent normalization.
    log10_mbreak : float or shape (P,) array_like of float
        Base-10 logarithm of break mass.
        Units: dex(Msun)
    alpha0 : float or shape (P,) array_like of float
        Intercept of the redshift-dependent low-mass slope.
    alpha1 : float or shape (P,) array_like of float
        Slope of the redshift-dependent low-mass slope.
    f0 : float or shape (P,) array_like of float
        Local pair fraction at redshift 0.
    alpha_f : float or shape (P,) array_like of float
        Power-law slope of pair fraction mass dependence.
    beta_f : float or shape (P,) array_like of float
        Power-law slope of pair fraction redshift dependence.
    gamma_f : float or shape (P,) array_like of float
        Power-law slope of pair fraction galaxy mass-ratio dependence.
    tau0 : float or shape (P,) array_like of float
        Local galaxy merger timescale at redshift 0.
    alpha_t : float or shape (P,) array_like of float
        Power-law slope of merger timescale mass dependence.
    beta_t : float or shape (P,) array_like of float
        Power-law slope of merger timescale redshift dependence.
    gamma_t : float or shape (P,) array_like of float
        Power-law slope of merger timescale galaxy mass-ratio dependence.
    alpha_mbh : float or shape (P,) array_like of float
        Power-law slope of M_BH - M_bulge relation.
    beta_mbh : float or shape (P,) array_like of float
        Intercept of M_BH - M_bulge relation.
   eps_mbh : float or shape (P,) array_like of float
        Intrinsic scatter of M_BH - M_bulge relation.

    Returns
    -------
    bmr : shape (P, M, B, Z, Q) array of float
        The bulge merger rate.

    Notes
    -----
    Assumes circular SMBH pairs.

    """
    # sample binary merger rate first
    print("Sampling SMBH pairs....")
    diff_n = differential_pair_number(
        log10_mbh,
        log10_mbulge,
        log10_mstellar,
        redz,
        mratq,
        a_sep,
        phi0=phi0,
        phi1=phi1,
        log10_mbreak=log10_mbreak,
        alpha0=alpha0,
        alpha1=alpha1,
        f0=f0,
        alpha_f=alpha_f,
        beta_f=beta_f,
        gamma_f=gamma_f,
        tau0=tau0,
        alpha_t=alpha_t,
        beta_t=beta_t,
        gamma_t=gamma_t, 
        alpha_mbh=alpha_mbh, 
        beta_mbh=beta_mbh,
        eps_mbh=eps_mbh
    )  # [P, H, Z, A]
    print("Done!")

    diff_n_dagn = dagn_frac * diff_n
    return diff_n_dagn


def differential_dagn_luminosity(
    log10_lbol,
    log10_mbh,
    log10_mbulge,
    log10_mstellar,
    redz, 
    mratq,
    a_sep,
    phi0, 
    phi1, 
    log10_mbreak, 
    alpha0, 
    alpha1,
    f0, 
    alpha_f, 
    beta_f, 
    gamma_f,
    tau0, 
    alpha_t, 
    beta_t, 
    gamma_t,
    alpha_mbh,
    beta_mbh,
    eps_mbh,
    erdf,
    dagn_frac=.03,
    cosmo=cosmo,
    redz_pair_grid=np.append([0], np.geomspace(1e-4, 100, num=1000)),
    fet0=.587,
    zet0=2.808,
    ket=-3.775,
    mbulge_disp=.2,
    erdf_args=None,
    erdf_kwargs=None,
):
    """Compute the dual agn number density as a function of luminosity.

    Parameters
    ----------
    log10_mbh : float or shape (H,) array_like of float
        Base-10 logarithm of black hole mass
        Units: dex(Msun)
    log10_mbulge : float or shape (B,) array_like of float
        Base-10 logarithm of galaxy bulge mass
        Units: dex(Msun)
    log10_mstellar : float or shape (M,) array_like of float
        Base-10 logarithm of galaxy stellar mass
        Units: dex(Msun
    redz : float or shape (Z,) array_like of float
        Redshift
    mratq : float or shape (Q,) array_like of float
        Galaxy mass ratio
    a_sep : float or array_like of float
        Binary separation, i.e., semi-major axis assuming circular
        orbits
    phi0 : float or shape (P,) array_like of float
        Intercept of the redshift-dependent normalization.
    phi1 : float or shape (P,) array_like of float
        Slope of the redshift-dependent normalization.
    log10_mbreak : float or shape (P,) array_like of float
        Base-10 logarithm of break mass.
        Units: dex(Msun)
    alpha0 : float or shape (P,) array_like of float
        Intercept of the redshift-dependent low-mass slope.
    alpha1 : float or shape (P,) array_like of float
        Slope of the redshift-dependent low-mass slope.
    f0 : float or shape (P,) array_like of float
        Local pair fraction at redshift 0.
    alpha_f : float or shape (P,) array_like of float
        Power-law slope of pair fraction mass dependence.
    beta_f : float or shape (P,) array_like of float
        Power-law slope of pair fraction redshift dependence.
    gamma_f : float or shape (P,) array_like of float
        Power-law slope of pair fraction galaxy mass-ratio dependence.
    tau0 : float or shape (P,) array_like of float
        Local galaxy merger timescale at redshift 0.
    alpha_t : float or shape (P,) array_like of float
        Power-law slope of merger timescale mass dependence.
    beta_t : float or shape (P,) array_like of float
        Power-law slope of merger timescale redshift dependence.
    gamma_t : float or shape (P,) array_like of float
        Power-law slope of merger timescale galaxy mass-ratio dependence.
    alpha_mbh : float or shape (P,) array_like of float
        Power-law slope of M_BH - M_bulge relation.
    beta_mbh : float or shape (P,) array_like of float
        Intercept of M_BH - M_bulge relation.
   eps_mbh : float or shape (P,) array_like of float
        Intrinsic scatter of M_BH - M_bulge relation.

    Returns
    -------
    bmr : shape (P, M, B, Z, Q) array of float
        The bulge merger rate.

    Notes
    -----
    Assumes circular SMBH pairs.

    """
    # sample binary merger rate first
    print("Sampling SMBH pairs....")
    diff_n_mass = differential_dagn_mass(
        log10_mbh,
        log10_mbulge,
        log10_mstellar,
        redz,
        mratq,
        a_sep,
        phi0=phi0,
        phi1=phi1,
        log10_mbreak=log10_mbreak,
        alpha0=alpha0,
        alpha1=alpha1,
        f0=f0,
        alpha_f=alpha_f,
        beta_f=beta_f,
        gamma_f=gamma_f,
        tau0=tau0,
        alpha_t=alpha_t,
        beta_t=beta_t,
        gamma_t=gamma_t, 
        alpha_mbh=alpha_mbh, 
        beta_mbh=beta_mbh,
        eps_mbh=eps_mbh,
        dagn_frac=dagn_frac,
        cosmo=cosmo,
        redz_pair_grid=redz_pair_grid,
        fet0=fet0,
        zet0=zet0,
        ket=ket,
        mbulge_disp=mbulge_disp
    )  # [P, H, Z, A]
    print("Done!")

    # compute erdf
    edd_consts = (4 * np.pi * const.G * const.m_p * const.c / const.sigma_T).to(u.erg / u.s / u.Msun).value
    log10_edd_consts = np.log10(edd_consts)
    log10_edd_rat = log10_lbol[:, None] - log10_mbh[None, :] - log10_edd_consts  # [L, H]
    # erdf_term = erdf(log10_edd_rat, log10_mbh, *erdf_args, **erdf_kwargs)  # [P, L, H]
    if erdf_args is not None:
        if erdf_kwargs is not None:
            erdf_term = erdf(log10_edd_rat, log10_mbh, *erdf_args, **erdf_kwargs)  # [P, L, H]
        else:
            erdf_term = erdf(log10_edd_rat, log10_mbh, *erdf_args)  # [P, L, H]
    elif erdf_kwargs is not None:
        erdf_term = erdf(log10_edd_rat, log10_mbh, **erdf_kwargs)  # [P, L, H]
    else:
        erdf_term = erdf(log10_edd_rat, log10_mbh)  # [P, L, H]

    # compute the mass-luminosity function
    diff_n_mass_lum = diff_n_mass[:, None, :, :, :] * erdf_term[:, :, :, None, None]

    # marginalize over mass
    diff_n_lum = np.trapezoid(diff_n_mass_lum, log10_mbh, axis=2)
    
    return diff_n_lum

# EDDINGTON RATIO DISTRIBUTION FUNCTIONS --------------------------------------
def _erdf_norm(
    log10_edd_rat_break=-1.338,
    low_slope=.38,
    high_slope=.38+2.260,
    log10_edd_rat_min=-3,
    log10_edd_rat_max=1
):
    # constant factors
    normalization = low_slope * np.log(10)
    a = 1,
    b = low_slope / (low_slope - high_slope)
    c = (2 * low_slope - high_slope) / (low_slope - high_slope)

    # minimum term
    min_term = np.power(10, -low_slope * (log10_edd_rat_min - log10_edd_rat_break))
    z = - np.power(10, (high_slope - low_slope) * (log10_edd_rat_min - log10_edd_rat_break))
    min_term = min_term * hyp2f1(a, b, c, z)

    # maximum term
    max_term = np.power(10, -low_slope * (log10_edd_rat_max - log10_edd_rat_break))
    z = - np.power(10, (high_slope - low_slope) * (log10_edd_rat_max - log10_edd_rat_break))
    max_term = max_term * hyp2f1(a, b, c, z)

    res = normalization / (min_term - max_term)
    return res
    

def erdf_ananna2022(
    log10_edd_rat,
    log10_mbh=None,
    log10_edd_rat_break=-1.338,
    low_slope=.38,
    high_slope=.38+2.260,
    log10_edd_rat_min=-3,
    log10_edd_rat_max=1
):
    # set up parameters for Monte Carlo compatibility
    log10_edd_rat_break = np.atleast_1d(log10_edd_rat_break)
    low_slope = np.atleast_1d(low_slope)
    high_slope = np.atleast_1d(high_slope)
    
    # calculate ERDF normalization
    normalization = _erdf_norm(
        log10_edd_rat_break=log10_edd_rat_break,
        low_slope=low_slope,
        high_slope=high_slope,
        log10_edd_rat_min=log10_edd_rat_min,
        log10_edd_rat_max=log10_edd_rat_max
    )

    # expand parameter dimensionality to be one larger than the Eddington ratio dimensionality
    while np.ndim(log10_edd_rat_break) < np.ndim(log10_edd_rat) + 1:
        log10_edd_rat_break = log10_edd_rat_break[..., None]

    while np.ndim(low_slope) < np.ndim(log10_edd_rat) + 1:
        low_slope = low_slope[..., None]

    while np.ndim(high_slope) < np.ndim(log10_edd_rat) + 1:
        high_slope = high_slope[..., None]

    while np.ndim(normalization) < np.ndim(log10_edd_rat) + 1:
        normalization = normalization[..., None]
    
    # calculate the ERDF
    log10_edd_rat_scaled = log10_edd_rat[None, ...] - log10_edd_rat_break[:, None]
    log10_low_slope_term = low_slope[:, None] * log10_edd_rat_scaled
    log10_high_slope_term = high_slope[:, None] * log10_edd_rat_scaled

    res = np.power(10, log10_low_slope_term) + np.power(10, log10_high_slope_term)
    res = normalization[:, None] / res

    # apply boundary constraints
    res = np.where(log10_edd_rat < log10_edd_rat_min, 0, res)
    res = np.where(log10_edd_rat > log10_edd_rat_max, 0, res)

    return np.squeeze(res)


def erdf_nobuta2012(
    log10_edd_rat,
    log10_mbh,
    log10_norm=22.46,
    slope=.469,
    scatter=.4/.469,
    log10_edd_rat_min=-3,
    log10_edd_rat_max=1
):
    # set up parameters for Monte Carlo compatibility
    log10_norm = np.atleast_1d(log10_norm)
    slope = np.atleast_1d(slope)
    scatter = np.atleast_1d(scatter)
    
    # compute the eddington ratio scaling in log space
    const_term = (4 * np.pi * const.G * const.m_p * const.c / const.sigma_T).to(u.erg / u.s / u.Msun).value
    log10_const_term = np.log10(const_term)
    
    # compute the log-space expected value
    log10_edd_rat_exp = log10_norm[:, None] / slope[:, None] - log10_mbh[None, :] - log10_const_term
    log10_edd_rat_exp = log10_edd_rat_exp / (1 - 1 / slope[:, None])

    a = (log10_edd_rat_min - log10_edd_rat_exp) / scatter
    b = (log10_edd_rat_max - log10_edd_rat_exp) / scatter
    res = truncnorm.pdf(log10_edd_rat, a, b, loc=log10_edd_rat_exp, scale=scatter)
    
    return np.squeeze(res)


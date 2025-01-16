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
import astropy.units as u

# Our own imports ---------------------------------------------------
from .config import COSMOLOGY as cosmo
from .scaling_relations import prob_mbulge_mstellar_elliptical, prob_mbulge_mstellar_spiral


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

# MASS FUNCTIONS -----------------------------------------
def galaxy_stellar_mass_function(log10_mstellar, redz, phi0, phi1, log10_mbreak, alpha0, alpha1):
    """Compute the galaxy stellar mass function.

    Computes the galaxy stellar mass function as in Chen et al. (2019)
    and Casey-Clyde et al. (submitted).

    Parameters
    ----------
    log10_mstellar : float or array_like of float
        Base-10 logarithm of galaxy stellar mass
        Units: dex(Msun)
    redz : float or array_like of float
        Redshift
    phi0 : float or array_like of float
        Intercept of the redshift-dependent normalization.
    phi1 : float or array_like of float
        Slope of the redshift-dependent normalization.
    log10_mbreak : float or array_like of float
        Base-10 logarithm of break mass.
        Units: dex(Msun)
    alpha0 : float or array_like of float
        Intercept of the redshift-dependent low-mass slope.
    alpha1 : float or array_like of float
        Slope of the redshift-dependent low-mass slope.

    Notes
    -----
    Any parameters provided as an array_like should have matching or
    broadcastable shapes.

    """
    # calculate the overall normalization and low-mass slope
    log10_norm = np.log10(np.log(10)) + phi0 + phi1 * redz
    slope = alpha0 + alpha1 * redz

    # scale the masses by the break mass
    log10_m_scaled = log10_mstellar - log10_mbreak

    # compute mass term
    log10_m_term = (1 + slope) * log10_m_scaled - np.power(10, log10_m_scaled) * np.log10(np.e)

    # compute the galaxy stellar mass function
    log10_gsmf = log10_norm + log10_m_term
    gsmf = np.power(10, log10_gsmf)
    return gsmf


def galaxy_pairing_rate(
    log10_mstellar, 
    redz, 
    mratq, 
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
    cosmo=cosmo
):
    """Compute the galaxy pairing rate.

    Computes the galaxy pairing rate as in Chen et al. (2019) and
    Casey-Clyde et al. (submitted).

    Parameters
    ----------
    log10_mstellar : float or array_like of float
        Base-10 logarithm of galaxy stellar mass
        Units: dex(Msun)
    redz : float or array_like of float
        Redshift
    mratq : float or array_like of float
        Galaxy mass ratio
    phi0 : float or array_like of float
        Intercept of the redshift-dependent normalization.
    phi1 : float or array_like of float
        Slope of the redshift-dependent normalization.
    log10_mbreak : float or array_like of float
        Base-10 logarithm of break mass.
        Units: dex(Msun)
    alpha0 : float or array_like of float
        Intercept of the redshift-dependent low-mass slope.
    alpha1 : float or array_like of float
        Slope of the redshift-dependent low-mass slope.
    f0 : float or array_like of float
        Local pair fraction at redshift 0.
    alpha_f : float or array_like of float
        Power-law slope of pair fraction mass dependence.
    beta_f : float or array_like of float
        Power-law slope of pair fraction redshift dependence.
    gamma_f : float or array_like of float
        Power-law slope of pair fraction galaxy mass-ratio dependence.
    tau0 : float or array_like of float
        Local galaxy merger timescale at redshift 0.
    alpha_t : float or array_like of float
        Power-law slope of merger timescale mass dependence.
    beta_t : float or array_like of float
        Power-law slope of merger timescale redshift dependence.
    gamma_t : float or array_like of float
        Power-law slope of merger timescale galaxy mass-ratio dependence.

    Notes
    -----
    Any parameters provided as an array_like should have matching or
    broadcastable shapes.

    """
    # galaxy stellar mass function
    gsmf = galaxy_stellar_mass_function(
        log10_mstellar, 
        redz,
        phi0=phi0,
        phi1=phi1,
        log10_mbreak=log10_mbreak,
        alpha0=alpha0,
        alpha1=alpha1,
    )
    
    # pair fraction
    pair_frac = galaxy_pair_fraction(
        log10_mstellar, 
        redz,
        mratq,
        f0=f0,
        alpha=alpha_f,
        beta=beta_f,
        gamma=gamma_f,
    )

    # merger timescale
    merger_time = galaxy_merger_timescale(
        log10_mstellar, 
        redz,
        mratq,
        tau0=tau0,
        alpha=alpha_t,
        beta=beta_t,
        gamma=gamma_t,
    )

    # compute galaxy pairing rate
    log10_gsmf = np.log10(gsmf)
    log10_pair_frac = np.log10(pair_frac)
    log10_merger_time = np.log10(merger_time)
    log10_gpr = log10_gsmf + log10_pair_frac - log10_merger_time
    gpr = np.power(10, log10_gpr)
    return gpr


def _compute_pairing_redz(
    log10_mstellar, 
    redz_merge, 
    mratq, 
    tau0, 
    alpha_t, 
    beta_t, 
    gamma_t, 
    redz_pair_grid=np.append([0], np.geomspace(1e-4, 100, num=1000))
):
    """Compute the pairing redshift of galaxies which merge at `redz_merge`.

    Parameters
    ----------
    log10_mstellar : float or shape (M,) array_like of float
        Base-10 logarithm of galaxy stellar mass
        Units: dex(Msun)
    redz : float or shape (Z,) array_like of float
        Redshift
    mratq : float or shape (Q,) array_like of float
        Galaxy mass ratio
    tau0 : float or shape (P,) array_like of float
        Local galaxy merger timescale at redshift 0.
    alpha_t : float or shape (P,) array_like of float
        Power-law slope of merger timescale mass dependence.
    beta_t : float or shape (P,) array_like of float
        Power-law slope of merger timescale redshift dependence.
    gamma_t : float or shape (P,) array_like of float
        Power-law slope of merger timescale galaxy mass-ratio dependence.

    Returns
    -------
    redz_pair : shape (P, M, Z, Q) array of float
        The pairing redshift.

    """
    # at each galaxy pairing redshift, calculate the galaxy merger timescale
    merger_time = galaxy_merger_timescale(
        log10_mstellar[None, :, None, None], 
        redz_pair_grid[None, None, :, None],
        mratq[None, None, None, :],
        tau0=tau0[:, None, None, None],
        alpha=alpha_t[:, None, None, None],
        beta=beta_t[:, None, None, None],
        gamma=gamma_t[:, None, None, None],
    )
    # shape: [P, M, N, Q]

    # compute universe age at pairing redshift
    age_pair = cosmo.age(redz_pair_grid).to(u.Gyr).value  # [N]
    
    # calculate the age of the universe at the merger redshift correponding to each pairing redshift
    age_merge_grid = age_pair[None, None, :, None] + merger_time

    # compute age of the universe at the merger redshift
    age_merge = cosmo.age(redz_merge).to(u.Gyr).value  # [Z]

    # calculate the pairing redshift corresponding to each merger redshift in z_bhb_range
    ## first we find the interpolation indexes
    interp_idxs = np.sum(age_merge_grid[..., None] >= age_merge, axis=2) - 1  # [P, M, Q, Z]
    interp_idxs = np.swapaxes(interp_idxs, -1, -2)  # [P, M, Z, Q]
    no_merge_idxs = np.greater_equal(interp_idxs+1, len(age_pair))
    interp_idxs[no_merge_idxs] = len(age_pair)-2
    
    ## next we interpolate to find the pairing redshifts corresponding to each merger redshift
    try:
        x0 = np.take_along_axis(age_merge_grid, interp_idxs, axis=2)  # [P, M, Z, Q]
        y0 = redz_pair_grid[interp_idxs]  # [P, M, Z, Q]
        x1 = np.take_along_axis(age_merge_grid, interp_idxs+1, axis=2)  # [P, M, Z, Q]
        y1 = redz_pair_grid[interp_idxs+1]  # [P, M, Z, Q]
    except IndexError as e:
        raise IndexError(
            f"interp_idxs.max: {np.max(interp_idxs)}"
            f"\nage_merge.max: {np.max(age_merge)}"
            f"\nage_merge_grid.max: {np.max(age_merge_grid)}"
                        ) from e

    # estimate pairing redshift
    redz_pair = y0 + (age_merge[None, :, None] - x0) * (y1 - y0) / (x1 - x0)  # [P, M, Z, Q]
    return redz_pair, no_merge_idxs


def galaxy_merger_rate(
    log10_mstellar, 
    redz, 
    mratq, 
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
    cosmo=cosmo,
    redz_pair_grid=np.append([0], np.geomspace(1e-4, 100, num=1000))
):
    """Compute the galaxy merger rate.

    Computes the galaxy merger rate as in Chen et al. (2019) and
    Casey-Clyde et al. (submitted).

    Parameters
    ----------
    log10_mstellar : float or shape (M,) array_like of float
        Base-10 logarithm of galaxy stellar mass
        Units: dex(Msun)
    redz : float or shape (Z,) array_like of float
        Redshift
    mratq : float or shape (Q,) array_like of float
        Galaxy mass ratio
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

    Returns
    -------
    gmr : shape (P, M, Z, Q) array of float
        The galaxy merger rate.

    """
    # compute the pairing redshift from the merger timescale
    redz_pair, no_merge = _compute_pairing_redz(
        log10_mstellar, 
        redz, 
        mratq, 
        tau0, 
        alpha_t, 
        beta_t, 
        gamma_t, 
        redz_pair_grid=redz_pair_grid
    )
    
    # galaxy pairing rate at pairing redshift
    gpr = galaxy_pairing_rate(
        log10_mstellar[None, :, None, None], 
        redz_pair,
        mratq[None, None, None, :],
        phi0=phi0[:, None, None, None],
        phi1=phi1[:, None, None, None],
        log10_mbreak=log10_mbreak[:, None, None, None],
        alpha0=alpha0[:, None, None, None],
        alpha1=alpha1[:, None, None, None],
        f0=f0[:, None, None, None],
        alpha_f=alpha_f[:, None, None, None],
        beta_f=beta_f[:, None, None, None],
        gamma_f=gamma_f[:, None, None, None],
        tau0=tau0[:, None, None, None],
        alpha_t=alpha_t[:, None, None, None],
        beta_t=beta_t[:, None, None, None],
        gamma_t=gamma_t[:, None, None, None],
    )
    gpr[no_merge] = 0

    # "Compute" galaxy merger rate. Included for completeness
    # or future modification as an assumption
    gmr = gpr
    return gmr


def bulge_merger_rate(
    log10_mbulge, 
    log10_mstellar,
    redz, 
    mratq, 
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
    cosmo=cosmo,
    redz_pair_grid=np.append([0], np.geomspace(1e-4, 100, num=1000)),
    fet0=.587,
    zet0=2.808,
    ket=-3.775,
    mbulge_disp=.2
):
    """Compute the bulge merger rate.

    Computes the galaxy merger rate as in Chen et al. (2019) and
    Casey-Clyde et al. (submitted).

    Parameters
    ----------
    log10_mbulge : float or shape (B,) array_like of float
        Base-10 logarithm of galaxy bulge mass
        Units: dex(Msun)
    log10_mstellar : float or shape (M,) array_like of float
        Base-10 logarithm of galaxy stellar mass
        Units: dex(Msun)
    redz : float or shape (Z,) array_like of float
        Redshift
    mratq : float or shape (Q,) array_like of float
        Galaxy mass ratio
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

    Returns
    -------
    bmr : shape (P, M, B, Z, Q) array of float
        The bulge merger rate.

    """
    # compute galaxy merger rate
    gmr = galaxy_merger_rate(
        log10_mstellar, 
        redz,
        mratq,
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
        redz_pair_grid=redz_pair_grid
    )  #[P, M, Z, Q]

    # compute early and late-type galaxy fractions
    et_frac = early_type_fraction(redz, fet0=fet0, zet0=zet0, ket=ket)
    lt_frac = 1 - et_frac

    # compute probability of each bulge mass, given the galaxy masses
    prob_mbulge_mgal_et = prob_mbulge_mstellar_elliptical(log10_mbulge[:, None], log10_mstellar[None, :], disp=mbulge_disp)  # [M, B]
    prob_mbulge_mgal_lt = prob_mbulge_mstellar_spiral(log10_mbulge[:, None], log10_mstellar[None, :])
    prob_mbulge_mgal = et_frac[None, None, :] * prob_mbulge_mgal_et[..., None]
    prob_mbulge_mgal = prob_mbulge_mgal + lt_frac[None, None, :] * prob_mbulge_mgal_lt[..., None]  # [B, M, Z]

    # compute bulge merger rate
    bmr = prob_mbulge_mgal[None, :, :, :, None] * gmr[:, None, : :, :]  # [P, B, M, Z, Q]
    return bmr


# MERGER RELATIONS ------------------------------------
def galaxy_pair_fraction(log10_mstellar, redz, mratq, f0, alpha, beta, gamma):
    """Compute the galaxy pair fraction.

    Computes the galaxy pair fraction as in Chen et al. (2019) and
    Casey-Clyde et al. (submitted).

    Parameters
    ----------
    log10_mstellar : float or array_like of float
        Base-10 logarithm of galaxy stellar mass
        Units: dex(Msun)
    redz : float or array_like of float
        Redshift
    mratq : float or array_like of float
        Galaxy mass ratio
    f0 : float or array_like of float
        Local pair fraction at redshift 0.
    alpha : float or array_like of float
        Power-law slope of mass dependence.
    beta : float or array_like of float
        Power-law slope of redshift dependence.
    gamma : float or array_like of float
        Power-law slope of galaxy mass-ratio dependence.

    Notes
    -----
    Any parameters provided as an array_like should have matching or
    broadcastable shapes.

    """
    # logarithmic local fraction
    log10_f0 = np.log10(f0)
    
    # mass term
    log10_m_term = alpha * (log10_mstellar - 11)

    # redshift term
    log10_z_term = beta * np.log10(1 + redz)

    # mass ratio term
    log10_q_term = gamma * np.log10(mratq)

    # compute pair fraction
    log10_pair_frac = log10_f0 + log10_m_term + log10_z_term + log10_q_term
    pair_frac = np.power(10, log10_pair_frac)
    return pair_frac


def galaxy_merger_timescale(log10_mstellar, redz, mratq, tau0, alpha, beta, gamma, cosmo=cosmo):
    """Compute the galaxy pair fraction.

    Computes the galaxy pair fraction as in Chen et al. (2019) and
    Casey-Clyde et al. (submitted).

    Parameters
    ----------
    log10_mstellar : float or array_like of float
        Base-10 logarithm of galaxy stellar mass
        Units: dex(Msun)
    redz : float or array_like of float
        Redshift
    mratq : float or array_like of float
        Galaxy mass ratio
    f0 : float or array_like of float
        Local pair fraction at redshift 0.
    alpha : float or array_like of float
        Power-law slope of mass dependence.
    beta : float or array_like of float
        Power-law slope of redshift dependence.
    gamma : float or array_like of float
        Power-law slope of galaxy mass-ratio dependence.

    Notes
    -----
    Any parameters provided as an array_like should have matching or
    broadcastable shapes.

    """
    # logarithmic local fraction
    log10_tau0 = np.log10(tau0)
    
    # mass term
    log10_m_term = alpha * (log10_mstellar - 11 - np.log10(.4/cosmo.h))

    # redshift term
    log10_z_term = beta * np.log10(1 + redz)

    # mass ratio term
    log10_q_term = gamma * np.log10(mratq)

    # compute pair fraction
    log10_merger_time = log10_tau0 + log10_m_term + log10_z_term + log10_q_term
    merger_time = np.power(10, log10_merger_time)
    return merger_time


# MISCELLANEOUS -----------------------------------------
def early_type_fraction(redz, fet0=0.587, zet0=2.808, ket=-3.775):
    et_frac = fet0 * np.where(redz < zet0, 1, np.power((1 + redz) / (1 + zet0), ket))
    return et_frac

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

# Our own imports ---------------------------------------------------
from .config import COSMOLOGY as cosmo
from .galaxy_models import bulge_merger_rate
from .scaling_relations import prob_mbh_mbulge


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
def binary_merger_rate(
    log10_mbh,
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
    alpha_mbh,
    beta_mbh,
    eps_mbh,
    cosmo=cosmo,
    redz_pair_grid=np.linspace(0, 100, num=1000),
    fet0=.587,
    zet0=2.808,
    ket=-3.775,
    mbulge_disp=.2
):
    """Compute the binary merger rate.

    Computes the galaxy merger rate as in Chen et al. (2019) and
    Casey-Clyde et al. (submitted).

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

    """
    # compute galaxy merger rate
    bmr = bulge_merger_rate(
        log10_mbulge,
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
        cosmo=cosmo,
        redz_pair_grid=redz_pair_grid,
        fet0=fet0,
        zet0=zet0,
        ket=ket,
        mbulge_disp=mbulge_disp
    )  #[P, B, M, Z, Q]

    # # compute the conversion assuming that the bulge mass is the mass of the primary
    # q_bhb = np.power(mratq[None, :], alpha_mbh[:, None])  # [P, Q]
    # log10_mbh1 = log10_mbh[None, :, None] - np.log10(1 + q_bhb[:, None, :])  # [P, H, Q]

    # # compute probability of each BH mass, given the bulge mass
    # p_mbh_mbulge = prob_mbh_mbulge(
    #     log10_mbh1[:, :, None, :], 
    #     log10_mbulge[None, None, :, None], 
    #     alpha_mbh=alpha_mbh[:, None, None, None], 
    #     beta_mbh=beta_mbh[:, None, None, None],
    #     disp=eps_mbh[:, None, None, None]
    # )  # [P, H, B, Q]

    # # compute BH merger rate
    # bhmr = p_mbh_mbulge[:, :, :, None, None, :] * bmr[:, None, :, :, :, :]  # [P, H, B, M, Z, Q]

    p_mbh_mbulge = prob_mbh_mbulge(
        log10_mbh[None, :, None], 
        log10_mbulge[None, None, :], 
        alpha_mbh=alpha_mbh[:, None, None], 
        beta_mbh=beta_mbh[:, None, None],
        disp=eps_mbh[:, None, None]
    )  # [P, H, B]

    # compute BH merger rate
    bhmr = p_mbh_mbulge[:, :, :, None, None, None] * bmr[:, None, :, :, :, :]  # [P, H, B, M, Z, Q]
    return bhmr


# GRAVITATIONAL WAVE BACKGROUND ---------------------------------
def characteristic_strain(log10_mbh, redz, mratqgal, bhmr, alpha_mbh, cosmo=cosmo):
    hc_scaling = (4 / 3
                  * np.power(np.pi, -1 / 3)
                  * np.power(const.G, 5 / 3) 
                  * np.power(const.c, -2) 
                  * np.power(1 / u.yr, -4 / 3) 
                  * np.power(u.Mpc, -3) 
                  * np.power(u.Msun, 5 / 3)).to('').value

    mratqbhb = np.power(mratqgal[None, :], alpha_mbh[:, None])

    dqbhbdqgal = alpha_mbh[:, None] * np.power(mratqgal[None, :], alpha_mbh[:, None] - 1)
    dqgaldqbhb = 1 / dqbhbdqgal

    m_term = np.power(10, 5 * log10_mbh / 3)
    z_term = np.power(1 + redz, -1/3)
    q_term = mratqbhb / np.power(1 + mratqbhb, 2)

    dtdz = dt_dz(redz, cosmo=cosmo)
    bhmr_term = bhmr * dtdz[None, None, :, None] * dqgaldqbhb[:, None, None, :]

    hc_integ = m_term[None, :, None, None] 
    hc_integ = hc_integ * z_term[None, None, :, None]
    hc_integ = hc_integ * q_term[:, None, None, :]
    hc_integ = hc_integ * bhmr_term

    hc_sq = np.trapezoid(hc_integ, mratqgal, axis=3)
    hc_sq = np.trapezoid(hc_sq, redz, axis=2)
    hc_sq = np.trapezoid(hc_sq, log10_mbh, axis=1)
    hc_sq = hc_scaling * hc_sq
    hc = np.sqrt(hc_sq)
    return hc


# UTILITY --------------------------------------------------------
def dt_dz(redz, cosmo=cosmo):
    dtdz = cosmo.hubble_time.to(u.Gyr).value * cosmo.lookback_time_integrand(redz)
    return dtdz

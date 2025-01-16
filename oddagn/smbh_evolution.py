#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SMBH pair evolution rates.

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
import astropy.constants as const

# Our own imports ---------------------------------------------------
from .utilities import coulomb_logarithm
from .galaxy_properties import velocity_dispersion, effective_radius
from .galaxy_properties import scale_radius, stellar_mass_density


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

# SEMI-MAJOR AXIS EVOLUTION -----------------------------------------
def da_dt_dyn(a, mstellar, mbh, q, gamma=1):
    """Rate of SMBH pair separation change under dynamical friction.

    Parameters
    ----------
    a : float or array_like of float
        SMBH pair separation
        Units: parsec
    mstellar : float or array_like of float
        Galaxy stellar mass.
        Units: Msun
    mbh : float or array_like of float
        Total SMBH mass.
        Units: Msun
    q : float or array_like of float
        SMBH mass ratio.
    gamma : float
        Determines the inner slope of the galaxy mass profile. Default
        is 1.

    Returns
    -------
    da_dt : float or array_like of float
        Rate for change for a.
        Units: parsec / Gyr

    """
    # ensure q <= 1
    q = np.where(q > 1, 1 / q, q)
        
    # define constants
    G = const.G.to(u.pc / u.Msun * (u.km / u.s)**2).value
    
    # compute secondary SMBH mass
    # mbh2 = mbh * q / (1 + q)
    log10_mbh = np.log10(mbh)
    log10_mbh2 = log10_mbh + np.log10(q) - np.log10(1 + q)
    mbh2 = np.power(10, log10_mbh2)

    # compute the remnant galaxy radius
    galaxy_radius = effective_radius(mstellar)

    # compute the coulomb logarithm
    coulog = coulomb_logarithm(a, mbh2, mstellar, galaxy_radius)

    # compute the velocity dispersion
    vel_disp = velocity_dispersion(a, mstellar, gamma=gamma)

    # compute the rate of change for SMBH separation
    da_dt = -.302 * coulog * G * mbh2 / (a * vel_disp)  # km / s
    return da_dt * (u.km / u.s).to(u.pc / u.Gyr)  # pc / Gyr


def da_dt_sh(a, mstellar, mbh, gamma=1, H=15):
    """Rate of SMBH pair separation change under stellar hardening.

    Parameters
    ----------
    a : float or array_like of float
        SMBH pair separation
        Units: parsec
    mstellar : float or array_like of float
        Galaxy stellar mass.
        Units: Msun
    mbh : float or array_like of float
        Total SMBH mass.
        Units: Msun
    gamma : float
        Determines the inner slope of the galaxy mass profile. Default
        is 1.
    H : float
        The hardening rate. Default is 15.

    Returns
    -------
    da_dt : float or array_like of float
        Rate for change for a.
        Units: parsec / Gyr

    """
    # compute gravitational constant in appropriate units
    G = const.G.to(u.pc / u.Msun * (u.km / u.s)**2).value

    # compute the influence radius
    r_inf = influence_radius(mbh, mstellar, gamma=gamma)

    # compute the galaxy stellar mass density and velocity dispersion
    # at the binary influence radius
    rho_inf = stellar_mass_density(r_inf, mstellar, gamma=gamma)  # Msun * pc**-3
    sigma_inf = velocity_dispersion(r_inf, mstellar, gamma=gamma)  # km / s

    # compute and return the binary separation evolution
    scale = - G * H * rho_inf / sigma_inf  # km s**-1 * pc**-2
    scale[np.isnan(scale)] = 0
    da_dt = scale * np.power(a, 2)  # km * s**-1
    return da_dt * (u.km / u.s).to(u.pc / u.Gyr)  # pc / Gyr


def da_dt_gw(a, mbh, q):
    """Rate of SMBH pair separation change under gravitational wave emission.

    Parameters
    ----------
    a : float or array_like of float
        SMBH pair separation
        Units: parsec
    mbh : float or array_like of float
        Total SMBH mass.
        Units: Msun
    q : float or array_like of float
        SMBH mass ratio.

    Returns
    -------
    da_dt : float or array_like of float
        Rate for change for a.
        Units: parsec / Gyr

    """
    # compute relevant constants in appropriate units
    G = const.G.to(u.pc / u.Msun * (u.km / u.s)**2).value
    c = const.c.to(u.km / u.s).value

    # pre-compute scaling
    scale = -(64 / 5) * np.power(G, 3) * np.power(c, -5)  # pc**3 * km * s**-1 * Msun**-3

    # compute mass and separation dependent terms
    log10_mbh = np.log10(mbh)
    log10_mbh1 = log10_mbh - np.log10(1 + q)
    log10_mbh2 = log10_mbh1 + np.log10(q)
    log10_m_term = log10_mbh1 + log10_mbh2 + log10_mbh
    m_term = np.power(10, log10_m_term)
    a_term = np.power(a, -3)  # pc**-3

    # compute and return the separation evolution rate
    da_dt = scale * m_term * a_term  # km / s
    return da_dt * (u.km / u.s).to(u.pc / u.Gyr)  # pc / Gyr


def da_dt(a, mstellar, mbh, q, gamma=1, H=15):
    """Rate of SMBH pair separation at all scales.

    Assumes dynamical friction, stellar hardening, and gravitational
    wave emission in appropriate regimes.

    Parameters
    ----------
    a : float or array_like of float
        SMBH pair separation
        Units: parsec
    mstellar : float or array_like of float
        Galaxy stellar mass.
        Units: Msun
    mbh : float or array_like of float
        Total SMBH mass.
        Units: Msun
    q : float or array_like of float
        SMBH mass ratio.
    gamma : float
        Determines the inner slope of the galaxy mass profile. Default
        is 1.
    H : float
        The hardening rate. Default is 15.

    Returns
    -------
    dadt : float or array_like of float
        Rate for change for `a`.
        Units: parsec / Gyr

    """
    # calculate the hardening radius
    # empirically, we find multiplying the hardening radius by a factor
    # of .75 makes the discontinuity when switching from dynamical
    # friction to stellar hardening smaller. This is OK to do because
    # the hardening radius is defined by an inequality, rather than
    # occuring at a specific, fixed value. A more detailed model of the
    # transition from dynamical friction to stellar hardening would
    # likely supercede the need for a hard transition like this, but
    # such a model has not been developed at this time.
    a_hard = .75 * hard_binary_separation(mbh, mstellar, gamma=gamma)

    # calculate evolution rates due to dynamical friction, stellar
    # hardening, and gravitational wave emission
    dadtdyn = da_dt_dyn(a, mstellar, mbh, q, gamma=gamma)
    dadtsh = da_dt_sh(a, mstellar, mbh, gamma=gamma, H=H)
    dadtgw = da_dt_gw(a, mbh, q)

    # combine all evolution rates in relevant regimes
    dadt = np.where(a > a_hard, dadtdyn, dadtsh+dadtgw)
    return np.abs(dadt)


# RESIDENCE TIMESCALES ------------------------------------
def dt_da(a, mstellar, mbh, q, gamma=1, H=15):
    """SMBH pair residence timescales at various separations.

    Assumes dynamical friction, stellar hardening, and gravitational
    wave emission in appropriate regimes.

    Parameters
    ----------
    a : float or array_like of float
        SMBH pair separation
        Units: parsec
    mstellar : float or array_like of float
        Galaxy stellar mass.
        Units: Msun
    mbh : float or array_like of float
        Total SMBH mass.
        Units: Msun
    q : float or array_like of float
        SMBH mass ratio.
    gamma : float
        Determines the inner slope of the galaxy mass profile. Default
        is 1.
    H : float
        The hardening rate. Default is 15.

    Returns
    -------
    dtda : float or array_like of float
        Residence timescale at each binary separation, `a`.
        Units: parsec / Gyr

    """
    # The residence timescale is the inverse of the evolution rate.
    # Thus we first calculate the evolution rate
    dadt = da_dt(a, mstellar, mbh, q, gamma=1, H=15)

    # inverting, we get the residence timescale
    dtda = 1 / dadt
    return dtda


# IMPORTANT RADII -----------------------------------------
def influence_radius(mbh, mstellar, gamma=1):
    """Binary influence radius.

    Calculates the influence radius of a SMBH binary, which we define
    as the distance from the SMBH binary where the contained stellar
    mass is twice the binary total mass. Assumes a Dehnen galaxy mass
    profile.

    Parameters
    ----------
    mbh : float or array_like of float
        Binary total mass.
        Units: Msun
    mstellar : float or array_like of float
        Total galaxy stellar mass.
        Units: Msun
    gamma : float
        Determines the inner slope of the galaxy mass profile. Default
        is 1.

    """
    r0 = scale_radius(mstellar, gamma=gamma)
    log10_mstellar = np.log10(mstellar)
    log10_mbh = np.log10(mbh)
    log10_scaled_mass = log10_mstellar - log10_mbh - np.log10(2)
    scaled_mass = np.power(10, log10_scaled_mass)
    # scaled_mass = mstellar / (2 * mbh)

    r_inf = r0 / (np.power(scaled_mass, 1 / (3 - gamma)) - 1)
    r_inf[np.isnan(r_inf)] = np.inf
    r_inf[r_inf < 0] = np.inf
    return r_inf


def hard_binary_separation(mbh, mstellar, gamma=1):
    """Binary hardening separation.

    Calculates the separation at which a SMBH binary hardens, which we
    define as the separation where the binary orbital velocity is
    greater than the velocity of nearby stars.

    Parameters
    ----------
    mbh : float or array_like of float
        Binary total mass.
        Units: Msun
    mstellar : float or array_like of float
        Total galaxy stellar mass.
        Units: Msun
    gamma : float
        Determines the inner slope of the galaxy mass profile. Default
        is 1.

    """
    r0 = scale_radius(mstellar, gamma=gamma)
    log10_mstellar = np.log10(mstellar)
    log10_mbh = np.log10(mbh)
    log10_scaled_mass = log10_mstellar - log10_mbh
    scaled_mass = np.power(10, log10_scaled_mass)
    # scaled_mass = mstellar / mbh

    a_hard = r0 / (np.power(10, log10_scaled_mass / (3 - gamma)) - 1)
    a_hard[np.isnan(a_hard)] = np.inf
    return a_hard


def isco_separation(mbh):
    """Calculate the innermost stable circular orbit separation of a SMBHB.

    Parameters
    ----------
    mbh : float or array_like of float
        Binary total mass.
        Units: Msun

    """
    # compute relevant constants in appropriate units
    G = const.G.to(u.pc * (u.km / u.s)**2 / u.Msun).value
    c = const.c.to(u.km / u.s).value

    # calculate the schwazschild radius
    R_S = const.G * mbh / np.power(const.c, 2)  # pc

    # calculate the ISCO radius
    r_isco = 3 * R_S  # pc

    # compute the ISCO separation
    a_isco = 2 * r_isco  #pc
    
    return a_isco  # pc


# FUNCTION CATEGORY n -----------------------------------------

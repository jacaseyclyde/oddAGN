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

# FUNCTION CATEGORY 1 -----------------------------------------
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
    mbh2 = mbh * q / (1 + q)

    # compute the remnant galaxy radius
    galaxy_radius = effective_radius(mstellar)

    # compute the coulomb logarithm
    coulog = coulomb_logarithm(a, mbh2, mstellar, galaxy_radius)

    # compute the velocity dispersion
    vel_disp = velocity_dispersion(a, mstellar, gamma=gamma)

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
    G = const.G.to(u.pc / u.Msun * (u.km / u.s)**2).value
    
    r_inf = influence_radius(mbh, mstellar, gamma=gamma)
    rho_inf = stellar_mass_density(r_inf, mstellar, gamma=gamma)  # Msun * pc**-3
    sigma_inf = velocity_dispersion(r_inf, mstellar, gamma=gamma)  # km / s

    scale = - G * H * rho_inf / sigma_inf  # km s**-1 * pc**-2
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
    G = const.G.to(u.pc / u.Msun * (u.km / u.s)**2).value
    c = const.c.to(u.km / u.s).value

    scale = -(64 / 5) * np.power(G, 3) * np.power(c, -5)  # pc**3 * km * s**-1 * Msun**-3

    reduced_mass = mbh * q * np.power(1 + q, -2.)
    m_term = reduced_mass * np.square(mbh)  # Msun**3
    a_term = np.power(a, -3)  # pc**-3

    da_dt = scale * m_term * a_term  # km / s
    return da_dt * (u.km / u.s).to(u.pc / u.Gyr)  # pc / Gyr


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
    scaled_mass = mstellar / (2 * mbh)

    r_inf = r0 / (np.power(scaled_mass, 1 / (3 - gamma)) - 1)
    return r_inf


# FUNCTION CATEGORY n -----------------------------------------

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


# FUNCTION CATEGORY 2 -----------------------------------------


# FUNCTION CATEGORY n -----------------------------------------

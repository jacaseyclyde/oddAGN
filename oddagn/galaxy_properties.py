#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Galaxy properties.

This module provides models of various galaxy properties describing the
distribution of matter in the galaxy and the scales of important 
galactic features.

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


# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# LOCAL UTILITIES
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# CLASSES
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# RADII -----------------------------------------
def effective_radius(mstellar):
    """Compute the effective radius of a galaxy.

    The effective radius of a galaxy can be computed from it's total stellar mass

    Parameters
    ----------
    mstellar : float or array_like of float
        Galaxy stellar mass.

    """
    # scale the mass
    scaled_mass = mstellar / 1e6
    r_eff = np.max([2.95 * np.power(scaled_mass, .596), 34.8 * np.power(scaled_mass, .399)], axis=0)
    return r_eff


def scale_radius(mstellar, gamma=1):
    """Galaxy scale radius, :math:`r_{0}`.

    To leading order, inside :math:`r_{0}` the mass enclosed goes as
    :math:`r_{0}^{\gamma-3}`, while to leading order outside
    :math:`r_{0}` the mass enclosed is constant (Dehnen 1993).

    Parameters
    ----------
    mstellar : float or array_like of float
        The total stellar mass of the galaxy.
        Units: Msun
    gamma : float
        Determines the inner slope of the galaxy mass profile

    Returns
    -------
    r0 : float or array_like of float
        The scale radius of the galaxy
        Units: parsec

    """
    r_eff = effective_radius(mstellar)
    half_mass_radius = r_eff / (.7549 - .00439 * gamma
                                           + .0322 * np.power(gamma, 2)
                                           - .00182 * np.power(gamma, 3))

    r0 = half_mass_radius * (np.power(2, 1 / (3 - gamma)) - 1)  # pc
    return r0


# MASS PROPERTIES -----------------------------------------
def mass_enclosed(r, mstellar, gamma=1):
    """Mass enclosed.

    Computes the total mass enclosed in radius :math:`r`.

    Parameters
    ----------
    r : float or array_like of float
        The radius at which to compute the enclosed mass.
        Units: parsec
    mstellar : float or array_like of float
        The total stellar mass of the galaxy.
        Units: Msun
    gamma : float
        Determines the inner slope of the galaxy mass profile

    Returns
    -------
    menc : float or array_like of float
        The mass enclosed.
        Units: Msun

    """
    # scale radius
    r0 = scale_radius(mstellar, gamma=gamma)

    # enclosed mass
    menc = mstellar * np.power(r / (r + r0), 3 - gamma)
    return menc


# VELOCITY DISPERSION -----------------------------------------
def velocity_dispersion(r, mstellar, gamma=1):
    """Virial velocity dispersion.

    Computes the velocity dispersion of stars at a given radius,
    assuming virialized velocities.

    Parameters
    ----------
    r : float or array_like of float
        The radius at which to compute velocity dispersion.
        Units: parsec
    mstellar : float or array_like of float
        The total stellar mass of the galaxy.
        Units: Msun
    gamma : float
        Determines the inner slope of the galaxy mass profile

    Returns
    -------
    vel_disp : float or array_like of float
        The velocity dispersion.
        Units: km / sec

    """
    # constants
    G = const.G.to(u.pc / u.Msun * (u.km / u.s)**2).value
    
    # scale radius
    menc = mass_enclosed(r, mstellar, gamma=gamma)

    # enclosed mass
    vel_disp = np.sqrt(G * menc / r)
    return vel_disp


# -----------------------------------------------------------------------------
# RUNTIME PROCEDURE
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    '''
    Complete description of the runtime of the script, what it does and how it
    should be used
    :timeComplexityTerm TERM_X: type - term used in the Complexity formula
    :timeComplexityDominantOperation  OP_X: type - operation considered to
        calculate the time complexity of this method
    :timeComplexity: O(OP_X*TERM_X²)
    '''
    # description of the operation perfomed Below
    foo(1, 2)

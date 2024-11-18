#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions.

This module provides utility functions used by other modules in this
package.

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

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# stdlib imports -------------------------------------------------------

# Third-party imports -----------------------------------------------
import numpy as np

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

# DYNAMICAL FRICTION UTILITIES -----------------------------------------
def coulomb_logarithm(a, mbh, mstellar, galaxy_radius):
    """Compute the Coulomb logarithm for a SMBH orbiting in a galaxy.

    Parameters
    ----------
    a : float or array_like of float
        The separation scale at which to calculate the logarithm.
        Units: parsecs
    mbh : float or array_like of float
        The mass of the orbiting SMBH.
        Units: Msun
    mstellar : float or array_like of float
        The mass of the galaxy.
        Units: Msun
    galaxy_radius : float or array_like of float
        The radius of the galaxy.
        Units: parsecs

    Returns
    -------
    coulog : float or array_like of float
        The computed Coulomb logarithm

    """
    coulog = np.log(a * mstellar / (mbh * galaxy_radius))
    return coulog


# FUNCTION CATEGORY 2 -----------------------------------------


# FUNCTION CATEGORY n -----------------------------------------


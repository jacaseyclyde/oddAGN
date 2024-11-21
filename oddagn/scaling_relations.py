#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SMBH-galaxy scaling relations

This module provides empirical scaling relations between SMBHs, their
host galaxies, and other galaxy properties such as effective radius.

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

# Our own imports ---------------------------------------------------
from .galaxy_properties import effective_radius


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

# MASS SCALINGS -----------------------------------------
def log_mbh_log_mbulge_scaling(log10_mbulge, alpha=1.17, beta=8.69):
    """Scaling between bulge mass and black hole mass.

    Computes the expected black hole mass for a given bulge mass
    according to the relation

    .. math::
    
        \log_{10} M_{\mathrm{BH}} = \alpha_{\mathrm{BH}} \log_{10} \left(M_{\mathrm{bulge}} / 10^{11} \; \mathrm{M}_{\odot}\right) + \beta_{\mathrm{BH}} \, ,

    where :math:`\alpha_{\mathrm{BH}}` is the slope of the
    :math:`M_{\mathrm{BH}} - M_{\mathrm{bulge}}` scaling and
    :math:`\beta_{\mathrm{BH}}` is the base-10 logarithm of the overall
    normalization, i.e., the expected mass of a SMBH in a 
    :math:`10^{11} \; \mathrm{M}_{\odot}` galactic bulge.

    Parameters
    ----------
    log10_mbulge : float or array_like of float
        Base-10 logarithm of bulge mass.
    alpha : float or array_like of float
        Scaling relation slope.
    beta : float or array_like of float
        Normalization of the scaling relation.

    Returns
    -------
    log10_mbh : float or array_like of float
        Base-10 logarithm of the expected SMBH mass.

    """
    log10_mbulge_scaled = log10_mbulge - 11
    log10_mbh = alpha * log10_mbulge_scaled + beta
    return log10_mbh


def log_mbulge_log_mbh_scaling(log10_mbh, alpha=1.17, beta=8.69):
    """Scaling between black hole mass and bulge mass.

    Computes the expected bulge mass for a given black hole mass
    according to the relation

    .. math::
    
        \log_{10} M_{\mathrm{bulge}} = \frac{1}{\alpha_{\mathrm{BH}}} \log_{10} \left(M_{\mathrm{BH}} / 10^{\beta_{\mathrm{BH}}} \; \mathrm{M}_{\odot}\right) + 11 \pm \frac{\Delta_{\mathrm{BH}}}{\alpha_{\mathrm{BH}}} \, ,

    where :math:`\alpha_{\mathrm{BH}}` is the slope of the
    :math:`M_{\mathrm{BH}} - M_{\mathrm{bulge}}` scaling and
    :math:`\beta_{\mathrm{BH}}` is the base-10 logarithm of the overall
    normalization, i.e., the expected mass of a SMBH in a 
    :math:`10^{11} \; \mathrm{M}_{\odot}` galactic bulge.

    Parameters
    ----------
    log10_mbh : float or array_like of float
        Base-10 logarithm of black hole mass.
    alpha : float or array_like of float
        Scaling relation slope.
    beta : float or array_like of float
        Normalization of the scaling relation.

    Returns
    -------
    log10_mbulge : float or array_like of float
        Base-10 logarithm of the expected bulge mass.

    """
    log10_mbulge_scaled = (log10_mbh - beta) / alpha
    log10_mbulge = log10_mbulge_scaled + 11
    return log10_mbulge


def log_mstellar_log_mbulge_scaling(log10_mbulge):
    """Calculate the galaxy stellar mass expected for a particular bulge mass.

    Parameters
    ----------
    log10_mbulge : float or array_like of float
        Base-10 logarithm of bulge mass.
        Units: log10(Msun)

    Returns
    -------
    log10_mstellar : float or array_like of float
        Base-10 logarithm of galaxy stellar mass.

    """
    # calculate the bulge fraction
    f_bulge = np.where(
        log10_mbulge >= 11.4,
        .76 + .004 * (log10_mbulge - 11.4),
        np.where(
            log10_mbulge < 10.6,
            .6 + .018 * (log10_mbulge - 9.5),
            .62 + .175 * (log10_mbulge - 10.6)
        )
    )

    # compute the galaxy stellar mass
    log10_mstellar = log10_mbulge - np.log10(f_bulge)
    return log10_mstellar


# PROBABILITIES -----------------------------------------
def prob_mbh_mbulge(log10_mbh, log10_mbulge, alpha_mbh, beta_mbh, disp=.4):
    log10_mbh_centroid = log_mbh_log_mbulge_scaling(
        log10_mbulge, 
        alpha=alpha_mbh, 
        beta=beta_mbh
    )

    # compute the probability
    loc = log10_mbh - log10_mbh_centroid
    exp_term = np.exp(-.5 * np.power(loc / disp, 2))
    prob = exp_term / (disp * np.sqrt(2 * np.pi))
    return prob

    
def prob_mbulge_mstellar_elliptical(log10_mbulge, log10_mstellar, disp=.2):
    log10_mbulge_centroid = log_mstellar_log_mbulge_scaling(log10_mstellar)

    # compute the probability
    loc = log10_mbulge - log10_mbulge_centroid
    exp_term = np.exp(-.5 * np.power(loc / disp, 2))
    prob = exp_term / (disp * np.sqrt(2 * np.pi))
    return prob
    
def prob_mbulge_mstellar_spiral(log10_mbulge, log10_mstellar):
    # assuming B/T ∈ [0.1, 0.3] (Sesana 2013) for late-type galaxies
    BT = np.power(10, log10_mbulge - log10_mstellar)
    prob = np.where((BT >= 0.10) & (BT <= 0.3), BT * np.log(10) / 0.2, 0)
    return prob


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

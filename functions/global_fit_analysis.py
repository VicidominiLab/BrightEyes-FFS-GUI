# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 12:57:37 2025

@author: eslenders
"""

import numpy as np

def make_fit_info(allfits, len_G, r, analysis):
    """
    Make fit info object for global fit

    Parameters
    ----------
    allfits : object
        GUI object containing information about all the fits for a given analysis.
    len_G : int
        number of time points in correlation curve.
    r : list of ints
        Two integer values for the start and end of the fit.
    analysis : object
        GUI analysis object.

    Returns
    -------
    G : np.array()
        2D array [Nt x Ng] with Ng correlations functions of length Nt each.
    start : int
        start index for fit.
    stop : int
        end index for fit.
    fitInfo : np.array()
        2D array with:
            column 0: 1's and 0's for fitted and fixed parameters
            column 1: start values for all parameters
            column 2: min fit bound values for all parameters
            column 3: max fit bound values for all parameters
    fitarraytemp : np.array()
        dummy array.
    fitstartvtemp : np.array()
        dummy array.
    Ntraces : int
        number of curves in the global fit.

    """
    
    Ntraces = len(allfits)
    G = np.zeros((len_G, Ntraces))
    weights = np.zeros((len_G, Ntraces))
    param = np.zeros((9, Ntraces)) # fitinfo, startv, minb, maxb
    
    for f in range(Ntraces):
        # get data
        fit = allfits[f]
        Gtemp = analysis.get_corr(fit.data)
        G[:,f] = Gtemp[:,1]
        Gstd = Gtemp[:, 2] + 1e-10 # convert standard deviation to weights
        weights[:,f] = 1 / Gstd**2
        weights[Gstd==1e-10,f] = 0 # points with 0 std are not taken into account
        minb = fit.minbound
        maxb = fit.maxbound
        stop = np.min((r[1], len_G))
        start = np.min((stop - 1, r[0]))
        start = np.max((0, start))
        fit.fitrange = [start, stop]
        fit_info = fit.fitarray[0:-1]
        param[:, f] = fit.startvalues
       
    return G, start, stop, param, fit_info, minb, maxb, Ntraces, weights


def make_fit_info_global_pch(allfits, len_hist, r, analysis):
    n_hist = len(allfits)
    hist = np.zeros((len_hist, n_hist))
    param = np.zeros((7, n_hist)) # c1, c2, q1, q2, bg, btime, dV0
    psf = np.zeros(int(2 * n_hist))
    fit_info = np.zeros(7)
    
    for i in range(n_hist):
        fit = allfits[i]
        fit_info[:] = fit.fitarray[0:-4]
        param[:, i] = fit.startvalues[0:7]  # c1, c2, q1, q2, bg, btime, dV0, w0, SP, n_bins
        psf[2*i:2*i+2] = fit.startvalues[7:9]
        n_bins = fit.startvalues[9]
        hist_temp = analysis.get_corr(fit.data)
        hist[:,i] = hist_temp[:,1]
        minb = fit.minbound[0:7]
        maxb = fit.maxbound[0:7]
        stop = np.min((r[1], len_hist))
        start = 0
        fit.fitrange = [start, stop]
        
    return hist, start, stop, param, fit_info, list(psf), n_bins, minb, maxb, n_hist

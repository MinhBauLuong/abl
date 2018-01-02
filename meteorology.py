#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit
"""

Space for meteorological functions, ex:
    - shear stuffs
    - stability stuffs
    - calculation of diagnostics from primitive variables

"""

def power_law(z,zref=80.0,Uref=8.0,alpha=0.2):
    return Uref*(z/zref)**alpha

def fit_power_law_alpha(z,U,zref=80.0,Uref=8.0):
    above0 = z > 0
    logz = np.log(z[above0]) - np.log(zref)
    logU = np.log(U[above0]) - np.log(Uref)
    func = lambda xdata,alpha: alpha*xdata
    popt, pcov = curve_fit(func,xdata=logz,ydata=logU,p0=0.2,bounds=(0,np.inf))
    alpha = popt[0]
    resid = U - Uref*(z/zref)**alpha
    SSres = np.sum(resid**2)
    SStot = np.sum((U - np.mean(U))**2)
    R2 = 1.0 - (SSres/SStot)
    return alpha, R2


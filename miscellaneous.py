#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Miscellaneous little functions that get called from a variety of codes.

"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

#==============================================================================
# 
#==============================================================================

def idl_colortable(table='windcolor43.tbl'):
    """
    Processes an IDL color table, which was extracted in IDL as follows:

      loadct,43,File='windcolor.tbl',RGB_TABLE=cmap
      cmap

    Table 43 (default used by NOAA) is called "wind1"

    The output (3,N) array was manually saved into the provided color
    table file and is processed here into a usable matplotlib colormap.

    Parameters
    ----------
    table : str
        Local or provided color table from IDL
    """
    try:
        tab = np.loadtxt(table) # rgb.shape == (3, N)
    except IOError:
        relpath = os.path.join(os.path.dirname(__file__),table)
        tab = np.loadtxt(relpath)
    tab /= 255.0
    Ncolors = tab.shape[1]
    x = np.linspace(0,1,Ncolors)
    cdict = { 'red': [], 'green': [], 'blue': [] }
    for level in range(Ncolors):
        rgb = tab[:,level]
        cdict['red'  ].append( (x[level],rgb[0],rgb[0]) )
        cdict['green'].append( (x[level],rgb[1],rgb[1]) )
        cdict['blue' ].append( (x[level],rgb[2],rgb[2]) )
    return LinearSegmentedColormap('NOAA_wind1', cdict)

#==============================================================================
# 
#==============================================================================
def parse(string):
    """
    Convert strings to float, of possible.
    
    Parameters
    ----------
    datenum : str,
        some string e.g. "2.00"
    """
    try:
        return float(string)
    except:
        return np.nan
#==============================================================================
# 
#==============================================================================
def matlab_datenum_to_python_datetime(datenum):
    """
    Parameters
    ----------
    datenum : int,
        some matlab datenum
    """
    return datetime.fromordinal(int(datenum)) +\
             timedelta(days=datenum%1) - timedelta(days = 366)
#==============================================================================
#              
#==============================================================================
def make_multi_index(tuple_1=None,tuple_2=None,tuple_3=None):
    """
    Returns a multi_index to be used in a data frame. 
    
    Parameters
    ----------    
    tuple_1 : tuple or list,
        values for the leftmost index
    tuple_2 : tuple or list,
        values for the next index to the right
    tuple_3 : tuple or list,
        values for the next index to the right
    
    Returns
    -------    
    pd.MultiIndex,  
        a pandas index instance
    
    Examples
    --------
    multi_index = make_multi_index(std_roughness,std_heights,['A','k'])    
    
    """    
    if (tuple_1 is None) or (tuple_2 is None): 
        return
        
    t1 = []
    t2 = []
    t3 = []
    for x1 in tuple_1:
        for x2 in tuple_2:
            if tuple_3 is not None:
                for x3 in tuple_3:
                    t1.append(x1)
                    t2.append(x2)
                    t3.append(x3)
            else:
                t1.append(x1)
                t2.append(x2)
    if tuple_3 is not None:
        arrays = [t1,t2,t3]
    else:
        arrays = [t1,t2]        
    tuples = list(zip(*arrays))
    
    return pd.MultiIndex.from_tuples(tuples) 
#==============================================================================
# 
#==============================================================================
def smooth_spectrum(freqs,psd,dl=0.1):    
    """
    Smooth a noisy spectrum (the further down the high-frequency spectral tail, the more points are used in the averaging).
    
    Parameters
    ----------
    freqs : np.array,
        frequencies
    psd : np.array,
        signal power
    dl : float,
        factor that modulates the rate at which we increase the smoothing window
        
    Returns
    -------
    np.array,
        smoothed frequencies
    np.array,
        smoothed signal power
    """
    nStart          = 0
    nEnd            = 1
    l               = 0
    freqs_smooth    = []
    psd_smooth      = []
    while nEnd<len(psd):
        freqs_smooth.append(np.mean(freqs[nStart:nEnd+1]))
        psd_smooth.append(np.mean(psd[nStart:nEnd+1]))
        nStart = nEnd ; l += dl ; m = int(np.round(2**l)) ; nEnd += m
    return np.array(freqs_smooth),np.array(psd_smooth)   

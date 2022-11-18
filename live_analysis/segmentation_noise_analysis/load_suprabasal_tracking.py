#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:04:31 2022

@author: xies
"""

import numpy as np
from skimage import io, measure, morphology
from scipy import linalg
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pylab as plt

from mathUtils import surface_area, parse_3D_inertial_tensor

from os import path
from glob import glob
from tqdm import tqdm
import pickle as pkl

dirname = dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/suprabasal_tracking'
ZZ = 72
XX = 460
T = 15
dx = 0.25

from toeplitzDifference import backward_difference,forward_difference

def get_interpolated_curve(cf,smoothing_factor=1e10):

    # Get rid of daughter cells]
    if len(cf) < 4:
        yhat = cf.Volume.values
        
    else:
        t = np.array(range(0,len(cf))) * 12
        v = cf.Volume.values
        # Spline smooth
        spl = UnivariateSpline(t, v, k=3, s=smoothing_factor)
        yhat = spl(t)
        
        # # Nuclear volume
        # nv = cf.Nucleus.values
        # # Spline smooth
        # spl = UnivariateSpline(t, nv, k=3, s=smoothing_factor)
        # nuc_hat = spl(t)

    return yhat
    
def get_growth_rate(cf,field):
    
    assert(field == 'Nucleus' or field == 'Volume')
    
    #@todo: detect + impute NaN for automatically tracked cells
    
    v = cf[field].values
    v_sm = cf[field + ' (sm)'].values
    
    Tb = backward_difference(len(v))
    Tf = forward_difference(len(v))
    gr_b = np.dot(Tb,v)
    gr_f = np.dot(Tb,v)
    
    Tb = backward_difference(len(v_sm))
    Tf = forward_difference(len(v_sm))
    gr_sm_b = np.dot(Tb,v_sm)
    gr_sm_f = np.dot(Tf,v_sm)
    
    gr_b[0] = np.nan
    gr_f[-1] = np.nan
    gr_sm_b[0] = np.nan
    gr_sm_f[-1] = np.nan

    return gr_b,gr_f,gr_sm_b,gr_sm_f

#%% Load the basal cell tracking

basal_tracking = io.imread(path.join(dirname,'suprabasal_tracking.tif'))
allIDs = np.unique(basal_tracking)[1:]

#%% Do pixel level measurements e.g. Surface Area

collated = {k:pd.DataFrame() for k in allIDs}

for t,im in enumerate(basal_tracking):

    properties = measure.regionprops(im, extra_properties = [surface_area])
    
    for p in properties:
        
        basalID = p['label']
        V = p['area'] * dx**2
        Z,Y,X = p['centroid']
        SA = p['surface_area'] * dx**2
        
        s = pd.Series({'basalID': basalID
                       ,'Volume':V
                       ,'Z':Z,'Frame': t
                       ,'Y-pixels':Y,'X-pixels':X
                       ,'Y':Y * dx**2,'X':X * dx**2
                       ,'Phase': 'G0'})
        
        collated[basalID] = collated[basalID].append(s,ignore_index=True)

#%% Calculate spline + growth rates + save

for basalID, df in collated.items():
    # put in the birth volume
    collated[basalID]['Birth volume'] = collated[basalID]['Volume'].values[0]
    
    if len(df) > 1:
        collated[basalID] = df
        
        Vsm = get_interpolated_curve(df)
        df['Volume (sm)'] = Vsm
        gr_f,gr_b,gr_sm_b,gr_sm_f = get_growth_rate(df,'Volume')
        df['Growth rate b'] = gr_f / 12.
        df['Growth rate f'] = gr_b / 12.
        df['Growth rate b (sm)'] = gr_sm_b / 12.
        df['Growth rate f (sm)'] = gr_sm_f / 12.
        df['Specific GR b (sm)'] = gr_sm_b / df['Volume (sm)']
        df['Specific GR f (sm)'] = gr_sm_f / df['Volume (sm)']
        
    collated[basalID] = df

#%
#@todo: daughter/division voluem analysis!

#%

with open(path.join(dirname,'suprabasal.pkl'),'wb') as f:
    pkl.dump(collated,f)
df = pd.concat(collated,ignore_index=True)


    
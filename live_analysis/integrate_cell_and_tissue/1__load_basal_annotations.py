#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:24:28 2022

@author: xies
"""

import numpy as np
from skimage import io, measure, morphology
from scipy import linalg
from scipy.interpolate import UnivariateSpline
import pandas as pd
import matplotlib.pylab as plt

from os import path
from glob import glob
from tqdm import tqdm
import pickle as pkl

dirname = dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
ZZ = 72
XX = 460
T = 15

#%% Load the basal cell tracking

basal_tracking = io.imread(path.join(dirname,'basal_tracks.tif'))
allIDs = np.unique(basal_tracking)[1:]

#%%

def surface_area(im):
    assert(im.ndim == 3)
    eroded = morphology.binary_erosion(im).astype(int)
    outline = im - eroded
    return outline.sum()

def parse_inertial_tensor(I):
    
    L,D = linalg.eig(I)
    L = np.real(L) # Assume no complex solution is necessary
    
    #Sort eigvec by eigenvalue magnitudes
    order = np.flip( np.argsort(L) )
    L = L[order]
    D = D[:,order]
    
    Znorm = np.zeros((3,3))
    Znorm[0,:] = 1
    
    delta = D - Znorm
    norms = np.sqrt( (delta**2).sum(axis=0) )
    
    # Flip Znorm if everything is too large
    if np.all(norms > 1):
        delta = D + Znorm
        norms = np.sqrt( (delta**2).sum(axis=0) )
    
    which_Zaxis = norms <= 1
    single_norm = norms[which_Zaxis]
    assert( len(single_norm) == 1)
    phi = np.rad2deg( np.arcsin(single_norm) )
    
    axial_moment = L[which_Zaxis]
    
    other_axes = norms > 1
    other_moments = L[other_axes]
    
    return axial_moment[0], phi[0], other_moments[0], other_moments[1]

#%%

# columns = ['basalID','Volume','Z','Y','X','Frame','Surface area'
#            ,'Axial angle','Axial component','Planar component 1','Planar component 2']

collated = {k:pd.DataFrame() for k in allIDs}

for t,im in enumerate(basal_tracking):

    properties = measure.regionprops(im, extra_properties = [surface_area])
    
    for p in properties:
        
        basalID = p['label']
        V = p['area']
        Z,Y,X = p['centroid']
        SA = p['surface_area']
        
        I = p['inertia_tensor']
        Iaxial, phi, Ia, Ib = parse_inertial_tensor(I)
        s = pd.Series({'basalID': basalID
                       ,'Volume':V
                       ,'Z':Z,'Y':Y,'X':X, 'Frame': t
                       ,'Surface area':SA
                       ,'Axial angle':phi
                       ,'Axial component':Iaxial
                       ,'Planar component 1':Ia,'Planar component 2':Ib})
    
        collated[basalID] = collated[basalID].append(s,ignore_index=True)
    
#%% Derive simple parameters from existing fields

from toeplitzDifference import backward_difference

def get_interpolated_curve(cf,smoothing_factor=1e5):

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
    
    v = cf[field].values
    v_sm = cf[field + ' (sm)'].values
    Tb = backward_difference(len(v))
    gr = np.dot(Tb,v)
    Tb = backward_difference(len(v_sm))
    gr_sm = np.dot(Tb,v)
    gr[0] = np.nan
    gr_sm[0] = np.nan

    
    return gr,gr_sm

#%% Calculate spline + growth rates

g1_anno = pd.read_csv(path.join(dirname,'tracked_cells/g1_frame.txt'),index_col=0)

for basalID, df in collated.items():
    
    df['Phase'] = '?'
    if len(df) > 1:
        df['Axial eccentricity'] = df['Axial component'] / df['Planar component 2']
        df['Planar eccentricity'] = df['Planar component 2'] / df['Planar component 1']
        df['SA to vol'] = df['Surface area'] / df['Volume']
        collated[basalID] = df
        
        Vsm = get_interpolated_curve(df)
        df['Volume (sm)'] = Vsm
        gr,gr_sm = get_growth_rate(df,'Volume')
        df['Growth rate'] = gr
        df['Growth rate (sm)'] = gr_sm

        # G1 annotations

        g1_frame = g1_anno.loc[basalID]['Frame']
        if g1_frame == '?':
            continue
        else:
            g1_frame = int(g1_frame)
            df['Phase'] = 'G1'
            df.loc[df['Frame'].values > g1_frame,'Phase'] = 'SG2'
            
    collated[basalID] = df

#%%
with open(path.join(dirname,'basal_no_daughters.pkl'),'wb') as f:
    pkl.dump(collated,f)


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

from mathUtils import surface_area, parse_3D_inertial_tensor

from os import path
from glob import glob
from tqdm import tqdm
import pickle as pkl

dirname = dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
ZZ = 72
XX = 460
T = 15

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
    
    gr[0] = np.nan
    gr_sm[0] = np.nan

    return gr_b,gr_f,gr_sm_b,gr_sm_f

#%% Load the basal cell tracking

basal_tracking = io.imread(path.join(dirname,'manual_basal_tracking/basal_tracks.tif'))
allIDs = np.unique(basal_tracking)[1:]

#%% Do pixel level measurements

collated = {k:pd.DataFrame() for k in allIDs}
for t,im in enumerate(basal_tracking):

    properties = measure.regionprops(im, extra_properties = [surface_area])
    
    for p in properties:
        
        basalID = p['label']
        V = p['area']
        Z,Y,X = p['centroid']
        SA = p['surface_area']
        
        I = p['inertia_tensor']
        Iaxial, phi, Ia, Ib, theta = parse_3D_inertial_tensor(I)
        s = pd.Series({'basalID': basalID
                       ,'Volume':V
                       ,'Z':Z,'Y':Y,'X':X, 'Frame': t
                       ,'Surface area':SA
                       ,'Axial angle':phi
                       ,'Axial component':Iaxial
                       ,'Planar component 1':Ia,'Planar component 2':Ib
                       ,'Planar angle':theta})
        
        collated[basalID] = collated[basalID].append(s,ignore_index=True)
    
#% Calculate spline + growth rates

g1_anno = pd.read_csv(path.join(dirname,'2020 CB analysis/tracked_cells/g1_frame.txt'),index_col=0)

for basalID, df in collated.items():
    
    df['Phase'] = '?'
    if len(df) > 1:
        df['Axial eccentricity'] = df['Axial component'] / df['Planar component 2']
        df['Planar eccentricity'] = df['Planar component 2'] / df['Planar component 1']
        df['SA to vol'] = df['Surface area'] / df['Volume']
        collated[basalID] = df
        
        Vsm = get_interpolated_curve(df)
        df['Volume (sm)'] = Vsm
        gr_f,gr_b,gr_sm_b,gr_sm_f = get_growth_rate(df,'Volume')
        df['Growth rate b'] = gr_f
        df['Growth rate f'] = gr_b
        df['Growth rate b (sm)'] = gr_sm_b
        df['Growth rate f (sm)'] = gr_sm_f
        df['Specific GR b (sm)'] = gr_sm_b / df['Volume (sm)']
        df['Specific GR f (sm)'] = gr_sm_f / df['Volume (sm)']
        df['Age'] = (df['Frame']-df['Frame'].min()) * 12.
        
        # G1 annotations
        g1_frame = g1_anno.loc[basalID]['Frame']
        if g1_frame == '?':
            continue
        else:
            g1_frame = int(g1_frame)
            df['G1S frame'] = g1_frame
            df['Phase'] = 'G1'
            df.loc[df['Frame'].values > g1_frame,'Phase'] = 'SG2'
            
    collated[basalID] = df

#%
#@todo: daughter/division voluem analysis!

#%

with open(path.join(dirname,'basal_no_daughters.pkl'),'wb') as f:
    pkl.dump(collated,f)



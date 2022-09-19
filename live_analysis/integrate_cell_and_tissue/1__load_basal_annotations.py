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

basal_tracking = io.imread(path.join(dirname,'manual_basal_tracking/basal_tracks.tif'))
allIDs = np.unique(basal_tracking)[1:]

#%% Do pixel level measurements

collated = {k:pd.DataFrame() for k in allIDs}
for t,im in enumerate(basal_tracking):

    properties = measure.regionprops(im, extra_properties = [surface_area])
    
    for p in properties:
        
        basalID = p['label']
        V = p['area'] * dx**2
        Z,Y,X = p['centroid']
        SA = p['surface_area'] * dx**2
        
        I = p['inertia_tensor']
        Iaxial, phi, Ia, Ib, theta = parse_3D_inertial_tensor(I)
        s = pd.Series({'basalID': basalID
                       ,'Volume':V
                       ,'Z':Z,'Frame': t
                       ,'Y-pixels':Y,'X-pixels':X
                       ,'Y':Y * dx**2,'X':X * dx**2
                       ,'Surface area':SA
                       ,'Axial angle':phi
                       ,'Axial component':Iaxial
                       ,'Planar component 1':Ia,'Planar component 2':Ib
                       ,'Planar angle':theta
                       ,'Apical area':np.nan
                       ,'Basal area':np.nan
                       ,'Basal orientation':np.nan
                       ,'Collagen orientation':np.nan})
        
        collated[basalID] = collated[basalID].append(s,ignore_index=True)

#%% Load "flattened" segmenations to look at apical v. basal area
# Load "collagen orientation + fibrousness image"

for t in range(T):
    f = path.join(dirname,f'Image flattening/flat_basal_tracking/t{t}.tif')
    im = io.imread(f)
    
    properties = measure.regionprops(im, extra_properties = [surface_area])
    for p in properties:
        
        basalID = p['label']
        bbox = p['bbox']
        Z_top = bbox[0]
        Z_bottom = bbox[3]
        
        mask = im == basalID
        apical_area = mask[Z_top:Z_top+3,...].max(axis=0)
        apical_area = apical_area.sum()
        
        basal_mask = mask[Z_bottom-3:Z_bottom,...]
        basal_area = basal_mask.max(axis=0).sum()
    
        basal_mask = basal_mask.max(axis=0)
        basal_orientation = measure.regionprops(basal_mask.astype(int))[0]['orientation']
        basal_orientation = np.rad2deg(basal_orientation)
        # Constran aingles to be > 0
        if basal_orientation < 0:
            basal_orientation + 180
                                                            
        idx = collated[basalID].index[collated[basalID]['Frame'] == t][0]
        collated[basalID].at[idx,'Apical area'] = apical_area * dx**2
        collated[basalID].at[idx,'Basal area'] = basal_area * dx**2
        
        collated[basalID].at[idx,'Basal orientation'] = basal_orientation
        
        f = path.join(dirname,f'Image flattening/collagen_orientation/t{t}.npy')
        im_ori = np.load(f)
        collagen_ori = im_ori[basal_mask]
        if collagen_ori < 0:
            collagen_ori + 180
        collated[basalID].at[idx,'Collagen orientation'] = collagen_ori
    

#%% Calculate spline + growth rates + save

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
        df['Growth rate b'] = gr_f / 12.
        df['Growth rate f'] = gr_b / 12.
        df['Growth rate b (sm)'] = gr_sm_b / 12.
        df['Growth rate f (sm)'] = gr_sm_f / 12.
        df['Specific GR b (sm)'] = gr_sm_b / df['Volume (sm)']
        df['Specific GR f (sm)'] = gr_sm_f / df['Volume (sm)']
        df['Age'] = (df['Frame']-df['Frame'].min()) * 12.
        df['Collagen alignment'] = np.abs(df['Collagen orientation'] - df['Basal orientation'])
        
        # G1 annotations
        g1_frame = g1_anno.loc[basalID]['Frame']
        if g1_frame == '?':
            continue
        else:
            g1_frame = int(g1_frame)
            df['G1S frame'] = g1_frame
            df['Phase'] = 'G1'
            df.loc[df['Frame'].values > g1_frame,'Phase'] = 'SG2'
            df['Time to G1S'] = df['Age'] - df['G1S frame']* 12
            
    collated[basalID] = df

#%
#@todo: daughter/division voluem analysis!

#%

with open(path.join(dirname,'basal_no_daughters.pkl'),'wb') as f:
    pkl.dump(collated,f)


#%% Visualize somethings

df = pd.concat(collated)

for t in range(T):
    t=9
    
    seg = basal_tracking[t,...]
    df_ = df[df['Frame'] == t]
    colorized = colorize_segmentation(seg,
                                      {k:v for k,v in zip(df_['basalID'].values,df_['Collagen orientation'].values)}
                                      ,dtype=int)
    io.imsave(path.join(dirname,f'3d_nuc_seg/Collagen_orientation/t{t}.tif'),colorized,check_contrast=False)
      
    
    
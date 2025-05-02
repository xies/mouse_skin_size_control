#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:08:18 2025

@author: xies
"""

# Core libraries
import numpy as np
from skimage import io
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pylab as plt

# Specific utils
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, \
    most_likely_label, colorize_segmentation
from mathUtils import get_neighbor_idx, surface_area, parse_3D_inertial_tensor, \
    argsort_counter_clockwise
from toeplitzDifference import backward_difference, forward_difference, central_difference
import pyvista as pv

# General utils
from tqdm import tqdm
from os import path
from basicUtils import nonans

dx = 0.25
dz = 1

# Filenames
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

all_df = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'))

def get_interpolated_growth_curve(cf,field='Cell volume',smoothing_factor=1e10):

    cell_types = cf['Cell type'].unique()
    for celltype in cell_types:
        I = cf['Cell type'] == celltype
        this_type = cf.loc[I]
        if len(this_type) < 4:
            yhat = this_type[field].values
            dydt = np.ones(len(this_type)) * np.nan        
        else:
            t = np.array(range(0,len(this_type))) * 12
            v = this_type[field].values
            # Spline smooth
            spl = UnivariateSpline(t, v, k=3, s=smoothing_factor)
            yhat = spl(t)
            dydt = spl.derivative(n=1)(t)
            
        cf.loc[I,f'{field} smoothed'] = yhat
        cf.loc[I,f'{field} smoothed growth rate'] = dydt

    return cf

def get_instantaneous_growth_rate(cf,field='Cell volume',time_field='Time'):
    
    assert(field == 'Cell volume' or field == 'Nuclear volume')
    
    #@todo: detect + impute NaN for automatically tracked cells
    # Group by different cell types
    cell_types = cf['Cell type'].unique()
    
    for celltype in cell_types:
    
        I = cf['Cell type'] == celltype
        if len(cf.loc[I]) < 3:
            gr_b = np.ones( len(cf.loc[I]) ) * np.nan
            gr_f = np.ones( len(cf.loc[I]) ) * np.nan
            gr_c = np.ones( len(cf.loc[I]) ) * np.nan
            gr_sm_b = np.ones( len(cf.loc[I]) ) * np.nan
            gr_sm_f = np.ones( len(cf.loc[I]) ) * np.nan
            gr_sm_c = np.ones( len(cf.loc[I]) ) * np.nan
        else:
            v = cf.loc[I,field].values
            v_sm = cf.loc[I,f'{field} smoothed'].values
            t = cf.loc[I,time_field].values
            
            Tb = backward_difference(len(v))
            Tf = forward_difference(len(v))
            gr_b = np.dot(Tb,v) / t
            gr_f = np.dot(Tf,v) / t
            
            Tc = central_difference(len(v))
            gr_c = np.dot(Tc,v) / t
            
            Tb = backward_difference(len(v_sm))
            Tf = forward_difference(len(v_sm))
            Tf = central_difference(len(v_sm))
            
            
            gr_sm_b = np.dot(Tb,v_sm) / t
            gr_sm_f = np.dot(Tf,v_sm) / t
            gr_sm_c = np.dot(Tc,v_sm) / t
        
        gr_b[0] = np.nan
        gr_f[-1] = np.nan
        gr_sm_b[0] = np.nan
        gr_sm_f[-1] = np.nan
        
        gr_c[0] = np.nan
        gr_c[-1] = np.nan
        gr_sm_c[0] = np.nan
        gr_sm_c[-1] = np.nan
        
        cf.loc[I,f'{field} rowth rate b'] = gr_f
        cf.loc[I,f'{field} rate f'] = gr_b
        cf.loc[I,f'{field} rate c'] = gr_c
        cf.loc[I,f'{field} rate b (sm)'] = gr_sm_b
        cf.loc[I,f'{field} rate f (sm)'] = gr_sm_f
        cf.loc[I,f'{field} rate c (sm)'] = gr_sm_c
        
    return cf

exp_model = lambda x,p1,p2 : p1 * np.exp(p2 * x)
def get_exponential_growth_rate(cf,field='Cell volume', time_field='Time'):
    
    cell_types = cf['Cell type'].unique()
    for celltype in cell_types:
        I = cf['Cell type'] == celltype
        I = I & cf[field].isna()
        
        if len(cf.loc[I]) < 4:
            cf.loc[I,f'{field} exponential growth rate'] = np.nan
        else:
            
            t = cf.loc[I][time_field].values
            y = cf.loc[I][field].values
            
            p,_ = curve_fit(exp_model,t,y,p0=[y[0],1],
                                     bounds = [ [0,0],
                                       [y.max(),np.inf]])
            V0,gamma = p
            cf.loc[I,f'{field} exponential growth rate'] = gamma
    
    return cf

#%%

all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'),index_col=['Frame','TrackID'])
all_df = all_df.reset_index()
all_df['Time'] = all_df['Frame'] * 12
tracks = [ _df for _,_df in all_df.groupby('TrackID')]

for i,track in enumerate(tracks):
    
    track = get_interpolated_growth_curve(track, field='Nuclear volume')
    track = get_instantaneous_growth_rate(track, field='Nuclear volume')
    track = get_exponential_growth_rate(track, field='Nuclear volume')
    
    tracks[i] = track








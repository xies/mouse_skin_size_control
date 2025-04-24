#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:08:18 2025

@author: xies
"""

# Core libraries
import numpy as np
from skimage import io, measure, draw, util, morphology, transform, img_as_bool
from scipy.spatial import distance, Voronoi, Delaunay
import pandas as pd
import matplotlib.pylab as plt

# 3D mesh stuff
from aicsshparam import shtools, shparam
from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, \
    discrete_mean_curvature_measure, sphere_ball_intersection

# Specific utils
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, \
    most_likely_label, colorize_segmentation
from mathUtils import get_neighbor_idx, surface_area, parse_3D_inertial_tensor, \
    argsort_counter_clockwise
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

def get_interpolated_curve(cf,field='Volume',smoothing_factor=1e10):

    # Get rid of daughter cells]
    if len(cf) < 4:
        yhat = cf[field].values
        dydt = np.ones(len(cf)) * np.nan
        
    else:
        t = np.array(range(0,len(cf))) * 12
        v = cf[field].values
        # Spline smooth
        spl = UnivariateSpline(t, v, k=3, s=smoothing_factor)
        yhat = spl(t)
        
        dydt = spl.derivative(n=1)(t)
        
        # # Nuclear volume
        # nv = cf.Nucleus.values
        # # Spline smooth
        # spl = UnivariateSpline(t, nv, k=3, s=smoothing_factor)
        # nuc_hat = spl(t)

    return yhat,dydt

def get_growth_rate(cf,field='Volume',time_field='Time'):
    
    assert(field == 'Nucleus' or field == 'Volume')
    
    #@todo: detect + impute NaN for automatically tracked cells
    #@todo: Filter out daughters
    
    v = cf[field].values
    v_sm = cf[field + ' (sm)'].values
    t = cf[time_field].values
    
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
    
    return gr_b,gr_f,gr_c,gr_sm_b,gr_sm_f,gr_sm_c

exp_model = lambda x,p1,p2 : p1 * np.exp(p2 * x)

def get_exponential_growth_rate(df,field='Volume (sm)', time_field='Age'):
    
    if len(df) < 3:
        return np.nan
    
    t = df[time_field].values
    y = df[field].values
    
    p,_ = curve_fit(exp_model,t,y,p0=[y[0],1],
                             bounds = [ [0,0],
                               [y.max(),np.inf]])
    V0,gamma = p
    
    return gamma

#%%

tracks = [ _df for _,_df in all_df.groupby('TrackID')]

track = tracks[0]

# Instantaneous growth rates
gr_f,gr_b,gr_c,gr_sm_b,gr_sm_f,gr_sm_c = get_growth_rate(df,'Volume')
df['Growth rate spl'] = dVdt
df['Growth rate b'] = gr_f
df['Growth rate f'] = gr_b
df['Growth rate c'] = gr_c
df['Growth rate b (sm)'] = gr_sm_b
df['Growth rate f (sm)'] = gr_sm_f
df['Growth rate c (sm)'] = gr_sm_c
df['Specific GR b (sm)'] = gr_sm_b / df['Volume (sm)']
df['Specific GR f (sm)'] = gr_sm_f / df['Volume (sm)']
df['Specific GR c (sm)'] = gr_sm_c / df['Volume (sm)']
df['Specific GR spl'] = dVdt / df['Volume (sm)']
df.loc[df['Daughter'],'Growth rate b'] = np.nan
df.loc[df['Daughter'],'Growth rate f'] = np.nan

# Exponential fits
gamma = get_exponential_growth_rate(df)
df['Exponential growth rate'] = gamma
nuc_gamma = get_exponential_growth_rate(df,field='Nuclear volume (sm)')
df['Exponential nuc growth rate'] = nuc_gamma
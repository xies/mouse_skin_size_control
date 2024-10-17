#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:24:28 2022

@author: xies
"""

import numpy as np
from skimage import io, measure, morphology
from scipy.interpolate import UnivariateSpline
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

from mathUtils import surface_area, parse_3D_inertial_tensor
from imageUtils import colorize_segmentation

from os import path,makedirs
from tqdm import tqdm
import pickle as pkl

dirname = dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
# dirname = '/Users/xies/Desktop/Code/mouse_skin_size_control/2024_analysis/test_dataset/'

print(f'--- Working on {dirname} ---')

ZZ = 72
XX = 460
T = 15
dx = 0.25
dz = 1

from toeplitzDifference import backward_difference,forward_difference,central_difference

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

#%% Load the basal cell tracking and measure from the basal cortical tracking only

DEMO = False

if DEMO:
    basal_tracking = io.imread(path.join(dirname,'example_mouse_skin_image.tif'))[:,:,5,:,:]
else:
    basal_tracking = io.imread(path.join(dirname,'manual_basal_tracking/basal_tracks_cyto.tif'))

allIDs = np.unique(basal_tracking)[1:]

#% Do pixel level measurements e.g. Surface Area

collated = {k:[] for k in allIDs}

for t,im in tqdm(enumerate(basal_tracking)):

    nuc_mask = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif')) > 0
    properties = measure.regionprops(im, intensity_image = nuc_mask, extra_properties = [surface_area]
                                     , spacing = [dz,dx,dx])
    
    for p in properties:
        
        basalID = p['label']
        V = p['area'] * dx**2
        Z,Y,X = p['centroid']
        SA = p['surface_area'] * dx**2
        
        I = p['inertia_tensor']
        Iaxial, phi, Ia, Ib, theta = parse_3D_inertial_tensor(I)
        
        nuc_vol = p['mean_intensity']*p['area']
        
        s = {'basalID': basalID
                       ,'Daughter':False
                       ,'Volume':V
                       ,'Z':Z,'Frame': t,'Time':t*12
                       ,'Y-pixels':Y/dx,'X-pixels':X/dx
                       ,'Y':Y,'X':X
                       ,'Surface area':SA
                       ,'Axial angle':phi
                       ,'Axial component':Iaxial
                       ,'Planar component 1':Ia,'Planar component 2':Ib
                       ,'Nuclear volume':nuc_vol
                       ,'Planar angle':theta
                       ,'Apical area':np.nan
                       ,'Basal area':np.nan
                       ,'Basal orientation':np.nan
                       ,'Collagen orientation':np.nan
                       ,'Collagen fibrousness':np.nan}
        
        collated[basalID].append(s)
        

collated = {basalID: pd.DataFrame(cell) for basalID,cell in collated.items()}

#%% Load "flattened" cortical segmenations to look at apical v. basal area
# E.g. collagen orientation + fibrousness

for t in tqdm(range(T)):

    if DEMO:
        im = io.imread(path.join(dirname,'flat_tracked_cells.tif'))[t,...]
    else:
        f = path.join(dirname,f'Image flattening/flat_basal_tracking/t{t}.tif')
        im = io.imread(f)
    
    if DEMO:
        Gx,Gy = io.imread(path.join(dirname,'collagen_gradients.tif'))[t,...]
    else:
        # Load the structuring matrix elements
        f = path.join(dirname,f'Image flattening/collagen_orientation/t{t}.npy')
        [Gx,Gy] = np.load(f)
    Jxx = Gx*Gx
    Jxy = Gx*Gy
    Jyy = Gy*Gy
    
    properties = measure.regionprops(im, extra_properties = [surface_area])
    for p in properties:
        
        basalID = p['label']
        bbox = p['bbox']
        Z_top = bbox[0]
        Z_bottom = bbox[3]
        
        mask = im == basalID
        apical_area = mask[Z_top:Z_top+3,...].max(axis=0)
        apical_area = apical_area.sum()
        
        # mid-level area
        mid_area = mask[np.round((Z_top+Z_bottom)/2).astype(int),...].sum()
        
        basal_mask = mask[Z_bottom-3:Z_bottom,...]
        basal_area = basal_mask.max(axis=0).sum()
    
        basal_mask = basal_mask.max(axis=0)
        #NB: skimage uses the 'vertical' as the orientation axis
        basal_orientation = measure.regionprops(basal_mask.astype(int))[0]['orientation']
        # Need to 'convert to horizontal--> subtract 90-deg from image
        basal_orientation = np.rad2deg(basal_orientation + np.pi/2)
        idx = collated[basalID].index[collated[basalID]['Frame'] == t][0]
        collated[basalID].at[idx,'Apical area'] = apical_area * dx**2
        collated[basalID].at[idx,'Basal area'] = basal_area * dx**2
        
        collated[basalID].at[idx,'Basal orientation'] = basal_orientation
        
        # Subtract the mid-area of central cell from the coronal area
        collated[basalID].at[idx,'Middle area'] = mid_area * dx**2
        
        # Characteristic matrix of collagen signal
        J = np.matrix( [[Jxx[basal_mask].sum(),Jxy[basal_mask].sum()],[Jxy[basal_mask].sum(),Jyy[basal_mask].sum()]] )
        l,D = np.linalg.eig(J) # NB: not sorted
        order = np.argsort(l)[::-1] # Ascending order
        l = l[order]
        D = D[:,order]
        
        # Orientation
        theta = np.rad2deg(np.arctan(D[1,0]/D[0,0]))
        fibrousness = (l[0] - l[1]) / l.sum()
        
        # theta = np.rad2deg( -np.arctan( 2*Jxy[basal_mask].sum() / (Jyy[basal_mask].sum()-Jxx[basal_mask].sum()) )/2 )
        # # Fibrousness
        # fib = np.sqrt((Jyy[basal_mask].sum() - Jxx[basal_mask].sum())**2 + 4 * Jxy[basal_mask].sum()) / \
        #     (Jxx[basal_mask].sum() + Jyy[basal_mask].sum())
        collated[basalID].at[idx,'Collagen orientation'] = theta
        collated[basalID].at[idx,'Collagen fibrousness'] = fibrousness
    
df = pd.concat(collated,ignore_index=True)

#%% Calculate spline + growth rates + save

if DEMO:
    g1_anno = pd.read_csv(path.join(dirname,'g1_frame.txt'),index_col=0)
else:
    g1_anno = pd.read_csv(path.join(dirname,'2020 CB analysis/tracked_cells/g1_frame.txt'),index_col=0)

for basalID, df in collated.items():
    
    # put in the birth volume
    df['Birth volume'] = df['Volume'].values[0]
    
    df['Phase'] = '?'
    df['Age'] = (df['Time']-df['Time'].min())
    
    # G1 annotations
    g1_frame = g1_anno.loc[basalID]['Frame'] #NB: 1-indexed!
    if not g1_frame == '?':
        g1_frame = int(g1_frame)
        df['G1S frame'] = g1_frame
        df['Phase'] = 'G1'
        df.loc[df['Frame'].values >= g1_frame,'Phase'] = 'SG2'
        df['Time to G1S'] = df['Age'] - df['G1S frame']* 12
    else:
        df['G1S frame'] = np.nan
        df['Time to G1s'] = np.nan
    
    # Derivations
    df['Axial eccentricity'] = df['Axial component'] / df['Planar component 2']
    df['Planar eccentricity'] = df['Planar component 2'] / df['Planar component 1']
    df['SA to vol'] = df['Surface area'] / df['Volume']
    
    # Spline-smoothing
    Vsm,dVdt = get_interpolated_curve(df)
    df['Volume (sm)'] = Vsm
    NVsm,dVdt = get_interpolated_curve(df,field='Nuclear volume')
    df['Nuclear volume (sm)'] = NVsm
    
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
    
    df_g1_only = df[df['Phase'] == 'G1']
    gamma_g1s = get_exponential_growth_rate(df_g1_only)
    nuc_gamma_g1s = get_exponential_growth_rate(df_g1_only, field='Nuclear volume (sm)')
    df['Exponential growth rate G1 only'] = gamma_g1s
    df['Exponential nuc growth rate G1 only'] = nuc_gamma_g1s
    
    # Collagen
    cos = np.cos(df['Collagen orientation'] - df['Basal orientation'])
    df['Collagen alignment'] = np.abs(cos) #alignment + anti-alignment are the same
    
    collated[basalID] = df

with open(path.join(dirname,'basal_no_daughters.pkl'),'wb') as f:
    pkl.dump(collated,f)
df = pd.concat(collated,ignore_index=True)

#%% Load the daughter cells + save
#NB: won't work with the demo dataset

# Load into a dict of individual cells (some daughter cell pairs have >1 time points)
daughter_tracking = io.imread(path.join(dirname,'manual_basal_tracking_lineage/basal_tracking_daughters_cyto.tif'))
daughterIDs = np.unique(daughter_tracking)[1:]
daughters = {k:list() for k in daughterIDs}

for t,im in enumerate(daughter_tracking):

    properties = measure.regionprops(im)
    if len(properties) > 0:
        for p in properties:
            basalID = p['label']
            V = p['area'] * dx**2
            s = {'basalID': basalID,'Daughter volume':V}
            daughters[basalID].append(s)

for k,v in daughters.items():
    daughters[k] = pd.DataFrame(v)

# Trim to only first daughter time point
for basalID,daughter in daughters.items():
    if len(daughter) > 0:
        daughters[basalID] = daughter.iloc[0]
    last_time = collated[basalID]['Time'].max()
    collated[basalID] = pd.concat(( collated[basalID],pd.DataFrame({'basalID':basalID,
                                                            'Time':last_time + 12,
                                                            'Volume':daughter['Daughter volume'].values[0],
                                                            'Daughter':True
                                                            },index=['Daughter']) ))

with open(path.join(dirname,'basal_with_daughters.pkl'),'wb') as f:
    pkl.dump(collated,f)
    
df = pd.concat(collated,ignore_index=True)

 #%% Visualize somethings

df = pd.concat(collated,ignore_index=True)

for t in tqdm(range(T)):
        
    seg = basal_tracking[t,...]
    df_ = df[df['Frame'] == t]
    colorized = colorize_segmentation(seg,
                                      {k:v for k,v in zip(df_['basalID'].values,df_['Collagen orientation'].values)}
                                      ,dtype=int)
    io.imsave(path.join(dirname,f'3d_nuc_seg/Collagen_orientation/t{t}.tif'),colorized.astype(np.uint16),check_contrast=False)
    
    colorized = colorize_segmentation(seg,
                                      {k:v for k,v in zip(df_['basalID'].values,df_['Basal orientation'].values)}
                                      ,dtype=int)
    io.imsave(path.join(dirname,f'3d_nuc_seg/basal_orientation/t{t}.tif'),
              colorized.astype(np.uint16),check_contrast=False)
    
    colorized = colorize_segmentation(seg,
                                      {k:v for k,v in zip(df_['basalID'].values,df_['Collagen alignment'].values)}
                                      ,dtype=float)
    io.imsave(path.join(dirname,f'3d_nuc_seg/collagen_alignment/t{t}.tif'),
              util.img_as_uint(colorized),check_contrast=False)
     
    
    
    
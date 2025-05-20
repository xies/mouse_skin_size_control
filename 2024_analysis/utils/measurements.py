#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 22:51:08 2025

@author: xies
"""

import numpy as np
from skimage import measure, img_as_bool, transform
import pandas as pd
from functools import reduce

from mathUtils import get_neighbor_idx, parse_3D_inertial_tensor, argsort_counter_clockwise
# 3D mesh stuff
from aicsshparam import shtools, shparam
from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, \
    discrete_mean_curvature_measure, sphere_ball_intersection
import pyvista as pv

# from toeplitzDifference import backward_difference, forward_difference, central_difference
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

#%% Calculation functions

# convert trianglulation to adjacency matrix (for easy editing)
def tri_to_adjmat(tri):
    num_verts = max(map(max,tri.simplices)) + 1
    A = np.zeros((num_verts,num_verts),dtype=bool)
    for idx in range(num_verts):
        neighbor_idx = get_neighbor_idx(tri,idx)
        A[idx,neighbor_idx] = True
    return A

# Find distance to nearest manually annotated point in points-list
def find_distance_to_closest_point(dense_3d_coords,annotation_coords_3d,spacing=[1,1,1]):
    distances = np.zeros(len(dense_3d_coords))
    dz,dx,_ = spacing

    for i,row in dense_3d_coords.iterrows():
        dx = row['X'] - annotation_coords_3d['X']
        dy = row['Y'] - annotation_coords_3d['Y']
        dz = row['Z'] - annotation_coords_3d['Z']
        D = np.sqrt(dx**2 + dy**2 + dz**2)
        
        distances[i] = D.min()
            
    return distances

def measure_nuclear_geometry_from_regionprops(nuc_labels, spacing = [1,1,1]):
    dz,dx,_ = spacing
    df = pd.DataFrame( measure.regionprops_table(nuc_labels,
                                                 properties=['label','area','solidity','bbox','centroid'],
                                                 spacing = [dz,dx,dx],
                                                 ))
    df = df.rename(columns={
                            'label':'TrackID',
                            'area':'Nuclear volume',
                            'bbox-0':'Nuclear bbox top',
                            'bbox-3':'Nuclear bbox bottom',
                            'solidity':'Nuclear solidity',
                            'centroid-0':'Z',
                            'centroid-1':'Y',
                            'centroid-2':'X',
                            })
    df = df.set_index('TrackID')
    
    return df

def measure_cyto_geometry_from_regionprops(cyto_dense_seg, spacing= [1,1,1]):
    # Measurements from cortical segmentation

    df = pd.DataFrame( measure.regionprops_table(cyto_dense_seg,
                                                      properties=['label','area','centroid'],
                                                      spacing=spacing,
                                                      ))
    df = df.rename(columns={'area':'Cell volume',
                            'label':'TrackID',
                            'centroid-0':'Z-cyto',
                            'centroid-1':'Y-cyto',
                            'centroid-2':'X-cyto',
                            })
    df = df.set_index('TrackID')
    for p in measure.regionprops(cyto_dense_seg, spacing=spacing):
        i = p['label']
        I = p['inertia_tensor']
        Iaxial, phi, Ia, Ib, theta = parse_3D_inertial_tensor(I)
        df.loc[i,'Axial component'] = Iaxial
        df.loc[i,'Planar component 1'] = Ia
        df.loc[i,'Planar component 2'] = Ib
        df.loc[i,'Axial angle'] = phi
        df.loc[i,'Planar angle'] = theta
    
    return df

def measure_cyto_intensity(cyto_dense_seg, intensity_image_dict:dict, spacing = [1,1,1]):
    
    df = []
    for chan_name,im in intensity_image_dict.items():
        _df = pd.DataFrame( measure.regionprops_table(cyto_dense_seg,intensity_image=im,
                                                      properties=['label','area','mean_intensity']))
        _df = _df.rename(columns={'mean_intensity':f'Mean {chan_name} intensity',
                                  'label':'TrackID'})
        _df[f'Total {chan_name} intensity'] = _df['area'] * _df[f'Mean {chan_name} intensity']
        _df = _df.drop(labels=['area'], axis='columns')
        _df = _df.set_index('TrackID')
        df.append(_df)
        
    df = reduce(lambda x,y: pd.merge(x,y,on='TrackID'), df)
    
    return df
    

def get_mask_slices(prop, border = 0, max_dims = None):
    
    zmin,ymin,xmin,zmax,ymax,xmax = prop['bbox']
    zmin = max(0,zmin - border)
    ymin = max(0,ymin - border)
    xmin = max(0,xmin - border)
    
    (ZZ,YY,XX) = max_dims
    zmax = min(ZZ,zmax + border)
    ymax = min(YY,ymax + border)
    xmax = min(XX,xmax + border)
    
    slc = [slice(None)] * 3
    
    slc[0] = slice( int(zmin),int(zmax))
    slc[1] = slice(int(ymin),int(ymax))
    slc[2] = slice(int(xmin),int(xmax))
    
    return slc

def estimate_sh_coefficients(cyto_seg, lmax, spacing = [1,1,1]):
    
    ZZ,YY,XX = cyto_seg.shape
    dz,dx,_ = spacing
    
    slices = {p['label']: get_mask_slices(p, border=1, max_dims=(ZZ,YY,XX))
              for p in measure.regionprops(cyto_seg)}
    cyto_masks = {label: (cyto_seg == label)[tuple(_s)] for label,_s in slices.items()}
    cyto_masks = {label: img_as_bool(
                transform.resize(mask, mask.shape*np.array([dz/dx,1,1])))
                  for label, mask in cyto_masks.items()}
    
    # @todo: currently, does not align 'z' axis because original AICS project was
    # for in vitro cells. Should extend codebase to align fully in 3d later
    # Currently, there is small amount of z-tilt, so probably OK to stay in lab frame
    aligned_masks = {label: np.squeeze(shtools.align_image_2d(m)[0])
                     for label,m in cyto_masks.items()}
    
    # Parametrize with SH coefficients and record
    sh_coefficients = []
    for label,mask in aligned_masks.items():
        (coeffs,_),_ = shparam.get_shcoeffs(image=mask, lmax=lmax)
        coeffs['TrackID'] = label
        M = shtools.convert_coeffs_dict_to_matrix(coeffs,lmax=lmax)
        mesh = shtools.get_even_reconstruction_from_coeffs(M)[0]
        coeffs['shcoeffs_surface_area'] = pv.wrap(mesh).area
        # coeffs['shcoeffs_volume'] = pv.wrap(mesh).volume # This is just L0M0
        
        # verts,faces,_,_ = measure.marching_cubes(mask)
        # mesh_mc = Trimesh(verts,faces)
        # coeffs['mc_surface_area'] = pv.wrap(mesh_mc).area
        # coeffs['mc_volume'] = pv.wrap(mesh_mc).volume
        
        sh_coefficients.append(coeffs)
        
    sh_coefficients = pd.DataFrame(sh_coefficients)
    # Drop all 0s
    # sh_coefficients = sh_coefficients.loc[:,~np.all(sh_coefficients==0, axis=0)]
    
    return sh_coefficients


def measure_flat_cyto_from_regionprops(flat_cyto, jacobian, spacing= [1,1,1]):
    '''
    
    PARAMETERS
    
    flat_cyto: flattened cytoplasmic segmentation
    jacobian: jacobian matrix of the gradient image of collagen signal
    spacing: default = [1,1,1] pixel sizes in microns
    
    '''
    
    
    # Measurements from cortical segmentation
    dz,dx,dx = spacing
    
    df = pd.DataFrame( measure.regionprops_table(flat_cyto,
                                                properties=['label'],
                                                spacing=[dz,dx,dx],
                                                ))
    df = df.rename(columns={'label':'TrackID'})
    df = df.set_index('TrackID')
    
    _,YY,XX = flat_cyto.shape
    basal_masks_2save = np.zeros((YY,XX))
    
    for p in measure.regionprops(flat_cyto, spacing=[dz,dx,dx]):
        i = p['label']
        bbox = p['bbox']
        Z_top = bbox[0]
        Z_bottom = bbox[3]
        
        mask = flat_cyto == i
        
        # Apical area (3 top slices)
        apical_area = mask[Z_top:Z_top+3,...].max(axis=0)
        apical_area = apical_area.sum()
        
        # mid-level area
        mid_area = mask[np.round((Z_top+Z_bottom)/2).astype(int),...].sum()
        
        # Apical area (3 bottom slices)
        basal_mask = mask[Z_bottom-4:Z_bottom,...]
        basal_mask = basal_mask.max(axis=0)
        basal_area = basal_mask.sum()
    
        basal_orientation = measure.regionprops(basal_mask.astype(int))[0]['orientation']
        basal_eccentricity = measure.regionprops(basal_mask.astype(int))[0]['eccentricity']
        
        df.at[i,'Apical area'] = apical_area * dx**2
        df.at[i,'Basal area'] = basal_area * dx**2
        df.at[i,'Basal orientation'] = basal_orientation
        df.at[i,'Basal eccentricity'] = basal_eccentricity
        df.at[i,'Middle area'] = mid_area * dx**2
        df.at[i,'Height'] = (Z_bottom-Z_top)*dz
        
        basal_masks_2save[basal_mask] = i
        
        # Characteristic matrix of collagen signal
        Jxx = jacobian[0]
        Jyy = jacobian[1]
        Jxy = jacobian[2]
        
        J = np.matrix( [[Jxx[basal_mask].sum(),Jxy[basal_mask].sum()],
                        [Jxy[basal_mask].sum(),Jyy[basal_mask].sum()]] )
        
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
        df.at[i,'Collagen orientation'] = theta
        df.at[i,'Collagen fibrousness'] = fibrousness
        df.at[i,'Collagen alignment'] = np.abs(np.cos(theta - basal_orientation))
        
    return df,basal_masks_2save


def detect_missing_frame_and_fill(cf):
    
    cf = cf.reset_index(drop=True)
    frames = cf['Frame'].values
    cf = cf.set_index('Frame')
    t = np.arange(frames[0],frames[-1]+1)
    missing_frames = set(t) - set(frames)

    fill = pd.DataFrame(np.nan, columns=cf.columns, index=list(missing_frames))
    fill.index.name = 'Frame'
    #clean up object types
    str_cols = cf.columns[cf.dtypes == object]
    bool_cols = cf.columns[cf.dtypes == bool]
    for col in str_cols:
        fill[col] = 'NA'
    for col in bool_cols:
        fill[col] = False
    fill['TrackID'] = cf.iloc[0]['TrackID']
    cf_filled = pd.concat((cf,fill))
    cf_filled = cf_filled.sort_index()
    
    return cf_filled.reset_index()
    

def get_interpolated_curve(cf,field='Cell volume',smoothing_factor=1e10):

    cell_types = cf['Cell type'].unique()
        
    for celltype in cell_types:
        I = cf['Cell type'] == celltype
        I = I & ~cf[field].isna() & ~cf['Border']
        
        this_type = cf.loc[I]
        if len(this_type) < 4:
            yhat = this_type[field].values
            dydt = np.ones(len(this_type)) * np.nan        
        else:
            t = np.array(range(0,len(this_type))) * 12
            v = this_type[field].values
            # Spline smooth
            spl = UnivariateSpline(t, v, k=2, s=smoothing_factor)
            yhat = spl(t)
            dydt = spl.derivative(n=1)(t)
            
        cf.loc[I,f'{field} smoothed'] = yhat
        cf.loc[I,f'{field} smoothed growth rate'] = dydt

    return cf

# def get_instantaneous_growth_rate(cf,field='Cell volume',time_field='Time'):
    
#     assert(field == 'Cell volume' or field == 'Nuclear volume')
    
#     #@todo: detect + impute NaN for automatically tracked cells
#     # Group by different cell types
#     cell_types = cf['Cell type'].unique()
    
#     for celltype in cell_types:
    
#         I = cf['Cell type'] == celltype
# #        I = I & cf[field].isna() & ~cf['Border']
        
#         if len(cf.loc[I]) < 3:
#             gr_b = np.ones( len(cf.loc[I]) ) * np.nan
#             gr_f = np.ones( len(cf.loc[I]) ) * np.nan
#             gr_c = np.ones( len(cf.loc[I]) ) * np.nan
#             gr_sm_b = np.ones( len(cf.loc[I]) ) * np.nan
#             gr_sm_f = np.ones( len(cf.loc[I]) ) * np.nan
#             gr_sm_c = np.ones( len(cf.loc[I]) ) * np.nan
#         else:
#             v = cf.loc[I,field].values
#             v_sm = cf.loc[I,f'{field} smoothed'].values
#             t = cf.loc[I,time_field].values
            
#             Tb = backward_difference(len(v))
#             Tf = forward_difference(len(v))
#             gr_b = np.dot(Tb,v) / t
#             gr_f = np.dot(Tf,v) / t
            
#             Tc = central_difference(len(v))
#             print(t)
#             print(v)
#             print(np.dot(Tc,v))
#             gr_c = np.dot(Tc,v) / np.diff(t)
            
#             Tb = backward_difference(len(v_sm))
#             Tf = forward_difference(len(v_sm))
#             Tf = central_difference(len(v_sm))
            
            
#             gr_sm_b = np.dot(Tb,v_sm) / t
#             gr_sm_f = np.dot(Tf,v_sm) / t
#             gr_sm_c = np.dot(Tc,v_sm) / t
        
#             gr_b[0] = np.nan
#             gr_f[-1] = np.nan
#             gr_sm_b[0] = np.nan
#             gr_sm_f[-1] = np.nan
            
#             gr_c[0] = np.nan
#             gr_c[-1] = np.nan
#             gr_sm_c[0] = np.nan
#             gr_sm_c[-1] = np.nan
        
#         cf.loc[I,f'{field} rowth rate b'] = gr_f
#         cf.loc[I,f'{field} rate f'] = gr_b
#         cf.loc[I,f'{field} rate c'] = gr_c
#         cf.loc[I,f'{field} rate b (sm)'] = gr_sm_b
#         cf.loc[I,f'{field} rate f (sm)'] = gr_sm_f
#         cf.loc[I,f'{field} rate c (sm)'] = gr_sm_c
        
#     return cf

exp_model = lambda x,p1,p2 : p1 * np.exp(p2 * x)
def get_exponential_growth_rate(cf,field='Cell volume', time_field='Time'):
    
    cell_types = cf['Cell type'].astype(str).unique()
    for celltype in cell_types:
        I = cf['Cell type'] == celltype
        I = I & ~cf[field].isna() & ~cf['Border']
        
        if len(cf.loc[I]) < 4:
            cf.loc[I,f'{field} exponential growth rate'] = np.nan
        else:
            
            t = cf.loc[I][time_field].values
            y = cf.loc[I][field].values
            
            try:
                p,_ = curve_fit(exp_model,t,y,p0=[y[0],1],
                                     bounds = [ [0,0],
                                       [y.max(),np.inf]])
                V0,gamma = p
                cf.loc[I,f'{field} exponential growth rate'] = gamma
            except:
                cf.loc[I,f'{field} exponential growth rate'] = np.nan
    
    return cf


def plot_track(cf,field='Nuyclear volume',time='Time',celltypefield='Cell type',
               linestyle: dict={'Basal':'-','Suprabasal':'--'},color='b',alpha=1):
    cf = cf.reset_index()
    cell_types = cf[celltypefield].astype(str).unique()

    prev_type = None
    for celltype in cell_types:
        
        if celltype == 'nan' or celltype =='NA':
            continue
        if celltype=='Basal':
            label = cf.iloc[0].TrackID
        else:
            label = None
            
        I = cf[celltypefield] == celltype
        t = cf.loc[I,time].values
        v = cf.loc[I,field].values

        if prev_type is not None:
            #grab last time point and connect
            Ilast = np.where(cf['Cell type'] == prev_type)[0][-1]
            t = np.insert(t, 0, cf.iloc[Ilast][time])
            v = np.insert(v, 0, cf.iloc[Ilast][field])
            
        plt.plot(t,v,linestyle=linestyle[celltype],alpha=alpha,
                 label=label, color=color)

        prev_type = celltype
        
    plt.show()
    plt.title(f'Track {cf.iloc[0].TrackID}, Border: {np.any(cf.Border)}')




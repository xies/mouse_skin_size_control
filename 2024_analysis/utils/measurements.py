#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 22:51:08 2025

@author: xies
"""

import numpy as np
from skimage import measure, img_as_bool, transform, util, filters, exposure
import pandas as pd
from functools import reduce

from mathUtils import get_neighbor_idx, parse_3D_inertial_tensor, argsort_counter_clockwise
from imageUtils import get_mask_slices
# 3D mesh stuff
from aicsshparam import shtools, shparam
from trimesh import Trimesh
# from trimesh.curvature import discrete_gaussian_curvature_measure, \
#     discrete_mean_curvature_measure, sphere_ball_intersection
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
                                                 properties=['label','area','solidity',
                                                             'bbox','centroid'],
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
    df['Nuclear height'] = df['Nuclear bbox top'] - df['Nuclear bbox bottom']
    df.drop(columns=['Nuclear bbox top','Nuclear bbox bottom'])
    
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

from meshUtils import mask2mesh, mesh2mask, rotate_mesh

def get_rotated_cell_image(mask,normal):
    
    mesh = mask2mesh(mask)
    
    normal = normal / np.dot(normal,normal)
    # calculate the angle between the z-axis and the surface normal
    dot_product = normal[0,...]
    phi = np.arccos(dot_product) # radians
    
    mesh = rotate_mesh(mesh,phi)
    
    mask = mesh2mask(mesh,pixel_size=1)
    
    return mask

      
# def measure_flat_cyto_from_regionprops(cyto, mesh, collagen_image, jacobian, spacing= [1,1,1]):
#     '''
    
#     PARAMETERS
    
#     flat_cyto: flattened cytoplasmic segmentation
#     collagen_image: flattened Max int projection of collagen image
#     jacobian: jacobian matrix of the gradient image of collagen signal
#     spacing: default = [1,1,1] pixel sizes in microns
    
#     '''
    
#     # Measurements from cortical segmentation
#     dz,dx,dx = spacing
    
#     df = pd.DataFrame( measure.regionprops_table(cyto,
#                                                 properties=['label'],
#                                                 spacing=[dz,dx,dx],
#                                                 ))
#     df = df.rename(columns={'label':'TrackID'})
#     df = df.set_index('TrackID')
    
#     # _,YY,XX = flat_cyto.shape
    
#     # basal_masks_2save = np.zeros((YY,XX))
    
#     for p in measure.regionprops(cyto, spacing=[dz,dx,dx]):
        
#         i = p['label']
#         bbox = p['bbox']
#         Z_top = bbox[0]
#         Z_bottom = bbox[3]
#         centroid = np.expand_dims(np.array(p['centroid']),0) * spacing
        
#         # mask = flat_cyto == i
#         mask = get_rotated_cell_image
        
#         # Apical area (3 bottom slices)
#         basal_mask = mask[Z_bottom-4:Z_bottom,...]
#          = basal_mask.max(axis=0)
#         basal_area = basal_mask.sum()
        
        
        
#         # Find the 'normal vector'
#         (closest_points, distances, triangle_id) = mesh.nearest.on_surface(centroid)
        
    
        # basal_orientation = np.rad2deg(measure.regionprops(basal_mask.astype(int))[0]['orientation'])
        # basal_eccentricity = measure.re
    #     basal_masks_2save[basal_mask] = i
        
    #     # Characteristic matrix of collagen signal
    #     Jxx = jacobian[0]
    #     Jyy = jacobian[1]
    #     Jxy = jacobian[2]
        
    #     J = np.matrix( [[Jxx[basal_mask].sum(),Jxy[basal_mask].sum()],
    #                     [Jxy[basal_mask].sum(),Jyy[basal_mask].sum()]] )
        
    #     l,D = np.linalg.eig(J) # NB: not sorted
    #     order = np.argsort(l)[::-1] # Ascending order
    #     l = l[order]
    #     D = D[:,order]
        
    #     # Orientation
    #     theta = np.rad2deg(np.arctan(D[1,0]/D[0,0])) # in degrees
    #     mag_diff = (l[0] - l[1]) / l.sum()
        
    #     # theta = np.rad2deg( -np.arctan( 2*Jxy[basal_mask].sum() / (Jyy[basal_mask].sum()-Jxx[basal_mask].sum()) )/2 )
    #     # # Fibrousness
    #     # fib = np.sqrt((Jyy[basal_mask].sum() - Jxx[basal_mask].sum())**2 + 4 * Jxy[basal_mask].sum()) / \
    #     #     (Jxx[basal_mask].sum() + Jyy[basal_mask].sum())
    #     df.at[i,'Collagen orientation'] = theta
    #     df.at[i,'Collagen coherence'] = mag_diff
    #     df.at[i,'Basal alignment'] = np.abs(np.cos(theta - basal_orientation))
        
    #     # Also include collagen intensity
    #     # Normalize collagen
    #     collagen_image = exposure.equalize_hist(collagen_image)
    #     df.at[i,'Collagen intensity'] = np.mean(collagen_image[basal_mask])
        
    # return df,basal_masks_2save

def measure_flat_cyto_from_regionprops(flat_cyto, collagen_image, jacobian, spacing= [1,1,1]):
    '''
    PARAMETERS
    
    flat_cyto: flattened cytoplasmic segmentation
    collagen_image: flattened Max int projection of collagen image
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
    
        basal_orientation = np.rad2deg(measure.regionprops(basal_mask.astype(int))[0]['orientation'])
        basal_eccentricity = measure.regionprops(basal_mask.astype(int))[0]['eccentricity']
        
        df.at[i,'Apical area'] = apical_area * dx**2
        df.at[i,'Basal area'] = basal_area * dx**2
        df.at[i,'Basal orientation'] = basal_orientation
        df.at[i,'Basal eccentricity'] = basal_eccentricity
        df.at[i,'Middle area'] = mid_area * dx**2
        df.at[i,'Cell height'] = (Z_bottom-Z_top)*dz
        
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
        theta = np.rad2deg(np.arctan(D[1,0]/D[0,0])) # in degrees
        mag_diff = (l[0] - l[1]) / l.sum()
        
        # theta = np.rad2deg( -np.arctan( 2*Jxy[basal_mask].sum() / (Jyy[basal_mask].sum()-Jxx[basal_mask].sum()) )/2 )
        # # Fibrousness
        # fib = np.sqrt((Jyy[basal_mask].sum() - Jxx[basal_mask].sum())**2 + 4 * Jxy[basal_mask].sum()) / \
        #     (Jxx[basal_mask].sum() + Jyy[basal_mask].sum())
        df.at[i,'Collagen orientation'] = theta
        df.at[i,'Collagen coherence'] = mag_diff
        df.at[i,'Basal alignment'] = np.abs(np.cos(theta - basal_orientation))
        
        # Also include collagen intensity
        # Normalize collagen
        collagen_image = exposure.equalize_hist(collagen_image)
        df.at[i,'Collagen intensity'] = np.mean(collagen_image[basal_mask])
        
    return df,basal_masks_2save

def measure_collagen_structure(collagen_image,blur_sigma=3):
    
    #https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131814#acks
    
    im = util.img_as_float(collagen_image)
    
    im_blur = filters.gaussian(im,blur_sigma)
    Gx = filters.sobel_h(im_blur)
    Gy = filters.sobel_v(im_blur)
    
    Jxx = Gx*Gx
    Jxy = Gx*Gy
    Jyy = Gy*Gy
    
    return (Jxx,Jxy,Jyy)
    

def detect_missing_frame_and_fill(cf):
    
    cf = cf.reset_index(drop=True)
    frames = cf[('Frame')].values
    cf = cf.set_index(('Frame'))
    t = np.arange(frames[0],frames[-1]+1)
    missing_frames = set(t) - set(frames)
    if len(missing_frames) == 0:
        return cf.reset_index()
    
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

    # cell_types = cf['Cell type'].unique()
        
    # for celltype in cell_types:
    # I = cf['Cell type'] == celltype
    I = ~cf[field].isna() & ~cf['Border'] & ~(cf['Cell cycle phase'] == 'Mitosis')
    
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
def get_exponential_growth_rate(cf:pd.DataFrame,field:str = 'Cell volume',time_field:str='Time', filtered:dict={}):
    ''' 
    
    Populate a cell track with exponential growth rates estimated from y = field and x = time_field
    
    exp_rate = log(y) / t

    Parameters
    ----------
    cf : pd.DataFrame
        Single pd.Dataframe containing a cell track.
    field : str, optional
        y axis data name. The default is 'Cell volume'.
    time_field : str, optional
        x axis data name. The default is 'Time'.
    filtered : dict, optional
        A single element dictionary mapping a filter name and a logical index for the subset
        of data to exstimate on. The default is {}.
            e.g. {'G1 only': I_g1}

    Returns
    -------
    cf : pd.DataFrame
        Cell track with exponential growth rate populated.

    '''
    cell_types = cf['Cell type'].astype(str).unique()
    for celltype in cell_types:
        I = cf['Cell type'] == celltype
        I = I & ~cf[field].isna() & ~cf['Border'] & ~(cf['Cell cycle phase'] == 'Mitosis')
        
        if len(filtered) > 0:
            assert(len(filtered) == 1)
            filter_name = list(filtered.keys())[0]
            Ifilter = filtered[filter_name]
            I = I & Ifilter
            new_field_name = f'{field} (filter_name) exponential growth rate'
        else:
            new_field_name = f'{field} exponential growth rate'
        
        # if np.any(cf['Cell cycle phase'] == 'Mitosis'):
        #     mitotic_idx = cf[cf['Cell cycle phase'] == 'Mitosis'].iloc[0].index
        # else:
        #     mitotic_idx = None
        
        if len(cf.loc[I]) < 4:
            cf.loc[I,new_field_name] = np.nan
        else:
            t = cf.loc[I][time_field].values
            y = cf.loc[I][field].values
            
            try:
                p,_ = curve_fit(exp_model,t,y,p0=[y[0],1],
                                     bounds = [ [0,-np.inf],
                                       [y.max(),np.inf]])
                V0,gamma = p
                cf.loc[I,new_field_name] = gamma
            except:
                cf.loc[I,new_field_name] = np.nan
    
    return cf

def get_prev_or_next_frame(df,this_frame,direction='next'):
    
    if direction == 'next':
        increment = +1
    if direction == 'prev':
        increment = -1
    
    assert(type(this_frame) == pd.Series)
    df = df.reset_index().set_index(['TrackID','Frame'])
    track = df.loc[this_frame.name,:].reset_index()
    if this_frame['Frame',''] + increment in track['Frame'].values:
        retrieved_frame = track.set_index('Frame').loc[this_frame['Frame',''] + increment]
        return retrieved_frame
    else:
        return None

def map_tzyx_to_labels(coords, tracks:np.array ):
    
    coords['label'] = np.nan
    assert(tracks.ndim == 4)
    for idx,row in coords.iterrows():
        T,Z,Y,X = row[['T','Z','Y','X']].astype(int)
        label = tracks[T,Z,Y,X]
        coords.loc[idx,'label'] = label
    return coords

def plot_track(cf,y=('Nuclear volume','Measurement'),
               x=('Time','Measurement'),
               celltypefield=('Cell type','Meta'),
               linestyle: dict={'Basal':'-','Suprabasal':'--'},
               color='b',alpha=1):
    
    cf = cf.reset_index()
    cell_types = cf[celltypefield].astype(str)

    prev_type = None
    for celltype in cell_types:
        
        if celltype == 'nan' or celltype =='NA':
            continue
        if celltype=='Basal':
            label = cf.iloc[0].TrackID
        else:
            label = None
            
        I = cf[celltypefield] == celltype
        t = cf.loc[I,x].values
        v = cf.loc[I,y].values

        if prev_type is not None:
            #grab last time point and connect
            Ilast = np.where(cf[celltypefield] == prev_type)[0][-1]
            t = np.insert(t, 0, cf.iloc[Ilast][x])
            v = np.insert(v, 0, cf.iloc[Ilast][y])
            
        plt.plot(t,v,linestyle=linestyle[celltype],alpha=alpha,
                 label=label, color=color)

        prev_type = celltype
        
    plt.show()
    plt.title(f'Track {cf.iloc[0].TrackID}, Border: {np.any(cf.Border)}')




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 22:51:08 2025

@author: xies
"""

import numpy as np
from skimage import measure, img_as_bool, transform, util, filters, exposure, io
import pandas as pd
from functools import reduce

from mathUtils import get_neighbor_idx, parse_3D_inertial_tensor, argsort_counter_clockwise
from imageUtils import get_mask_slices
# 3D mesh stuff
from scipy.spatial import Voronoi, Delaunay
from aicsshparam import shtools, shparam
from trimesh import Trimesh, smoothing
from trimesh.curvature import discrete_gaussian_curvature_measure, \
    discrete_mean_curvature_measure, sphere_ball_intersection
import pyvista as pv
from os import path, makedirs

# from toeplitzDifference import backward_difference, forward_difference, central_difference
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import curve_fit
from sklearn import preprocessing

import matplotlib.pyplot as plt


def reslice_by_heightmap(im,heightmap,top_border,bottom_border):

    ZZ,YY,XX = im.shape
    
    flat = np.zeros((-top_border + bottom_border,XX,XX))

    Iz_top = heightmap + top_border
    Iz_bottom = heightmap + bottom_border
    
    for x in range(XX):
        for y in range(YY):
            
            flat_indices = np.arange(0,-top_border+bottom_border)
            
            z_coords = np.arange(Iz_top[y,x],Iz_bottom[y,x])
            # sanitize for out-of-bounds
            z_coords[z_coords < 0] = 0
            z_coords[z_coords >= ZZ] = ZZ-1
            I = (z_coords > 0) & (z_coords < ZZ)
            
            flat[flat_indices[I],y,x] = im[z_coords[I],y,x]
            flat[flat_indices[I],y,x] = im[z_coords[I],y,x]
            
    return flat

# Suppress batch effects
def scale_by_region(df):

    scaled = []
    for region,_df in df.groupby('Region'):
        meas = _df.xs('Measurement',level=1,axis=1)
        _X = meas.values
        _X[np.isinf(_X)] = np.nan

        _X = preprocessing.StandardScaler().fit_transform(_X)
        _df = pd.DataFrame(index=_df.index,columns=meas.columns,
                           data=_X)
        _df['Region'] = region
        scaled.append(_df)

    return pd.concat(scaled)


def get_mesh_from_bm_image(bm_height_image, spacing=[1,.25,.25], decimation_factor=30):

    dz,dx,dx = spacing
    mask = (bm_height_image > 0)
    Z,Y,X = np.where(mask)
    X = X[1:]; Y = Y[1:]; Z = Z[1:]
    X = X*dx; Y = Y*dx; Z = Z*dz

    # Decimate the grid to avoid pixel artefacts
    X_ = X[::decimation_factor]; Y_ = Y[::decimation_factor]; Z_ = Z[::decimation_factor]
    grid = pv.PolyData(np.stack((X_,Y_,Z_)).T)
    mesh = grid.delaunay_2d()
    faces = mesh.faces.reshape((mesh.n_cells, 4))[:, 1:]
    mesh = Trimesh(mesh.points,faces)
    mesh = smoothing.filter_humphrey(mesh,alpha=1)

    # Check the face normals (if mostly aligned with +z, then keep sign if not, then invert sign)
    if ((mesh.facets_normal[:,2] > 0).sum() / len(mesh.facets_normal)) < 0.5:
        mesh.invert()

    return mesh

def get_tissue_curvatures(mesh,kappa:float=5,query_pts=None):
    if query_pts is None:
        query_pts = mesh.vertices

    mean_curve = discrete_mean_curvature_measure(
        mesh, query_pts, kappa)/sphere_ball_intersection(1, kappa)
    gaussian_curve = discrete_gaussian_curvature_measure(
        mesh, query_pts, kappa)/sphere_ball_intersection(1, kappa)

    return mean_curve, gaussian_curve

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
    df = df.drop(columns=['Nuclear bbox top','Nuclear bbox bottom'])

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
    # Drop all 0s (l>m)
    sh_coefficients = sh_coefficients.loc[:,~np.all(sh_coefficients==0, axis=0)]

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
    I = ~cf[field].isna() & ~cf['Border'].astype(bool) & ~(cf['Cell cycle phase'] == 'Mitosis')

    this_type = cf.loc[I]
    if len(this_type) < 5:
        yhat = this_type[field].values
        dydt = np.ones(len(this_type)) * np.nan
    else:
        t = np.array(range(0,len(this_type))) * 12
        v = this_type[field].values
        # Spline smooth
        spl = make_smoothing_spline(t, v)
        yhat = spl(t)
        dydt = spl.derivative(nu=1)(t)

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
def get_exponential_growth_rate(cf:pd.DataFrame,
                                field:str = 'Cell volume',
                                time_field:str='Time',
                                filtered:dict={}):
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
        I = I & ~cf[field].isna() & ~cf['Border'].astype(bool) \
            & ~(cf['Cell cycle phase'] == 'Mitosis')

        if len(filtered) > 0:
            assert(len(filtered) == 1)
            filter_name = list(filtered.keys())[0]
            Ifilter = filtered[filter_name]
            I = I & Ifilter
            new_field_name = f'{field} {filter_name} exponential growth rate'
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

def get_prev_or_next_frame(df,frame_track_of_interest,direction='next',increment:int=1):

    if direction == 'next':
        sign = +1
    if direction == 'prev':
        sign = -1

    increment = sign*increment

    assert(type(frame_track_of_interest) == pd.Series)
    #NB: Re-indexing is slow
    frame_of_interest = frame_track_of_interest['Frame','']
    track_of_interest = frame_track_of_interest['TrackID','']
    index2retrieve = (frame_of_interest + increment,track_of_interest)

    if index2retrieve in df.index:
        retrieved_frame = df.loc[index2retrieve]
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
    plt.ylabel(f'{y}')

def measure_all_this_frame(dirname : str,
                           tracked_nuc: np.array,
                           tracked_cyto : np.array,
                           intensity_images : dict,
                           spacing=[1,1,1], dt=12,
                           z_shift : int=10,
                           save_flag : bool=False,
                           lmax : int=5):

    assert(tracked_nuc.ndim == 4) # 4D images
    assert(tracked_cyto.ndim == 4)

    dz,_,dx = spacing
    T,Z,Y,X = tracked_cyto.shape

    all_df = []
    for t in range(15):

        #----- read segmentation files -----
        nuc_seg = tracked_nuc[t,...]
        cyto_seg = tracked_cyto[t,...]
        ZZ,YY,XX = nuc_seg.shape

        # --- 1. Voxel-based cell geometry measurements ---
        df_nuc = measure_nuclear_geometry_from_regionprops(nuc_seg,spacing = [dz,dx,dx])
        df_cyto = measure_cyto_geometry_from_regionprops(cyto_seg,spacing = [dz,dx,dx])
        df = pd.merge(left=df_nuc,right=df_cyto,left_on='TrackID',right_on='TrackID',how='left')
        df['Frame'] = t
        df['Time'] = t * dt

        intensity_df = measure_cyto_intensity(cyto_seg,intensity_images)
        df = pd.merge(left=df,right=intensity_df,left_on='TrackID',right_on='TrackID',how='left')
        df['Nuclear bbox top'] = df['Nuclear bbox top']

        # ----- 3. Use flattened 3d cortical segmentation and measure geometry and collagen
        # from cell-centric coordinates ----
        f = path.join(dirname,f'Image flattening/flat_tracked_cyto/t{t}.tif')
        flat_cyto = io.imread(f)

        # Calculate collagen structuring matrix
        collagen_image = io.imread(path.join(dirname,f'Image flattening/flat_collagen/t{t}.tif'))
        (Jxx,Jxy,Jyy) = measure_collagen_structure(collagen_image,blur_sigma=3)

        df_flat,basal_masks_2save = measure_flat_cyto_from_regionprops(
            flat_cyto, collagen_image, (Jxx, Jyy, Jxy), spacing = [dz,dx,dx])
        df = pd.merge(df,df_flat,left_on='TrackID',right_on='TrackID',how='left')

        if not path.exists(path.join(dirname,'Image flattening/basal_masks')):
            makedirs(path.join(dirname,'Image flattening/basal_masks'))
        if save_flag:
            io.imsave(path.join(dirname,f'Image flattening/basal_masks/t{t}.tif'),basal_masks_2save)

        # Book-keeping
        df = df.drop(columns=['bbox-1','bbox-2','bbox-4','bbox-5'])
        df['X-pixels'] = df['X'] / dx
        df['Y-pixels'] = df['Y'] / dx

        # Derive NC ratio
        df['NC ratio'] = df['Nuclear volume'] / df['Cell volume']

        dense_coords = np.array([df['Y-pixels'],df['X-pixels']]).T
        dense_coords_3d_um = np.array([df['Z'],df['Y'],df['X']]).T

        #----- 4. Nuc-to-BM heights -----
        # Load heightmap and calculate adjusted height
        heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
        heightmap_shifted = heightmap + z_shift
        df['Height to BM'] = heightmap_shifted[
            np.round(df['Y']).astype(int),np.round(df['X']).astype(int)] - df['Z']

        #----- Find border cells -----
        # Generate a dense mesh based sole only 2D/3D nuclear locations
        #% Use Delaunay triangulation in 2D to approximate the basal layer topology
        tri_dense = Delaunay(dense_coords)
        # Use the dual Voronoi to get rid of the border/infinity cells
        vor = Voronoi(dense_coords)
        # Find the voronoi regions that touch the image border
        vertices_outside = np.where(np.any((vor.vertices < 0) | (vor.vertices > XX),axis=1))
        regions_outside = np.where([ np.any(np.in1d(np.array(r), vertices_outside)) for r in vor.regions])
        regions_outside = np.hstack([regions_outside, np.where([-1 in r for r in vor.regions])])
        Iborder = np.in1d(vor.point_region, regions_outside)
        border_nuclei = df.loc[Iborder].index
        df['Border'] = False
        df.loc[ border_nuclei, 'Border'] = True

        #----- Cell coordinates mesh for geometry -----
        # Generate 3D mesh for curvature analysis -- no need to specify precise cell-cell junctions
        Z,Y,X = dense_coords_3d_um.T
        cell_coords_mesh = Trimesh(vertices = np.array([X,Y,Z]).T, faces=tri_dense.simplices)
        mean_curve_coords = -discrete_mean_curvature_measure(
            cell_coords_mesh, cell_coords_mesh.vertices, 5)/sphere_ball_intersection(1, 5)
        gaussian_curve_coords = -discrete_gaussian_curvature_measure(
            cell_coords_mesh, cell_coords_mesh.vertices, 5)/sphere_ball_intersection(1, 5)
        df['Mean curvature - cell coords'] = mean_curve_coords
        df['Gaussian curvature - cell coords'] = gaussian_curve_coords

        # ---- 5. Get 3D mesh from the BM image ---
        # from scipy import interpolate
        from trimesh import smoothing
        bm_height_image = io.imread(path.join(dirname,f'Image flattening/height_image/t{t}.tif'))
        mask = (bm_height_image > 0)
        Z,Y,X = np.where(mask)
        X = X[1:]; Y = Y[1:]; Z = Z[1:]
        X = X*dx; Y = Y*dx; Z = Z*dz

        # Decimate the grid to avoid artefacts
        X_ = X[::30]; Y_ = Y[::30]; Z_ = Z[::30]
        grid = pv.PolyData(np.stack((X_,Y_,Z_)).T)
        mesh = grid.delaunay_2d()
        faces = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
        mesh = Trimesh(mesh.points,faces)
        mesh = smoothing.filter_humphrey(mesh,alpha=1)
        # Check the face normals (if mostly aligned with +z, then keep sign if not, then invert sign)
        if ((mesh.facets_normal[:,2] > 0).sum() / len(mesh.facets_normal)) > 0.5:
            curvature_sign = 1
        else:
            curvature_sign = -1

        closest_mesh_to_cell,_,_ = mesh.nearest.on_surface(dense_coords_3d_um[:,::-1])

        mean_curve = discrete_mean_curvature_measure(
            mesh, closest_mesh_to_cell, 5)/sphere_ball_intersection(1, 5)
        gaussian_curve = discrete_gaussian_curvature_measure(
            mesh, dense_coords_3d_um, 5)/sphere_ball_intersection(1, 5)
        df['Mean curvature'] = curvature_sign * mean_curve
        df['Gaussian curvature'] = gaussian_curve

        #----- 6. Use manual 3D topology to compute neighborhoods lengths -----
        # Load the actual neighborhood topology
        # A = np.load(path.join(dirname,f'Image flatteniowng/flat_adj_dict/adjdict_t{t}.npy'),allow_pickle=True).item()
        # D = distance.squareform(distance.pdist(dense_coords_3d_um))


        # --- 2. 3D shape decomposition ---

        # 2a: Estimate cell and nuclear mesh using spherical harmonics
        sh_coefficients = estimate_sh_coefficients(cyto_seg, lmax, spacing = [dz,dx,dx])
        sh_coefficients = sh_coefficients.set_index('TrackID')
        sh_coefficients.columns = 'cyto_' + sh_coefficients.columns
        df = pd.merge(df,sh_coefficients,left_on='TrackID',right_on='TrackID',how='left')
        sh_coefficients = estimate_sh_coefficients(nuc_seg, lmax, spacing = [dz,dx,dx])
        sh_coefficients = sh_coefficients.set_index('TrackID')
        sh_coefficients.columns = 'nuc_' + sh_coefficients.columns
        df = pd.merge(df,sh_coefficients,left_on='TrackID',right_on='TrackID',how='left')

        #----- Use macrophage annotations to find distance to them -----
        #NB: the macrophage coords are in um
        macrophage_xyz = pd.read_csv(path.join(dirname,f'3d_cyto_seg/macrophages/t{t}.csv'))
        macrophage_xyz = macrophage_xyz.rename(columns={'axis-0':'Z','axis-1':'Y','axis-2':'X'})
        macrophage_xyz['X'] = macrophage_xyz['X'] * dx
        macrophage_xyz['Y'] = macrophage_xyz['Y'] * dx
        df['Distance to closest macrophage'] = find_distance_to_closest_point(pd.DataFrame(dense_coords_3d_um,columns=['Z','Y','X']), macrophage_xyz)

        # Merge with manual annotations
        df = df.reset_index()

        # Append the DF
        all_df.append(df)

    all_df = pd.concat(all_df,ignore_index=False)
    all_df = all_df.set_index(['Frame','TrackID'])

    return all_df

#--- Bookkeepers ---
from imageUtils import trim_multimasks_to_shared_bounding_box

def extract_nuc_and_cell_mask_from_idx(idx : tuple,
                                        tracked_nuc_by_region:dict,
                                        tracked_cyto_by_region:dict,):
    '''
    Returns a tuple of nuc_mask,cyto_mask if given the measurement index of the cell.
    Index should be in the format (frame,'Region_trackID'), where frame is int

    '''
    assert(len(idx)) == 2

    frame = idx[0]
    region,trackID = idx[1].split('_')
    trackID = int(trackID)
    nuc_mask = tracked_nuc_by_region[region][frame,...] == trackID
    cyto_mask = tracked_cyto_by_region[region][frame,...] == trackID
    nuc_mask,cyto_mask = trim_multimasks_to_shared_bounding_box((nuc_mask,cyto_mask))

    return nuc_mask,cyto_mask

def get_microenvironment_mask(trackID,
                              adjdict: dict,
                              cyto_seg: np.array):
    adjacentIDs = adjdict[trackID]
    mask = np.zeros_like(cyto_seg,dtype=bool)
    for ID in adjacentIDs:
        mask[cyto_seg == ID] = True

    return mask

def extract_nuc_and_cell_and_microenvironment_mask_from_idx(idx : tuple,
                                        adjdict_by_region:dict,
                                        tracked_nuc_by_region:dict,
                                        tracked_cyto_by_region:dict,):
    '''
    Returns a tuple of nuc_mask,cyto_mask,microenvironment_mask
    if given the measurement index of the cell.

    Index should be in the format (frame,'Region_trackID'), where frame is int

    '''

    assert(len(idx)) == 2

    frame = idx[0]
    region,trackID = idx[1].split('_')
    trackID = int(trackID)
    nuc_mask = tracked_nuc_by_region[region][frame,...] == trackID
    cyto_mask = tracked_cyto_by_region[region][frame,...] == trackID
    microenvironment_mask = get_microenvironment_mask(trackID,adjdict_by_region[region][frame],
                                                      tracked_cyto_by_region[region][frame,...])
    nuc_mask,cyto_mask,microenvironment_mask = trim_multimasks_to_shared_bounding_box((nuc_mask,cyto_mask,microenvironment_mask))

    return nuc_mask,cyto_mask,microenvironment_mask

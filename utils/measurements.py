#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 22:51:08 2025

@author: xies
"""

import numpy as np
from skimage import measure, img_as_bool, transform, util, filters, exposure, io, morphology
import pandas as pd
from functools import reduce

from mathUtils import get_neighbor_idx, parse_3D_inertial_tensor, argsort_counter_clockwise
from imageUtils import get_mask_slices
# 3D mesh stuff
from scipy.spatial import Voronoi, Delaunay, distance
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

from collections.abc import Callable



def aggregate_over_adj(adj: dict, aggregators: dict[str,Callable],
                       df = pd.DataFrame, fields2aggregate=list[str]):

    df_aggregated = pd.DataFrame(
        columns = [f'{k} adjac {f}' for k in aggregators.keys() for f in fields2aggregate],
        index=df.index, dtype=float)

    # for agg_name in aggregators.keys():
    #     for field in fields2aggregate:
    #         df_aggregated[f'{agg_name} adjac {field}'] = np.nan

    for centerID,neighborIDs in adj.items():
        neighbors = df.loc[neighborIDs]
        if len(neighbors) > 0:
            for agg_name, agg_func in aggregators.items():
                for field in fields2aggregate:
                    if neighbors[field].values.dtype == float:
                        if not np.all(np.isnan(neighbors[field].values)):
                            df_aggregated.loc[centerID,f'{agg_name} adjac {field}'] = \
                                agg_func(neighbors[field].values)
                    else:
                        df_aggregated.loc[centerID,f'{agg_name} adjac {field}'] = \
                            agg_func(neighbors[field].values)

    df_aggregated.index.name = 'TrackID'

    return df_aggregated.reset_index()

def get_aggregated_3D_distances(df:pd.DataFrame,adjDict:dict,aggregators:dict):
    D = distance.squareform(distance.pdist(df[['Z','Y','X']]))
    D = pd.DataFrame(data=D,index=df.index,columns=df.index)

    distances = pd.DataFrame(index=adjDict.keys(),
                             columns = [f'{agg_name} distance to neighbors' for agg_name in aggregators.keys()])

    distances.index.name = 'TrackID'
    for cellID,neighborIDs in adjDict.items():
        for agg_name, agg_func in aggregators.items():
            distances.loc[cellID,f'{agg_name} distance to neighbors'] = agg_func( D.loc[cellID,neighborIDs].values )

    return distances.sort_index().reset_index()

def frac_neighbors_are_border(v):
    frac = v.sum() / len(v)
    return frac

def frac_sphase(v):
    has_cell_cycle = v[v != 'NA']
    if len(has_cell_cycle) > 0:
        frac = (has_cell_cycle == 'SG2').sum() / len(has_cell_cycle)
    else:
        frac = np.nan
    return frac

def find_touching_labels(labels, centerID, threshold, selem=morphology.disk(3)):
    this_mask = labels == centerID
    this_mask_dil = morphology.binary_dilation(this_mask,selem)
    touchingIDs,counts = np.unique(labels[this_mask_dil],return_counts=True)
    touchingIDs[counts > threshold] # should get rid of 'conrner touching'

    touchingIDs = touchingIDs[touchingIDs > 2] # Could touch background pxs
    touchingIDs = touchingIDs[touchingIDs != centerID] # nonself

    return touchingIDs


#% Reconstruct adj network from cytolabels that touch
def get_adjdict_from_2d_segmentation(seg2d:np.array, touching_threshold:int = 2):
    '''

    Parameters
    ----------
    seg2d : np.array
        2D cytoplasmic segmentation on which to determine adjacency
    touching_threshold : int, optional
        Minimum number of overlap pixels. The default is 2.

    Returns
    -------
    A : dict
        Dictionary of adjacent labels:
            {centerID : neighborIDs }

    '''
    #@todo: OK for 3D segmentation? currently no...
    assert(seg2d.ndim == 2) # only works with 2D images for now

    A = {centerID:find_touching_labels(seg2d, centerID, touching_threshold)
         for centerID in np.unique(seg2d)[1:]}

    return A

def reslice_by_heightmap(im,heightmap,top_border,bottom_border):

    ZZ,YY,XX = im.shape

    flat = np.zeros((-top_border + bottom_border,XX,XX),dtype=im.dtype)

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

    return mesh.copy()

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

def measure_flat_cyto_from_regionprops(flat_cyto, collagen_image, jacobian, spacing= [1,1,1],
                                       slicees_to_average:int = 3):
    '''
    PARAMETERS

    flat_cyto: flattened cytoplasmic segmentation
    collagen_image: flattened Max int projection of collagen image
    jacobian: jacobian matrix of the gradient image of collagen signal
    spacing: default = [1,1,1] pixel sizes in microns
    slicees_to_average: default = 3 z-slices to sum

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

        # Apical area (3 top slices default)
        apical_area = mask[Z_top:Z_top+slicees_to_average,...].max(axis=0)
        apical_area = apical_area.sum()

        # mid-level area
        mid_area = mask[np.round((Z_top+Z_bottom)/2).astype(int),...].sum()

        # Apical area (3 bottom slices default)
        basal_mask = mask[Z_bottom-slicees_to_average-1:Z_bottom,...]
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
        df.at[i,'Basal alignment to basal footprint'] = np.abs(np.cos(theta - basal_orientation))

        # Also include collagen intensity
        # Normalize collagen
        collagen_image = exposure.equalize_hist(collagen_image)
        df.at[i,'Subbasal collagen intensity'] = np.mean(collagen_image[basal_mask])

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

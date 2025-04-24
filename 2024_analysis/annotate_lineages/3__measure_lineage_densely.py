#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:48:47 2025

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
from os import path,makedirs
from basicUtils import nonans
from functools import reduce

dx = 0.25
dz = 1
Z_SHIFT = 10

# Differentiating thresholds
centroid_height_cutoff = 3.5 #microns above BM

# for expansion
footprint = morphology.cube(3)

# Filenames
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

#%%

# convert trianglulation to adjacency matrix (for easy editing)
def tri_to_adjmat(tri):
    num_verts = max(map(max,tri.simplices)) + 1
    A = np.zeros((num_verts,num_verts),dtype=bool)
    for idx in range(num_verts):
        neighbor_idx = get_neighbor_idx(tri,idx)
        A[idx,neighbor_idx] = True
    return A

# Find distance to nearest manually annotated point in points-list
def find_distance_to_closest_point(dense_3d_coords,annotation_coords_3d):
    distances = np.zeros(len(dense_3d_coords))

    for i,row in dense_3d_coords.iterrows():
        dx = row['X'] - annotation_coords_3d['X']
        dy = row['Y'] - annotation_coords_3d['Y']
        dz = row['Z'] - annotation_coords_3d['Z']
        D = np.sqrt(dx**2 + dy**2 + dz**2)
        
        distances[i] = D.min()
            
    return distances

def measure_nuclear_geometry_from_regionprops(nuc_labels, spacing = [1,1,1]):
    df = pd.DataFrame( measure.regionprops_table(nuc_labels,
                                                 properties=['label','area','solidity','bbox'],
                                                 spacing = [dz,dx,dx],
                                                 ))
    df = df.rename(columns={
                            'label':'TrackID',
                            'area':'Nuclear volume',
                            'bbox-0':'Nuclear bbox top',
                            'bbox-3':'Nuclear bbox bottom',
                            'solidity':'Nuclear solidity'},
                   )
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
                                      'centroid-0':'Z',
                                      'centroid-1':'Y',
                                      'centroid-2':'X'},
                             )
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

def estimate_sh_coefficients(cyto_seg, lmax, spacing = [dz,dx,dx]):
    
    ZZ,YY,XX = cyto_seg.shape
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
        coeffs['shcoeffs_volume'] = pv.wrap(mesh).volume
        
        # verts,faces,_,_ = measure.marching_cubes(mask)
        # mesh_mc = Trimesh(verts,faces)
        # coeffs['mc_surface_area'] = pv.wrap(mesh_mc).area
        # coeffs['mc_volume'] = pv.wrap(mesh_mc).volume
        
        sh_coefficients.append(coeffs)
        
    sh_coefficients = pd.DataFrame(sh_coefficients)
    
    return sh_coefficients


def measure_flat_cyto_from_regionprops(flat_cyto, jacobian, spacing= [1,1,1]):
    '''
    
    PARAMETERS
    
    flat_cyto: flattened cytoplasmic segmentation
    jacobian: jacobian matrix of the gradient image of collagen signal
    spacing: default = [1,1,1] pixel sizes in microns
    
    '''
    
    
    # Measurements from cortical segmentation
    
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

#%%

SAVE = False
VISUALIZE = False
LMAX = 10 # Number of spherical harmonics components to calculate

all_df = []

# Load segmentations
tracked_nuc = io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif'))
tracked_cyto = io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif'))

# Load channels
h2b = io.imread(path.join(dirname,'Cropped_images/B.tif'))
fucci_g1 = io.imread(path.join(dirname,'Cropped_images/R.tif'))

for t in tqdm(range(15)):
    
    #----- read segmentation files -----
    nuc_seg = tracked_nuc[t,...]
    cyto_seg = tracked_cyto[t,...]
    ZZ,YY,XX = nuc_seg.shape
    
    # --- 1. Voxel-based cell geometry measurements ---
    df_nuc = measure_nuclear_geometry_from_regionprops(nuc_seg,spacing = [dz,dx,dx])
    df_cyto = measure_cyto_geometry_from_regionprops(cyto_seg,spacing = [dz,dx,dx])
    df = df_cyto.join(df_nuc)
    int_images = {'H2B':h2b[t,...],'FUCCI':fucci_g1[t,...]}
    intensity_df = measure_cyto_intensity(cyto_seg,int_images)
    df = df.join(intensity_df)
    
    # --- 2. Mesh-based cell geometry measurements ---
    
    # 2a: Estimate cell and nuclear mesh using spherical harmonics
    sh_coefficients = estimate_sh_coefficients(cyto_seg, LMAX, spacing = [dz,dx,dx])
    sh_coefficients = sh_coefficients.set_index('TrackID')
    sh_coefficients.columns = 'cyto_' + sh_coefficients.columns
    df = df.join(sh_coefficients)
    sh_coefficients = estimate_sh_coefficients(nuc_seg, LMAX, spacing = [dz,dx,dx])
    sh_coefficients = sh_coefficients.set_index('TrackID')
    sh_coefficients.columns = 'nuc_' + sh_coefficients.columns
    df = df.join(sh_coefficients)
    
    # ----- 3. Use flattened 3d cortical segmentation and measure geometry and collagen
    # from cell-centric coordinates ----
    f = path.join(dirname,f'Image flattening/flat_tracked_cyto/t{t}.tif')
    flat_cyto = io.imread(f)
    # Load the structuring matrix elements for collagen
    f = path.join(dirname,f'Image flattening/collagen_orientation/t{t}.npy')
    [Gx,Gy] = np.load(f)
    Jxx = Gx*Gx
    Jxy = Gx*Gy
    Jyy = Gy*Gy
    
    df_flat,basal_masks_2save = measure_flat_cyto_from_regionprops(
        flat_cyto, (Jxx, Jyy, Jxy), spacing = [dz,dx,dx])
    df = df.join(df_flat)
    
    if not path.exists(path.join(dirname,'Image flattening/basal_masks')):
        makedirs(path.join(dirname,'Image flattening/basal_masks'))
    if SAVE:
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
    heightmap_shifted = heightmap + Z_SHIFT
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
    
    #----- 3D cell mesh for geometry -----
    # Generate 3D mesh for curvature analysis -- no need to specify precise cell-cell junctions
    Z,Y,X = dense_coords_3d_um.T
    mesh = Trimesh(vertices = np.array([X,Y,Z]).T, faces=tri_dense.simplices)
    mean_curve_coords = discrete_mean_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    gaussian_curve_coords = discrete_gaussian_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    df['Mean curvature - cell coords'] = mean_curve_coords
    df['Gaussian curvature - cell coords'] = gaussian_curve_coords
    
    # ---- 5. Get 3D mesh from the BM image ---
    # from scipy import interpolate
    from trimesh import smoothing
    bm_height_image = io.imread(path.join(dirname,f'Image flattening/height_image/t{t}.tif'))
    mask = (bm_height_image[...,0] > 0)
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

    closest_mesh_to_cell,_,_ = mesh.nearest.on_surface(dense_coords_3d_um[:,::-1])
    # pl =pv.Plotter()
    # pl.add_mesh(pv.wrap(mesh))
    # pl.add_points(dense_coords_3d_um,color='r')
    # pl.add_points(closest_mesh_to_cell,color='b')
    # pl.show()
    
    mean_curve = discrete_mean_curvature_measure(mesh, closest_mesh_to_cell, 2)/sphere_ball_intersection(1, 2)
    gaussian_curve = discrete_gaussian_curvature_measure(mesh, dense_coords_3d_um, 5)/sphere_ball_intersection(1, 5)
    df['Mean curvature'] = mean_curve
    df['Gaussian curvature'] = gaussian_curve
    
    #----- 6. Use manual 3D topology to compute neighborhoods -----
    # Load the actual neighborhood topology
    A = np.load(path.join(dirname,f'Image flattening/flat_adj/adjmat_t{t}.npy'))
    D = distance.squareform(distance.pdist(dense_coords_3d_um))
    
    #----- Use macrophage annotations to find distance to them -----
    #NB: the macrophage coords are in um
    macrophage_xyz = pd.read_csv(path.join(dirname,f'3d_cyto_seg/macrophages/t{t}.csv'))
    macrophage_xyz = macrophage_xyz.rename(columns={'axis-0':'Z','axis-1':'Y','axis-2':'X'})
    macrophage_xyz['X'] = macrophage_xyz['X'] * dx
    macrophage_xyz['Y'] = macrophage_xyz['Y'] * dx
    df['Distance to closest macrophage'] = \
        find_distance_to_closest_point(pd.DataFrame(dense_coords_3d_um,columns=['Z','Y','X']), macrophage_xyz)
    
    # Load basal masks for current frame
    frame_basal_mask = io.imread(path.join(dirname,f'Image flattening/basal_masks/t{t}.tif'))

    # Save the DF
    all_df.append(df)
    
    #Save a bunch of intermediates
    # Save segmentation with text labels @ centroid
    # im_cellposeID = draw_labels_on_image(dense_coords,df_dense['basalID'],[XX,XX],font_size=12)
    # im_cellposeID.save(path.join(dirname,f'3d_nuc_seg/cellposeIDs/t{t}.tif'))
    
all_df = pd.concat(all_df,ignore_index=False)

if SAVE:
    
    all_df.to_csv(path.join(dirname,'tissue_dataframe.csv'))
    print(f'Saved to: {dirname}')


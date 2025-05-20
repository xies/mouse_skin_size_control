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
import seaborn as sb
from aicsshparam import shtools

# Specific utils
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, \
    most_likely_label, colorize_segmentation
from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, \
    discrete_mean_curvature_measure, sphere_ball_intersection
import pyvista as pv

# General utils
from tqdm import tqdm
from os import path,makedirs
from basicUtils import nonans
import pickle as pkl

dt = 12 #hrs
dx = 0.25
dz = 1
Z_SHIFT = 10

# for expansion
footprint = morphology.cube(3)

# Filenames
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

#%% 

from measurements import measure_nuclear_geometry_from_regionprops, \
        measure_cyto_geometry_from_regionprops, measure_cyto_intensity, measure_flat_cyto_from_regionprops, \
        estimate_sh_coefficients, find_distance_to_closest_point

SAVE = True
VISUALIZE = False
LMAX = 10 # Number of spherical harmonics components to calculate

all_df = []

# Load manual tracking
with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'rb') as file:
    tracks = pkl.load(file)
manual = pd.concat(tracks,ignore_index=True)
manual['Frame'] = manual['Frame'].astype(float)
manual = manual.set_index(['Frame','TrackID'])
manual = manual.drop(columns=['ID','X','Y','Z','Border'])

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
    df = df_nuc.join(df_cyto)
    df['Frame'] = t
    df['Time'] = t * dt
    int_images = {'H2B':h2b[t,...],'FUCCI':fucci_g1[t,...]}
    intensity_df = measure_cyto_intensity(cyto_seg,int_images)
    df = df.join(intensity_df)
    
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
    # A = np.load(path.join(dirname,f'Image flattening/flat_adj/adjmat_t{t}.npy'))
    # D = distance.squareform(distance.pdist(dense_coords_3d_um))
    
    # --- 2. 3D shape decomposition ---
    
    # 2a: Estimate cell and nuclear mesh using spherical harmonics
    sh_coefficients = estimate_sh_coefficients(cyto_seg, LMAX, spacing = [dz,dx,dx])
    sh_coefficients = sh_coefficients.set_index('TrackID')
    sh_coefficients.columns = 'cyto_' + sh_coefficients.columns
    df = df.join(sh_coefficients)
    sh_coefficients = estimate_sh_coefficients(nuc_seg, LMAX, spacing = [dz,dx,dx])
    sh_coefficients = sh_coefficients.set_index('TrackID')
    sh_coefficients.columns = 'nuc_' + sh_coefficients.columns
    df = df.join(sh_coefficients)
    
    
    #----- Use macrophage annotations to find distance to them -----
    #NB: the macrophage coords are in um
    macrophage_xyz = pd.read_csv(path.join(dirname,f'3d_cyto_seg/macrophages/t{t}.csv'))
    macrophage_xyz = macrophage_xyz.rename(columns={'axis-0':'Z','axis-1':'Y','axis-2':'X'})
    macrophage_xyz['X'] = macrophage_xyz['X'] * dx
    macrophage_xyz['Y'] = macrophage_xyz['Y'] * dx
    df['Distance to closest macrophage'] = find_distance_to_closest_point(pd.DataFrame(dense_coords_3d_um,columns=['Z','Y','X']), macrophage_xyz)
    
    # Load basal masks for current frame
    frame_basal_mask = io.imread(path.join(dirname,f'Image flattening/basal_masks/t{t}.tif'))

    # Merge with manual annotations
    df = df.reset_index()
    df.set_index(['Frame','TrackID'])
    
    # Save the DF
    all_df.append(df)
    
    #Save a bunch of intermediates
    # Save segmentation with text labels @ centroid
    # im_cellposeID = draw_labels_on_image(dense_coords,df_dense['basalID'],[XX,XX],font_size=12)
    # im_cellposeID.save(path.join(dirname,f'3d_nuc_seg/cellposeIDs/t{t}.tif'))
    
all_df = pd.concat(all_df,ignore_index=False)
all_df = all_df.set_index(['Frame','TrackID'])
all_df = all_df.join(manual)
all_df.to_csv(path.join(dirname,'Mastodon/single_timepoints.csv'))

#%% Dimensionality reduce the SPH coefficients

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'),index_col=['Frame','TrackID'])
# df = all_df

# # Nuclear first
# nuc_columns = df.columns[df.columns.str.startswith('nuc_')]
# # nuc_columns = nuc_columns[df[nuc_columns].sum() > 0]
# # Drop NaN columns (these are often late mitotic figures)
# Ivalid = np.any(df[nuc_columns].notna(),axis=1) & ~df.Border
# scaler = StandardScaler()
# nuc_sph = np.array( scaler.fit_transform(df.loc[Ivalid,nuc_columns]) )
# pca = PCA()
# nuc_sph_PCA = pca.fit_transform(nuc_sph)

# loading_matrix = pd.DataFrame(pca.components_, columns=nuc_columns)

#%% Visualize the different PCA dimensions

# pl = pv.Plotter()

# for comp in range(0,10):
    
#     new_dict = dict(zip(loading_matrix.columns.str.split('nuc_',expand=True).get_level_values(1),loading_matrix.loc[comp]))
#     M = shtools.convert_coeffs_dict_to_matrix(new_dict,lmax=10)
#     mesh,_ = shtools.get_even_reconstruction_from_coeffs(M,npoints=1024)
    
    
#     pl.add_mesh(pv.wrap(mesh), opacity=0.5)

# pl.show_axes()
# pl.show()

# pl.save_graphic(path.join(dirname,f'shape_mode_analysis/nuc_modes/comp_{comp}.pdf'))







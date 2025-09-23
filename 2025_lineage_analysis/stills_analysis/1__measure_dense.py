#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:48:47 2025

@author: xies
"""

# Core libraries
import numpy as np
from skimage import io, util, morphology, exposure
from scipy.spatial import Voronoi, Delaunay
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

# Specific utils
from imageUtils import draw_labels_on_image, colorize_segmentation, normalize_exposure_by_axis
from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, \
    discrete_mean_curvature_measure, sphere_ball_intersection
import pyvista as pv

# General utils
from tqdm import tqdm
from os import path,makedirs
import pickle as pkl

dx = 0.4
dz = 1
Z_SHIFT = 10
KAPPA = 5 # microns

# for expansion
footprint = morphology.cube(3)

# Filenames
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/Shared/K10 paw/K10-R2'

# Activate measurement suites
['cyto','intensity']


# Load segmentations
cyto_seg = io.imread(path.join(dirname,'Cropped/R_cp_cleaned.tif'))

# Load raw channels
collagen = io.imread(path.join(dirname,'Cropped/B.tif'))
collagen = normalize_exposure_by_axis(collagen,axis=0)
k10 = io.imread(path.join(dirname,'Cropped/G.tif'))
k10 = normalize_exposure_by_axis(k10,axis=0)
membrane = io.imread(path.join(dirname,'Cropped/R.tif'))
membrane = normalize_exposure_by_axis(membrane,axis=0)

ZZ,YY,XX = cyto_seg.shape

#%%

from measurements import measure_cyto_geometry_from_regionprops, measure_cyto_intensity, \
            measure_flat_cyto_from_regionprops, reslice_by_heightmap, \
            estimate_sh_coefficients, find_distance_to_closest_point, \
            measure_collagen_structure, get_mesh_from_bm_image, get_tissue_curvatures
from imageUtils import colorize_segmentation, filter_seg_by_largest_object

VISUALIZE = False

flat_collagen_top_margin = 2
flat_collagen_bottom_margin = 8

flat_cyto_top_margin = -60 #NB: top -> more apical but lower z-index
flat_cyto_bottom_margin = 5

LMAX = 5 # Number of spherical harmonics components to calculate

# --- 1. Filter seg by largest object ---
cyto_seg = filter_seg_by_largest_object(cyto_seg)
io.imsave(path.join(dirname,'Cropped/R_cp_cleaned_by_largest.tif'),cyto_seg)

# --- 2. Voxel-based cell geometry measurements ---
df_cyto = measure_cyto_geometry_from_regionprops(cyto_seg,spacing = [dz,dx,dx])
df_cyto = df_cyto.rename(columns={'X-cyto':'X','Y-cyto':'Y','Z-cyto':'Z'})
int_images = {'K10':k10,'Collagen':collagen,'Membrane':membrane}
intensity_df = measure_cyto_intensity(cyto_seg,int_images)
df = pd.merge(left=df_cyto,right=intensity_df,left_on='TrackID',right_on='TrackID',how='left')

# ----- 3. Generate flattened 3d cortical segmentation and measure geometry and collagen
# from cell-centric coordinates ----
heightmap = io.imread(path.join(dirname,'Image flattening/heightmap.tif'))

flat_cyto = reslice_by_heightmap( cyto_seg, heightmap,
                            top_border=flat_cyto_top_margin, bottom_border=flat_cyto_bottom_margin)
flat_cyto = flat_cyto.astype(np.int16)
flat_collagen = reslice_by_heightmap( cyto_seg, heightmap,
                            top_border=flat_collagen_top_margin, bottom_border=flat_collagen_bottom_margin)
flat_collagen = flat_collagen.mean(axis=0)

io.imsave(path.join(dirname,'Image flattening/flat_cyto.tif'), flat_cyto.astype(np.uint16),check_contrast=False)
io.imsave(path.join(dirname,'Image flattening/flat_collagen.tif'), flat_collagen.astype(np.uint16),check_contrast=False)

# Calculate collagen structuring matrix
(Jxx,Jxy,Jyy) = measure_collagen_structure(flat_collagen,blur_sigma=3)

df_flat,basal_masks_2save = measure_flat_cyto_from_regionprops(
    flat_cyto, flat_collagen, (Jxx, Jyy, Jxy), spacing = [dz,dx,dx])
df = pd.merge(df,df_flat,left_on='TrackID',right_on='TrackID',how='left')

io.imsave(path.join(dirname,'Image flattening/basal_masks.tif'),basal_masks_2save)

# Book-keeping
df['X-pixels'] = df['X'] / dx
df['Y-pixels'] = df['Y'] / dx

dense_coords = np.array([df['Y-pixels'],df['X-pixels']]).T
dense_coords_3d_um = np.array([df['Z'],df['Y'],df['X']]).T

#----- 4. Nuc-to-BM heights -----
# Load heightmap and calculate adjusted height
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
bm_height_image = io.imread(path.join(dirname,'Image flattening/height_image.tif'))
bg_mesh = get_mesh_from_bm_image(bm_height_image,spacing=[dz,dx,dx],decimation_factor=30)
closest_mesh_to_cell,_,_ = bg_mesh.nearest.on_surface(dense_coords_3d_um[:,::-1])
mean_curve, gaussian_curve = get_tissue_curvatures(bg_mesh,kappa=KAPPA, query_pts = closest_mesh_to_cell)

df['Mean curvature'] = mean_curve
df['Gaussian curvature'] = gaussian_curve

# --- 6. 3D shape decomposition ---

# 2a: Estimate cell and nuclear mesh using spherical harmonics
sh_coefficients = estimate_sh_coefficients(cyto_seg, LMAX, spacing = [dz,dx,dx])
sh_coefficients = sh_coefficients.set_index('TrackID')
sh_coefficients.columns = 'cyto_' + sh_coefficients.columns
df = pd.merge(df,sh_coefficients,left_on='TrackID',right_on='TrackID',how='left')

# Merge with manual annotations
# df = df.reset_index()
# df = df.set_index(['Frame','TrackID'])

#%% Colorize cell with measurement for visualization

measurement = df['Mean curvature']
measurement /= np.max([measurement.max(),np.abs(measurement.min())])
colorized = colorize_segmentation(cyto_seg,measurement.to_dict(),dtype=float)
io.imsave('/Users/xies/Desktop/curvature.tif',
          util.img_as_int(exposure.rescale_intensity(colorized,in_range=(-1,1)) ) )



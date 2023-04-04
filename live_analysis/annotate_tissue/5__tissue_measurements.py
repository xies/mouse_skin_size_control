#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 00:02:34 2022

@author: xies
"""

import numpy as np
from skimage import io, measure, draw, util
from scipy.spatial import distance, Voronoi, Delaunay
import pandas as pd

from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
from basicUtils import euclidean_distance

import matplotlib.pylab as plt
# from matplotlib.path import Path
# from roipoly import roipoly
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, most_likely_label, colorize_segmentation
from mathUtils import *

from tqdm import tqdm
from glob import glob
from os import path
import csv

dx = 0.25
XX = 460
Z_SHIFT = 10

# Differentiating thresholds
centroid_height_cutoff = 3.5 #microns above BM

VISUALIZE = True
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

'''
NB: idx - the order in array in dense segmentation

'''

#%% Load the segmentation and coordinates

def find_differentiating_cells(df,height_cutoff,heightmap):
    
    # I = df['Height to BM'] > height_cutoff
    diff_height_th = heightmap[np.round(
        df['Y-pixels']).astype(int),np.round(df['X-pixels']).astype(int)] - height_cutoff
    
    # Check that the bottom bbox of nucleus is not within cutoff of the 'heightmap'
    I = df['Nuclear bbox bottom'] < diff_height_th
    
    df['Differentiating'] = False
    df.loc[I,'Differentiating'] = True
    
    return df

# convert trianglulation to adjacency matrix (for easy editing)
def tri_to_adjmat(tri):
    num_verts = max(map(max,tri.simplices)) + 1
    A = np.zeros((num_verts,num_verts),dtype=bool)
    for idx in range(num_verts):
        neighbor_idx = get_neighbor_idx(tri,idx)
        A[idx,neighbor_idx] = True
    return A

#%%

df = []

for t in tqdm(range(15)):
    
    dense_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    manual_tracks = io.imread(path.join(dirname,f'manual_basal_tracking/t{t}.tif'))
    
    #NB: only use 0-index for the df_dense dataframe
    df_dense = pd.DataFrame( measure.regionprops_table(dense_seg,
                                                       properties=['label','area','centroid','solidity','bbox']))
    df_dense = df_dense.rename(columns={'centroid-0':'Z','centroid-1':'Y-pixels','centroid-2':'X-pixels'
                                        ,'label':'CellposeID','area':'Nuclear volume','bbox-0':'Nuclear bbox top'
                                        ,'bbox-3':'Nuclear bbox bottom'
                                        ,'solidity':'Nuclear solidity'})
    df_dense = df_dense.drop(columns=['bbox-1','bbox-2','bbox-4','bbox-5'])
    df_dense['Nuclear volume'] = df_dense['Nuclear volume'] * dx**2
    df_dense['X'] = df_dense['X-pixels'] * dx**2
    df_dense['Y'] = df_dense['Y-pixels'] * dx**2
    df_dense['Frame'] = t
    df_dense['basalID'] = np.nan
    
    #@todo: include thresholded volumes
    # Load thresholded images
    

    #NB: best to use the manual mapping since it guarantees one-to-one mapping from cellpose to manual cellIDs
    df_manual = pd.DataFrame(measure.regionprops_table(manual_tracks,intensity_image = dense_seg,
                                                       properties = ['label'],
                                                       extra_properties = [most_likely_label]))
    df_manual = df_manual.rename(columns={'label':'basalID','most_likely_label':'CellposeID'})
    assert(np.isnan(df_manual['CellposeID']).sum() == 0)
    
    # Reverse the mapping from CellposeID to basalID
    for _,this_cell in df_manual.iterrows():
         df_dense.loc[ df_dense['CellposeID'] == this_cell['CellposeID'],'basalID'] = this_cell['basalID']

    dense_coords = np.array([df_dense['Y-pixels'],df_dense['X-pixels']]).T
    dense_coords_3d_um = np.array([df_dense['Z'],df_dense['Y'],df_dense['X']]).T
    
    # Load heightmap and calculate adjusted height
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    heightmap_shifted = heightmap + Z_SHIFT
    df_dense['Height to BM'] = heightmap_shifted[np.round(df_dense['Y']).astype(int),np.round(df_dense['X']).astype(int)] - df_dense['Z']
    
    # Based on adjusted height, determine a 'cutoff'
    # df_dense['Differentiating'] = df_dense['Height to BM'] > HEIGHT_CUTOFF
    df_dense = find_differentiating_cells(df_dense,centroid_height_cutoff,heightmap)
    
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
    border_nuclei = df_dense.loc[Iborder]['CellposeID'].values
    df_dense['Border'] = False
    df_dense.loc[ np.in1d(df_dense['CellposeID'],border_nuclei), 'Border'] = True
    
    # Generate 3D mesh for curvature analysis -- no need to specify precise cell-cell junctions
    Z,Y,X = dense_coords_3d_um.T
    mesh = Trimesh(vertices = np.array([X,Y,Z]).T, faces=tri_dense.simplices)
    # mesh_sm = trimesh.smoothing.filter_laplacian(mesh,lamb=0.01)
    mean_curve = discrete_mean_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    gaussian_curve = discrete_gaussian_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    df_dense['Mean curvature'] = mean_curve
    df_dense['Gaussian curvature'] = gaussian_curve
    
    # Load the actual neighborhood topology
    A = np.load(path.join(dirname,f'Image flattening/flat_adj/adjmat_t{t}.npy'))
    D = distance.squareform(distance.pdist(dense_coords_3d_um))
    
    A_diff = A.copy()
    A_diff[:,~df_dense['Differentiating']] = 0
    A_diff = A_diff + A_diff.T
    A_diff[A_diff > 1] = 1
    
    A_planar = A - A_diff
    
    # Resave adjmat as planar v. diff
    im = draw_adjmat_on_image(A_planar,dense_coords,[XX,XX])
    io.imsave(path.join(dirname,f'Image flattening/flat_adj/t{t}_planar.tif'),im.astype(np.uint16),check_contrast=False)
    im = draw_adjmat_on_image(A_diff,dense_coords,[XX,XX])
    io.imsave(path.join(dirname,f'Image flattening/flat_adj/t{t}_diff.tif'),im.astype(np.uint16),check_contrast=False)
    
    df_dense['Nuclear surface area'] = np.nan
    df_dense['Nuclear axial component'] = np.nan
    df_dense['Nuclear axial angle'] = np.nan
    df_dense['Nuclear planar component 1'] = np.nan
    df_dense['Nuclear planar component 2'] = np.nan
    df_dense['Nuclear axial eccentricity'] = np.nan
    df_dense['Nuclear planar eccentricity'] = np.nan
    df_dense['Nuclear bbox top'] = np.nan
    df_dense['Nuclear bbox bottom'] = np.nan

    df_dense['Mean neighbor dist'] = np.nan
    df_dense['Mean neighbor nuclear volume'] = np.nan
    df_dense['Std neighbor nuclear volume'] = np.nan
    df_dense['Coronal area'] = np.nan
    df_dense['Coronal angle'] = np.nan
    df_dense['Coronal eccentricity'] = np.nan
    df_dense['Mean planar neighbor height'] = np.nan
    df_dense['Mean diff neighbor height'] = np.nan
    
    # Use this to make specific neighborhood measurements
    props = measure.regionprops(dense_seg,extra_properties = [surface_area])
    for i,this_cell in df_dense.iterrows(): #NB: i needs to be 0-index
        
        bbox = props[i]['bbox']
        df_dense['Nuclear bbox top'] = bbox[0]
        df_dense['Nuclear bbox bottom'] = bbox[3]
    
        I = props[i]['inertia_tensor']
        SA = props[i]['surface_area'] * dx**2
        Iaxial,phi,Ia,Ib,theta = parse_3D_inertial_tensor(I)
        df_dense.at[i,'Nuclear surface area'] = SA  * dx**2
        df_dense.at[i,'Nuclear axial component'] = Iaxial
        df_dense.at[i,'Nuclear axial angle'] = phi
        df_dense.at[i,'Nuclear planar component 1'] = Ia
        df_dense.at[i,'Nuclear planar component 2'] = Ib
        df_dense.at[i,'Nuclear axial eccentricity'] = Ia/Iaxial
        df_dense.at[i,'Nuclear planar eccentricity'] = Ib/Ia
        df_dense.at[i,'Nuclear planar orientation'] = theta
        
        # Use neighbor matrices
        planar_neighbor_idx = np.where(A_planar[i,:])[0]
        diff_neighbor_idx = np.where(A_diff[i,:])[0]
        df_dense.at[i,'Num planar neighbors'] = len(planar_neighbor_idx)
        df_dense.at[i,'Num diff neighbors'] = len(diff_neighbor_idx)
        
        if len(np.hstack([diff_neighbor_idx,planar_neighbor_idx])) > 0:
            neighbor_heights = df_dense.loc[np.hstack([diff_neighbor_idx,planar_neighbor_idx])]['Height to BM']
            df_dense.at[i,'Mean neighbor height'] = neighbor_heights.mean()
            
        # Differentiating neighbor heights
        if len(diff_neighbor_idx) > 0:
            neighbor_heights = df_dense.loc[diff_neighbor_idx]['Height to BM']
            df_dense.at[i,'Mean diff neighbor height'] = neighbor_heights.mean()
            
        # +1 Planar neighbors (i.e. those that are still basal)
        if len(planar_neighbor_idx) > 0:
            
            neighbor_heights = df_dense.loc[planar_neighbor_idx]['Height to BM']
            df_dense.at[i,'Mean planar neighbor height'] = neighbor_heights.mean()
            
            neighbor_dists = D[i, planar_neighbor_idx]
            df_dense.at[i,'Mean neighbor dist'] = neighbor_dists.mean()
                    
            neighbor_dists = D[i, planar_neighbor_idx]
            df_dense.at[i,'Mean neighbor nuclear volume'] = df_dense.iloc[planar_neighbor_idx]['Nuclear volume'].mean()
            df_dense.at[i,'Std neighbor nuclear volume'] = df_dense.iloc[planar_neighbor_idx]['Nuclear volume'].std()
                
            # get 2d coronal area
            X = dense_coords[planar_neighbor_idx,1]
            Y = dense_coords[planar_neighbor_idx,0]
            if len(X) > 2:
                order = argsort_counter_clockwise(X,Y)
                X = X[order]
                Y = Y[order]
                im = np.zeros([XX,XX])
                rr,cc = draw.polygon(Y,X)
                im[rr,cc] = 1
                p = measure.regionprops(im.astype(int))[0]
                df_dense.at[i,'Coronal area'] = p['area']  * dx**2
                df_dense.at[i,'Coronal eccentricity'] = p['eccentricity']
                theta = np.rad2deg(p['orientation'])
                if theta < 0:
                    theta = theta + 180
                df_dense.at[i,'Coronal angle'] = theta
                # df_dense.at[i,'Coronal '
                
    
    # Save the DF
    df.append(df_dense)
    
    #Save a bunch of intermediates
    # Save segmentation with text labels @ centroid
    # im_cellposeID = draw_labels_on_image(dense_coords,df_dense['basalID'],[XX,XX],font_size=12)
    # im_cellposeID.save(path.join(dirname,f'3d_nuc_seg/cellposeIDs/t{t}.tif'))
    
    df_dense_ = df_dense.loc[ ~np.isnan(df_dense['basalID']) ]
    # colorized = colorize_segmentation(dense_seg,
    #                                   {k:v for k,v in zip(df_dense_['CellposeID'].values,df_dense_['basalID'].values)})
    # io.imsave(path.join(dirname,f'3d_nuc_seg/cellposeIDs/t{t}.tif'),colorized,check_contrast=False)
    
    colorized = colorize_segmentation(dense_seg,
                                      {k:v for k,v in zip(df_dense['CellposeID'].values,df_dense['Differentiating'].values)})
    io.imsave(path.join(dirname,f'3d_nuc_seg/Differentiating/t{t}.tif'),colorized.astype(int8),check_contrast=False)
    
    # colorized = colorize_segmentation(dense_seg.astype(float),
    #                                   {k:v for k,v in zip(df_dense['CellposeID'].values,df_dense['Height to BM'].values)})
    # io.imsave(path.join(dirname,f'3d_nuc_seg/height_to_BM/t{t}.tif'), \
    #           util.img_as_uint(colorized/colorized.max()),check_contrast=False)
    
    
    
df = pd.concat(df,ignore_index=True)

#%
df.to_csv(path.join(dirname,'tissue_dataframe.csv'))

#%%
 
    # Colorize nuclei based on mean curvature (for inspection)
    # mean_colors = (mean-mean.min())/mean.max()
    # colorized = colorize_segmentation(dense_seg,{k:v for k ,v in zip(df_dense['label'].values, mean)})
 

    #% Compute local curvature
    # Visualize mesh
    # from mpl_toolkits.mplot3d import Axes3D as ax3d
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_trisurf(dense_coords_3d[:,1],dense_coords_3d[:,2],Z,cmap=plt.cm.viridis)
    # ax.scatter(dense_coords_3d[:,1],dense_coords_3d[:,2],local_neighborhood,color='k')
    

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
from imageUtils import draw_labels_on_image, draw_adjmat_on_image
from mathUtils import argsort_counter_clockwise, polygon_area, surface_area, parse_3D_inertial_tensor

from tqdm import tqdm
from glob import glob
from os import path
import csv

XX = 460
VISUALIZE = True
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

'''
NB: idx - the order in array in dense segmentation

'''


#%% Load the segmentation and coordinates

def get_neighbor_idx(tri,idx):
    neighbors = np.unique(tri.simplices[np.any(tri.simplices == idx,axis=1),:])
    neighbors = neighbors[neighbors != idx] # return only nonself
    return neighbors.astype(int)

def get_neighbor_distances(tri,idx,coords):
    neighbor_idx = get_neighbor_idx(tri,idx)
    this_coord = coords[idx,:]
    neighbor_coords = coords[neighbor_idx,:]
    D = np.array([euclidean_distance(this_coord,n) for n in neighbor_coords])
    return D

def colorize_segmentation(seg,value_dict):
    '''
    Given a segmentation label image, colorize the segmented labels using a dictionary of label: value
    
    (Or you instead use)
    '''
    
    assert( len(np.unique(seg)[1:] == len(value_dict)) )
    colorized = np.zeros_like(seg)
    for k,v in value_dict.items():
        colorized[seg == k] = v
    return colorized
    
# convert trianglulation to adjacency matrix (for easy editing)
def tri_to_adjmat(tri):
    num_verts = max(map(max,tri.simplices)) + 1
    A = np.zeros((num_verts,num_verts),dtype=bool)
    for idx in range(num_verts):
        neighbor_idx = get_neighbor_idx(tri,idx)
        A[idx,neighbor_idx] = True
    return A

# Construct triangulation
def adjmat2triangle(G):
    triangles = set()
    for u,w in G.edges:
        for v in set(G.neighbors(u)).intersection(G.neighbors(w)):
            triangles.add(frozenset([u,v,w]))
    return triangles


def most_likely_label(labeled,im):
    label = 0
    if len(im[im>0]) > 0:
        unique,counts = np.unique(im[im > 0],return_counts=True)
        label = unique[counts.argmax()]
        if label == 0:
            label = np.nan
    return label

    
#%%

df = []

for t in tqdm(range(15)):
    
    dense_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    manual_tracks = io.imread(path.join(dirname,f'manual_basal_tracking/t{t}.tif'))
    
    #NB: only use 0-index for the df_dense dataframe
    df_dense = pd.DataFrame( measure.regionprops_table(dense_seg,
                                                       properties=['label','area','centroid','solidity']))
    df_dense = df_dense.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X'
                                        ,'label':'CellposeID','area':'Nuclear volume'
                                        ,'solidity':'Nuclear solidity'})
    df_dense['Frame'] = t
    df_dense['basalID'] = np.nan

    #NB: best to use this since it guarantees one-to-one 
    df_manual = pd.DataFrame(measure.regionprops_table(manual_tracks,intensity_image = dense_seg,
                                                       properties = ['label'],
                                                       extra_properties = [most_likely_label]))
    df_manual = df_manual.rename(columns={'label':'basalID','most_likely_label':'CellposeID'})
    assert(np.isnan(df_manual['CellposeID']).sum() == 0)
    
    # Reverse the mapping
    for _,this_cell in df_manual.iterrows():
         df_dense.loc[ df_dense['CellposeID'] == this_cell['CellposeID'],'basalID'] = this_cell['basalID']

    dense_coords = np.array([df_dense['Y'],df_dense['X']]).T
    dense_coords_3d = np.array([df_dense['Z'],df_dense['Y'],df_dense['X']]).T
    
    # Load heightmap and calculate adjusted height
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    df_dense['Height to BM'] = df_dense['Z'] - heightmap[np.round(df_dense['Y']).astype(int),np.round(df_dense['X']).astype(int)]
    
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
    
    Z,Y,X = dense_coords_3d.T
    mesh = Trimesh(vertices = np.array([X,Y,Z]).T, faces=tri_dense.simplices)
    # mesh_sm = trimesh.smoothing.filter_laplacian(mesh,lamb=0.01)
    mean_curve = discrete_mean_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    gaussian_curve = discrete_gaussian_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    df_dense['Mean curvature'] = mean_curve
    df_dense['Gaussian curvature'] = gaussian_curve
    
    # Load the actual neighborhood topology
    A = np.load(path.join(dirname,f'Image flattening/flat_adj/adjmat_t{t}.npy'))
    D = distance.squareform(distance.pdist(dense_coords_3d))
    
    df_dense['Nuclear surface area'] = np.nan
    df_dense['Nuclear axial component'] = np.nan
    df_dense['Nuclear axial angle'] = np.nan
    df_dense['Nuclear planar component 1'] = np.nan
    df_dense['Nuclear planar component 2'] = np.nan
    df_dense['Num neighbors'] = np.nan
    df_dense['Mean neighbor dist'] = np.nan
    df_dense['Min neighbor dist'] = np.nan
    df_dense['Max neighbor dist'] = np.nan
    df_dense['Coronal area'] = np.nan
    df_dense['Coronal angle'] = np.nan
    df_dense['Coronal eccentricity'] = np.nan
    
    props = measure.regionprops(dense_seg,extra_properties = [surface_area])
    for i,this_cell in df_dense.iterrows(): #NB: i needs to be 0-index
        
        I = props[i]['inertia_tensor']
        SA = props[i]['surface_area']
        Iaxial,phi,Ia,Ib,theta = parse_3D_inertial_tensor(I)
        df_dense.at[i,'Nuclear surface area'] = SA
        df_dense.at[i,'Nuclear axial component'] = Iaxial
        df_dense.at[i,'Nuclear axial angle'] = phi
        df_dense.at[i,'Nuclear planar component 1'] = Ia
        df_dense.at[i,'Nuclear planar component 2'] = Ib
        df_dense.at[i,'Nuclear planar orientation'] = theta
        
        neighbor_idx = np.where(A[i,:])[0]
        df_dense.at[i,'Num neighbors'] = len(neighbor_idx)
        if len(neighbor_idx) > 0:
            neighbor_dists = D[i, neighbor_idx]
            df_dense.at[i,'Mean neighbor dist'] = neighbor_dists.mean()
            df_dense.at[i,'Min neighbor dist'] = neighbor_dists.min()
            df_dense.at[i,'Max neighbor dist'] = neighbor_dists.max()
            
            # get 2d coronal area
            X = dense_coords[neighbor_idx,1]
            Y = dense_coords[neighbor_idx,0]
            if len(X) > 2:
                order = argsort_counter_clockwise(X,Y)
                X = X[order]
                Y = Y[order]
                im = np.zeros([XX,XX])
                rr,cc = draw.polygon(Y,X)
                im[rr,cc] = 1
                p = measure.regionprops(im.astype(int))[0]
                df_dense.at[i,'Coronal area'] = p['area']
                df_dense.at[i,'Coronal eccentricity'] = p['eccentricity']
                theta = np.rad2deg(p['orientation'])
                if theta < 0:
                    theta = theta + 180
                df_dense.at[i,'Coronal angle'] = theta
                # df_dense.at[i,'Coronal '
            
                    
    df_dense['Coronal density'] = df_dense['Num neighbors'] / df_dense['Coronal area']
    
    # Save the DF
    df.append(df_dense)
    
    #Save a bunch of intermediates
    # Save segmentation with text labels @ centroid
    # im_cellposeID = draw_labels_on_image(dense_coords,df_dense['basalID'],[XX,XX],font_size=12)
    # im_cellposeID.save(path.join(dirname,f'3d_nuc_seg/cellposeIDs/t{t}.tif'))
    
    df_dense_ = df_dense.loc[ ~np.isnan(df_dense['basalID']) ]
    colorized = colorize_segmentation(dense_seg,
                                      {k:v for k,v in zip(df_dense_['CellposeID'].values,df_dense_['basalID'].values)})
    io.imsave(path.join(dirname,f'3d_nuc_seg/cellposeIDs/t{t}.tif'),colorized,check_contrast=False)
    
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
    

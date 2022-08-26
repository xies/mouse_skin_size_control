#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 00:02:34 2022

@author: xies
"""

import numpy as np
from skimage import io, measure
from glob import glob
from os import path
from scipy.spatial import distance, Voronoi, Delaunay
from re import findall
import pandas as pd
import matplotlib.pylab as plt
from tqdm import tqdm
# from matplotlib.path import Path
# from roipoly import roipoly
from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
    

XX = 460
VISUALIZE = True
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/3d_segmentation/'

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

def euclidean_distance(X,Y):
    X = np.array(X,dtype=float)
    Y = np.array(Y,dtype=float)
    assert(X.ndim == Y.ndim)
    return np.sqrt( np.sum((X-Y)**2) )


#%%

df = []
for t in range(12):
    
    dense_seg = io.imread(path.join(dirname,f'naive_tracking/t{t}.tif'))
    growth_curves = io.imread(path.join(dirname,f'manual_basal_tracking/t{t}.tif'))
    
    df_central = pd.DataFrame( measure.regionprops_table(growth_curves, properties=['label','area','centroid']))
    df_central = df_central.rename(columns={'label':'ManualID'
                                            ,'centroid-0':'Z'
                                            ,'centroid-1':'Y'
                                            ,'centroid-2':'X'})
    
    # Find central cell shape tensors
    inertial_tensors = []
    df_central['Shape ratio 1'] = np.nan
    df_central['Shape ratio 2'] = np.nan
    for i,this_cell in df_central.iterrows():
        manualID = this_cell['ManualID']
        I = measure.inertia_tensor(growth_curves == manualID)
        eig_val, _ = np.linalg.eig(I)
        eig_val = sorted(eig_val)
        df_central.at[i,'Shape ratio 1'] = eig_val[2] / eig_val[1]
        df_central.at[i,'Shape ratio 2'] = eig_val[1] / eig_val[0]
        
    
    df_dense = pd.DataFrame( measure.regionprops_table(dense_seg, intensity_image = growth_curves,
                                                       properties=['label','area','centroid','max_intensity']))
    df_dense = df_dense.rename(columns={'max_intensity':'ManualID'
                                            ,'centroid-0':'Z'
                                            ,'centroid-1':'Y'
                                            ,'centroid-2':'X'})
    
    dense_coords = np.array([df_dense['Y'],df_dense['X']]).T
    dense_coords_3d = np.array([df_dense['Z'],df_dense['Y'],df_dense['X']]).T
    
    manual_coords_3d = np.array([df_central['Z'],df_central['Y'],df_central['X']]).T
    
    df_central['Frame'] = t+1
    
    
    #@todo: Load heightmap and calculate adjusted height
    
    
    #% Use Delaunay triangulation in 2D to approximate the basal layer topology
    
    tri = Delaunay(dense_coords)
    
    if VISUALIZE and t==1:
        
        plt.figure()
        plt.triplot(dense_coords[:,1], dense_coords[:,0], tri.simplices,'r-')
        # plt.plot(dense_coords[118,1], dense_coords[118,0], 'ko')
        io.imshow(dense_seg.max(axis=0))
        plt.show()
    
    # Use the dual Voronoi to get rid of the border/infinity cells
    
    vor = Voronoi(dense_coords)
    # Find the voronoi regions that touch the image border
    vertices_outside = np.where(np.any((vor.vertices < 0) | (vor.vertices > XX),axis=1))
    regions_outside = np.where([ np.any(np.in1d(np.array(r), vertices_outside)) for r in vor.regions])
    
    regions_outside = np.hstack([regions_outside, np.where([-1 in r for r in vor.regions])])
    Iborder = np.in1d(vor.point_region, regions_outside)
    border_nuclei = df_dense.loc[Iborder]['label'].values
    
    df_dense['Border'] = False
    df_dense.loc[ np.in1d(df_dense['label'],border_nuclei), 'Border'] = True
    
    
    # if VISUALIZE and t==1:
    #     _im = np.zeros_like(dense_seg.max(axis=0))
    #     for n in border_nuclei:
    #         _im[dense_seg.max(axis=0) == n] = n
    #     plt.figure()
    #     plt.plot(dense_coords[Iborder,1], dense_coords[Iborder,0], 'ko')
    #     # spatial.voronoi_plot_2d(vor)
    #     io.imshow(dense_seg.max(axis=0))
    #     # io.imshow((_im > 0).astype(int))
    #     # plt.show()
    
    
    #%
    # get # of neighbors from triangulation
    num_neighbors = [len(get_neighbor_idx(tri,i)) for i in range(len(df_dense))]
    df_dense['Num basal neighbors'] = num_neighbors
    
    #% Find other geometries in 3D
    # Transfer from dense df to sparse DF
    
    df_dense['Neighbor mean dist'] = np.nan
    df_dense['Neighbor max dist'] = np.nan
    df_dense['Neighbor min dist'] = np.nan
    
    # Get distribution of Euclidean distances (in 3D) to the neighbors
    # Then take min mean max
    neighbors = np.zeros_like(dense_seg)
    for i,this_cell in df_dense.iterrows():
        
        this_coord = np.array([this_cell['Z'],this_cell['Y'],this_cell['X']])
        neighbor_idx = np.unique(get_neighbor_idx(tri,i))
        Ineighbor_on_border = df_dense.iloc[neighbor_idx]['Border']
        D = get_neighbor_distances(tri,i,dense_coords_3d)
        D = D[~Ineighbor_on_border]
        if len(D) == 0:
            continue
        
        df_dense.at[i,'Neighbor mean dist'] = D.mean()
        df_dense.at[i,'Neighbor max dist'] = D.max()
        df_dense.at[i,'Neighbor min dist'] = D.min()
        
        if VISUALIZE and t == 1:
            neighbors[np.in1d(dense_seg.flatten(), df_dense.iloc[neighbor_idx]['label']).reshape(dense_seg.shape)] \
                = this_cell['ManualID']

    if VISUALIZE and t==1:
        io.imsave(f'/Users/xies/Desktop/t{t}_neighbors.tif', neighbors)
    
    
    #% Compute local curvature
    # Visualize mesh
    # from mpl_toolkits.mplot3d import Axes3D as ax3d
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_trisurf(dense_coords_3d[:,1],dense_coords_3d[:,2],Z,cmap=plt.cm.viridis)
    # ax.scatter(dense_coords_3d[:,1],dense_coords_3d[:,2],local_neighborhood,color='k')
    
    Z,Y,X = dense_coords_3d.T
    mesh = Trimesh(vertices = np.array([X,Y,Z]).T, faces=tri.simplices)
    # mesh_sm = trimesh.smoothing.filter_laplacian(mesh,lamb=0.01)
    
    mean = discrete_mean_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    gaussian = discrete_gaussian_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    df_dense['Mean curvature'] = mean
    df_dense['Gaussian curvature'] = gaussian
    
    # Colorize nuclei based on mean curvature (for inspection)
    # mean_colors = (mean-mean.min())/mean.max()
    # colorized = np.zeros_like(dense_seg,dtype=float)
    # for i,this_cell in df_dense.iterrows():
    #     if not this_cell['Border']:
    #         colorized[dense_seg == this_cell['label']] = mean_colors[i]

    df_ = pd.merge(df_dense,df_central,how='inner',on=['ManualID'])
    df_ = df_.drop(columns=['Z_y','Y_y','X_y'])
    df_ = df_.rename(columns={'Z_x':'Z','Y_x':'Y','X_x':'X',
                              'label':'CellposeID',
                              'area_y':'Cell volume','area_x':'Cellpose volume'})

    df.append(df_)

df_dense = pd.concat(df,ignore_index=True)





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

VISUALIZE = True
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/3d_segmentation/'

#%% Load the segmentation and coordinates

t = 1

dense_seg = io.imread(path.join(dirname,f'naive_tracking/t{t}.tif'))
growth_curves = io.imread(path.join(dirname,f'manual_basal_tracking/t{t}.tif'))

df_central = pd.DataFrame( measure.regionprops_table(growth_curves, properties=['label','area','centroid']))
df_central = df_central.rename(columns={'label':'ManualID'
                                        ,'centroid-0':'Z'
                                        ,'centroid-1':'Y'
                                        ,'centroid-2':'X'})
df_dense = pd.DataFrame( measure.regionprops_table(dense_seg, intensity_image = growth_curves,
                                                   properties=['label','area','centroid','max_intensity']))
df_dense = df_dense.rename(columns={'max_intensity':'ManualID'
                                        ,'centroid-0':'Z'
                                        ,'centroid-1':'Y'
                                        ,'centroid-2':'X'})

dense_coords = np.array([df_dense['Y'],df_dense['X']]).T
dense_coords_3d = np.array([df_dense['Z'],df_dense['Y'],df_dense['X']]).T

manual_coords_3d = np.array([df_central['Z'],df_central['Y'],df_central['X']]).T

#%% Use Delaunay triangulation in 2D to approximate the basal layer topology

tri = Delaunay(dense_coords)

if VISUALIZE:
    
    plt.triplot(dense_coords[:,1], dense_coords[:,0], tri.simplices,'r-')
    plt.plot(dense_coords[:,1], dense_coords[:,0], 'ro')
    io.imshow(dense_seg.max(axis=0))
    # plt.show()
    
def get_neighbor_coords(tri,idx):
    return tri.simplices[np.any(tri.simplices == idx,axis=1),:]

# get # of neighbors from triangulation
num_neighbors = [len(get_neighbor_coords(tri,i)) for i in range(len(df_dense))]

df_dense['Num basal neighbors'] = num_neighbors

#%% Find other geometries in 3D
# Transfer from dense df to sparse DF
df_central['DenseID'] = [df_dense[df_dense['ManualID'] == row['ManualID']].iloc[0]['label'] for i,row in df_central.iterrows()]
df_central['DenseID'] = [df_dense[df_dense['ManualID'] == row['ManualID']].iloc[0]['Num basal neighbors'] for i,row in df_central.iterrows()]





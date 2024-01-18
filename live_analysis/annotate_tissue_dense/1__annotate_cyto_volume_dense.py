#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:46:23 2024

@author: xies
"""

import numpy as np
from skimage import io, measure, draw, util, morphology
# from scipy.spatial import distance, Voronoi, Delaunay
import pandas as pd

# from trimesh import Trimesh
# from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
from basicUtils import euclidean_distance

import matplotlib.pylab as plt
# from matplotlib.path import Path
# from roipoly import roipoly
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, most_likely_label, colorize_segmentation
# from mathUtils import get_neighbor_idx, surface_area, parse_3D_inertial_tensor, argsort_counter_clockwise

from tqdm import tqdm
# from glob import glob
from os import path

dx = 0.25
XX = 460
Z_SHIFT = 10

# Differentiating thresholds
centroid_height_cutoff = 3.5 #microns above BM
""
SAVE = True
VISUALIZE = True
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

# FUCCI threshold (in stds)
alpha_threshold = 1
#NB: idx - the order in array in dense segmentation


#%%

df = []

for t in [6,7,8,9]:
# t = 8
    
    nuc_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    cyto_seg = io.imread(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual/t{t}.tif'))
    manual_tracks = io.imread(path.join(dirname,f'manual_basal_tracking/sequence/t{t}.tif'))
    
    #% Label transfer from nuc3D -> cyto3D
    
    # For now detect the max overlap from cyto3d label with the nuc3d label
    df_nuc = pd.DataFrame( measure.regionprops_table(nuc_seg, intensity_image = cyto_seg
                                                   ,properties=['label','centroid','max_intensity',
                                                                'euler_number','area']
                                                   ,extra_properties = [most_likely_label]))
    
    df_nuc = df_nuc.rename(columns={'centroid-0':'Y','centroid-1':'X','area':'Nuclear volume'
                                      ,'most_likely_label':'CytoID','label':'CellposeID'})
    
    
    df_cyto = pd.DataFrame( measure.regionprops_table(cyto_seg
                                                      , properties=['centroid','label','area']))
    df_cyto = df_cyto.rename(columns={'area':'Cell volume (auto)'})
    df_cyto.index = df_cyto['label']
    
    nuc_coords = np.array([df_nuc['Y'],df_nuc['X']]).T
    cyto_coords = np.array([df_cyto['centroid-0'],df_cyto['centroid-1']]).T
    
    
    
    # Print non-injective mapping
    uniques,counts = np.unique(df_nuc['CytoID'],return_counts=True)
    bad_idx = np.where(counts > 1)[0]
    for i in bad_idx:
        print(f'CytoID being duplicated: {uniques[i]}')
    #% Relabel cyto seg with nuclear CellposeID
    
    
    df_cyto['CellposeID'] = np.nan
    for i,cyto in df_cyto.iterrows():
        cytoID = cyto['label']
        I = np.where(df_nuc['CytoID'] == cytoID)[0]
        # if len(I) > 1:
        #     print(f't = {t}: ERROR at CytoID {cytoID} = {I}')
        #     error()
        if len(I) == 1:
            df_cyto.at[i,'CellposeID'] = df_nuc.loc[I,'CellposeID']
    
    
    #----- map from cellpose to manual -----
    #NB: best to use the manual mapping since it guarantees one-to-one mapping from cellpose to manual cellIDs
    df_manual = pd.DataFrame(measure.regionprops_table(manual_tracks,intensity_image = nuc_seg,
                                                       properties = ['label','area'],
                                                       extra_properties = [most_likely_label]))
    df_manual = df_manual.rename(columns={'label':'basalID','most_likely_label':'CellposeID','area':'Cell volume (manual)'})
    assert(np.isnan(df_manual['CellposeID']).sum() == 0)
    
    for _,this_cell in df_manual.iterrows():
         df_nuc.loc[ df_nuc['CellposeID'] == this_cell['CellposeID'],'basalID'] = this_cell['basalID']
         df_nuc.loc[df_nuc['CellposeID'] == this_cell['CellposeID'],'Cell volume (manual)'] = this_cell['Cell volume (manual)']
         
    
    df_merge = df_nuc.merge(df_cyto,on='CellposeID')
    df_merge['NC ratio'] = df_merge['Nuclear volume'] / df_merge['Cell volume (auto)']
    
    df_merge['Frame'] = t
    df.append(df_merge)
    
df = pd.concat(df,ignore_index=True)




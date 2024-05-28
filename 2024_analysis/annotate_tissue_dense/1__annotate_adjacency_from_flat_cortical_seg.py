#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:10:00 2022

@author: xies
"""

import numpy as np
from skimage import io, measure, util, morphology
from glob import glob
from os import path
from scipy import ndimage as ndi
from scipy.spatial import distance
import pandas as pd
import matplotlib.pylab as plt
from tqdm import tqdm
import networkx as nx

from imageUtils import draw_labels_on_image, draw_adjmat_on_image, draw_adjmat_on_image_3d, most_likely_label

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
XX = 460
T = 15
touching_threshold = 2 #px
corona = 2

selem = ndi.generate_binary_structure(2,1)
# selem = ndi.iterate_structure(selem, corona)

# KEEP THIS!!
def most_likely_label(labeled,im):
    label = 0
    if len(im[im>0]) > 0:
        unique,counts = np.unique(im[im > 0],return_counts=True)
        label = unique[counts.argmax()]
    return label

#%% Load the flat cytoplasmic segmentations

DEMO = False

for t in tqdm(range(15)):
    
    if DEMO:
        cyto_seg = io.imread('/Users/xies/Desktop/Code/mouse_skin_size_control/2024_analysis/test_dataset/example_contact_map.tif')[t,...]
        dense_nuc_seg = io.imread('/Users/xies/Desktop/Code/mouse_skin_size_control/2024_analysis/test_dataset/example_mouse_skin_image.tif')[3,t,...]
    else:
        cyto_seg = io.imread(path.join(dirname,f'Image flattening/flat_cyto_seg_manual/t{t}.tif'))
        dense_nuc_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    
    #% Label transfer from nuc3D -> cyto2D
    
    # For now detect the max overlap label with the nuc projection
    df_nuc = pd.DataFrame( measure.regionprops_table(dense_nuc_seg.max(axis=0), intensity_image = cyto_seg
                                                   ,properties=['label','centroid','max_intensity',
                                                                'euler_number','area']
                                                   ,extra_properties = [most_likely_label]))
    df_nuc = df_nuc.rename(columns={'centroid-0':'Y','centroid-1':'X'
                                      ,'most_likely_label':'CytoID','label':'CellposeID'})
    
    
    df_cyto = pd.DataFrame( measure.regionprops_table(cyto_seg, properties=['centroid','label']))
    df_cyto.index = df_cyto['label']
    
    nuc_coords = np.array([df_nuc['Y'],df_nuc['X']]).T
    cyto_coords = np.array([df_cyto['centroid-0'],df_cyto['centroid-1']]).T
    
    
    # Print non-injective mapping
    uniques,counts = np.unique(df_nuc['CytoID'],return_counts=True)
    bad_idx = np.where(counts > 1)[0]
    for i in bad_idx:
        if not DEMO:
            print(f'CytoID being duplicated: {uniques[i]}')
    #% Relabel cyto seg with nuclear CellposeID
    
    df_cyto['CellposeID'] = np.nan
    for i,cyto in df_cyto.iterrows():
        cytoID = cyto['label']
        I = np.where(df_nuc['CytoID'] == cytoID)[0]
        if len(I) > 1:
            
            if not DEMO:
                print(f't = {t}: ERROR at CytoID {cytoID} = {I}')
            # error()
        elif len(I) == 1:
            df_cyto.at[i,'CellposeID'] = df_nuc.loc[I,'CellposeID']
    
    #% Reconstruct adj network from cytolabels that touch
    A = np.zeros((len(df_nuc),len(df_nuc)))
    adj_dict = {}
    for i,cyto in df_cyto.iterrows():
        
        if np.isnan(cyto['CellposeID']):
            continue
        
        this_idx = np.where(df_nuc['CellposeID'] == cyto['CellposeID'])[0]
        
        this_mask = cyto_seg == cyto['label']
        this_mask_dil = morphology.binary_dilation(this_mask,selem)
        
        touchingIDs,counts = np.unique(cyto_seg[this_mask_dil],return_counts=True)
        touchingIDs[counts > touching_threshold] # should get rid of 'conrner touching'
        # if i == 87:
        #     error
        touchingIDs = touchingIDs[touchingIDs > 2] # Could touch background pxs
        touchingIDs = touchingIDs[touchingIDs != cyto['label']] # nonself
        
        # Convert CytoID to CellposeID
        touching_cellposeIDs = np.array([df_cyto.loc[tID]['CellposeID'] for tID in touchingIDs])
        touching_cellposeIDs = touching_cellposeIDs[~np.isnan(touching_cellposeIDs)].astype(int)
        
        # Convert CellposeID to idx in df_nuc
        touching_idx = np.where(np.in1d(df_nuc['CellposeID'], touching_cellposeIDs))[0]
        
        A[this_idx,touching_idx] = 1
        
        adj_dict[cyto['CellposeID']] = touching_cellposeIDs
        
    
    #% Save as matrix and image
    im_adj = draw_adjmat_on_image(A,nuc_coords,[XX,XX]).astype(np.uint16)
    # io.imsave(path.join(dirname,f'Image flattening/flat_adj/t{t}.tif'),im_adj,check_contrast=False)
    
    # save matrix
    # np.save(path.join(dirname,f'Image flattening/flat_adj/adjmat_t{t}.npy'),A)
    
    # np.save(path.join(dirname,f'Image flattening/flat_adj_dict/adjdict_t{t}.npy'),adj_dict)
    

#%% Visualize adjacencies on the image itself (either in flat or in 3D)

A = np.load(path.join(dirname,f'Image flattening/flat_adj/adjmat_t{t}.npy'))

dense_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
df_nuc = pd.DataFrame( measure.regionprops_table(dense_seg,properties=['label','centroid']))
df_nuc = df_nuc.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X'})

nuc_coords_3d = np.array([df_nuc['Z'],df_nuc['Y'],df_nuc['X']]).T
im_adj = draw_adjmat_on_image_3d(A,nuc_coords_3d,[72,XX,XX])
selem = morphology.disk(3)
for z,im in enumerate(im_adj):
    im_adj[z,...] = morphology.dilation(im, selem)
# io.imsave('/Users/xies/Desktop/blah.tif',im_adj.astype(np.uint16))

# Construct triangulation
def adjmat2triangle(G):
    triangles = set()
    for u,w in G.edges:
        for v in set(G.neighbors(u)).intersection(G.neighbors(w)):
            triangles.add(frozenset([u,v,w]))
    return triangles

from networkx import Graph

G = nx.Graph(A)
triangles = adjmat2triangle(G)
triangle_neighbors = np.array([list(x) for x in list(triangles)])

from matplotlib.tri import Triangulation
tri = Triangulation(x=nuc_coords_3d[:,1], y=nuc_coords_3d[:,2], triangles = triangle_neighbors)

#%%

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.plot_trisurf(tri, Z = nuc_coords_3d[:,0])





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:46:23 2024

@author: xies
"""

import numpy as np
from skimage import io, measure, draw, util, morphology
import pandas as pd

from basicUtils import euclidean_distance
from mathUtils import surface_area

import matplotlib.pylab as plt
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, most_likely_label, colorize_segmentation

from tqdm import tqdm
from os import path

dx = 0.25
XX = 460
Z_SHIFT = 10

# Differentiating thresholds
centroid_height_cutoff = 3.5 #microns above BM

VISUALIZE = True
DEMO = True
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'


# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
dirname = '/Users/xies/Desktop/Code/mouse_skin_size_control/2024_analysis/test_dataset/'
im_demo = io.imread(path.join(dirname,'example_mouse_skin_image.tif'))
contact_map_demo = io.imread(path.join(dirname,'example_mouse_skin_cell_contact_map.tif'))


# FUCCI threshold (in stds)
alpha_threshold = 1
#NB: idx - the order in array in dense segmentation

#%%

DEBUG = True
SAVE= True

df = []

for t in range(15):
    
    if DEMO:
        nuc_seg = im_demo[t,:,3,:,:]
        cyto_seg = im_demo[t,:,4,:,:]
        manual_tracks = im_demo[t,:,5,:,:]
        
    else:
        nuc_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
        cyto_seg = io.imread(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual/t{t}_cleaned.tif'))
        manual_tracks = io.imread(path.join(dirname,f'manual_basal_tracking/sequence/t{t}.tif'))
    
    #% Label transfer from nuc3D -> cyto3D
    
    # For now detect the max overlap from cyto3d label with the nuc3d label
    df_nuc = pd.DataFrame( measure.regionprops_table(nuc_seg, intensity_image = cyto_seg
                                                   ,properties=['label','centroid','max_intensity',
                                                                'euler_number','area']
                                                   ,extra_properties = [most_likely_label,surface_area]))
    
    df_nuc = df_nuc.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X','area':'Nuclear volume'
                                    ,'most_likely_label':'CytoID','label':'CellposeID'})
    
    df_cyto = pd.DataFrame( measure.regionprops_table(cyto_seg
                                                      , properties=['centroid','label','area']))
    df_cyto = df_cyto.rename(columns={'area':'Cell volume','label':'CytoID'})
    df_cyto.index = df_cyto['CytoID']
    
    
    # Print non-injective mapping
    uniques,counts = np.unique(df_nuc['CytoID'],return_counts=True)
    bad_idx = np.where(counts > 1)[0]
    for i in bad_idx:
        print(f't = {t}; CytoID being duplicated: {uniques[i]}')
    #% Relabel cyto seg with nuclear CellposeID
    
    df_cyto['CellposeID'] = np.nan
    for i,cyto in df_cyto.iterrows():
        cytoID = cyto['CytoID']
        I = np.where(df_nuc['CytoID'] == cytoID)[0]
        if not DEMO and len(I) > 1:
            print(f't = {t}: ERROR at CytoID {cytoID} = {df_nuc.loc[I]}')
            error()
        if len(I) == 1:
            df_cyto.at[i,'CellposeID'] = df_nuc.loc[I,'CellposeID']

    if not DEMO and DEBUG:
    #----- map from cellpose to manual -----
        #NB: best to use the manual mapping since it guarantees one-to-one mapping from cellpose to manual cellIDs
        df_manual = pd.DataFrame(measure.regionprops_table(manual_tracks,intensity_image = nuc_seg,
                                                           properties = ['label','area'],
                                                           extra_properties = [most_likely_label]))
        df_manual = df_manual.rename(columns={'label':'basalID','most_likely_label':'CellposeID','area':'Cell volume (manual)'})
        assert(np.isnan(df_manual['CellposeID']).sum() == 0)
        
        # Find the corresponding manually segmented ID/volume
        for _,this_cell in df_manual.iterrows():
             df_nuc.loc[ df_nuc['CellposeID'] == this_cell['CellposeID'],'basalID'] = this_cell['basalID']
             df_nuc.loc[df_nuc['CellposeID'] == this_cell['CellposeID'],'Cell volume (manual)'] = this_cell['Cell volume (manual)']
             # Find if all the adjacent cells are cyto annotated
             # Loading dict of CellposeID: neighbor cellposeIDs
             adj_dict = np.load(path.join(dirname,f'Image flattening/flat_adj_dict/adjdict_t{t}.npy'),allow_pickle=True).item()
             for neighbor in adj_dict[this_cell['CellposeID']]:
                 neighbor_cytoID = df_nuc[df_nuc['CellposeID'] == neighbor]['CytoID'].values
                 if np.any(neighbor_cytoID == 0) or np.any(np.isnan(neighbor_cytoID)):
                     print(f'Missing cytoID at: t = {t} Nuc CellposeID = {neighbor}')
    
    df_merge = df_nuc.merge(df_cyto.drop(columns=['centroid-0','centroid-1','centroid-2','CytoID']),on='CellposeID',how='left')
    # df_merge = df_merge.rename(columns={'CytoID_x':'CytoID'})
    # df_merge['NC ratio'] = df_merge['Nuclear volume'] / df_merge['Cell volume']
    
    df_merge['Frame'] = t
    df.append(df_merge)
    
df = pd.concat(df,ignore_index=True)

if SAVE:
    df.to_csv(path.join(dirname,'cyto_dataframe.csv'))
    print(f'Saved to: {dirname}')


#%% Find missing neighboring cyto segs
# with open(path.join(dirname,'basal_no_daughters.pkl'),'rb') as f:
#     collated = pkl.load(f)

# for basalID,cell in collated.items():
#     for i,row in cell.iterrows():
        


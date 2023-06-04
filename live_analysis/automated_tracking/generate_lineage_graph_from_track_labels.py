#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 14:33:21 2023

@author: xies
"""

import networkx as nx
from skimage import io, measure
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

cell_tracks = io.imread(path.join(dirname,'manual_basal_tracking/basal_tracks_nuclei.tif'))
daughter_tracks = io.imread(path.join(dirname,'manual_basal_tracking_daughters/basal_track_daughter_nuclei.tif'))

#%% Parse cell + daughter images into a table (w/generations notated), ignore mothers for now

cells = []; daughters = []; mothers = []
for t in tqdm(range(15)):
    
    df = pd.DataFrame(measure.regionprops_table(cell_tracks[t,...],properties=['centroid','label']))
    df['Frame'] = t
    cells.append( df )
    df = pd.DataFrame(measure.regionprops_table(daughter_tracks[t,...],properties=['centroid','label']))
    df['Frame'] = t
    daughters.append( df )


cells = pd.concat(cells,ignore_index=True)
cells['Generation'] = 1
daughters = pd.concat(daughters,ignore_index=True)
daughters['Generation'] = 2

# Wrangle the field names
cells = cells.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X','label':'CellID'})
daughters = daughters.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X','label':'CellID'})

df = pd.concat((cells,daughters),ignore_index=True)
# Construct a unique spotID for each cellID/frame pair
df['SpotID'] = df['CellID'].astype('string') + '_' + df['Frame'].astype('string')

#%% 

cellIDs = df['CellID'].unique() 

# 1) Isolate each single cells and construct a three generation branch
trees = []
for ID in tqdm(cellIDs):
    
    this_cell = df[df['CellID'] == ID]
    
    tree = nx.DiGraph()
    c = this_cell[this_cell['Generation'] == 1].sort_values('Frame')
    d1 = this_cell[this_cell['Generation'] == 2].iloc[0]
    d2 = this_cell[this_cell['Generation'] == 2].iloc[1]
    for idx,row in this_cell.iterrows():
        tree.add_node(row['SpotID'])
    
    # Add the main cell
    successive_frame_pairs = np.stack((c.index[:-1],c.index[1:])).T
    for pair in successive_frame_pairs:
        tree.add_edge(c.loc[pair[0]]['SpotID'],c.loc[pair[1]]['SpotID'])
    
    last_frame = c.iloc[-1]['SpotID']
    # Add the two daughters
    tree.add_edge(last_frame,d1['SpotID'])
    tree.add_edge(last_frame,d2['SpotID'])
    trees.append(tree)

nx.draw_networkx(tree)

#%%
        
        
        
    

# 2) Go through each Division time point and see if a mother cell in the set hasthe same coordinate
#   If so, update generation of that mother cell by +2 and add branch to dividing cell
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
import matplotlib.pyplot as plt
import pickle as pkl

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

cell_tracks = io.imread(path.join(dirname,'manual_basal_tracking/basal_tracks_nuclei.tif'))
seg_notrack = io.imread(path.join(dirname,'3d_nuc_seg/manual_seg_no_track.tif'))

TT = 15

def euclidean_distance(d1,d2):
    D = np.sqrt((d1['X'] -d2['X'])**2 + (d1['Y'] - d2['Y'])**2 + (d1['Z'] - d2['Z'])**2)
    return D
    

#%% Parse cell images into a table

SPOTID = 0

cells = [];
for t in tqdm(range(TT)):
    
    df = pd.DataFrame(measure.regionprops_table(cell_tracks[t,...],properties=['centroid','label']))
    df['Frame'] = t
    cells.append( df )

cells = pd.concat(cells,ignore_index=True)
cells['Generation'] = 1

# Wrangle the field names
df = cells.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X','label':'basalID'})

# Construct a unique spotID for each cellID/frame pair
df['SpotID'] = df['basalID'].astype('string') + '_' + df['Frame'].astype('string')

# Each cell is now within its own lineage
df['LineageID'] = df['basalID']

# Prase mother and daughter annotations and join into table

SPOT_ID = 100000
lineage_annotations = pd.read_excel(path.join(dirname,'manual_basal_tracking_lineage/lineage_annotations.xlsx'),index_col=0)

for ID in tqdm(df['basalID'].unique()):
    this_anno = lineage_annotations.loc[ID]
    # Mother
    mother_frame = int(this_anno['Mother frame'])
    mother_mask = (seg_notrack[mother_frame,...] == this_anno['Mother cellposeID']).astype(int)
    mother_ = pd.DataFrame(measure.regionprops_table(mother_mask,properties=['centroid']))
    mother_['basalID'] = ID
    mother_['LineageID'] = ID
    mother_['Frame'] = mother_frame
    mother_['Generation'] = 0
    mother_['SpotID'] = str(SPOT_ID)
    SPOT_ID += 1
    
    # Daughters
    daughter_frame = int(this_anno['Daughter frame'])
    daughter1_mask = (seg_notrack[daughter_frame,...] == this_anno['Daughter1']).astype(int)
    daughter1_ = pd.DataFrame(measure.regionprops_table(daughter1_mask,properties=['centroid']))
    daughter1_['SpotID'] = str(SPOT_ID)
    SPOT_ID += 1
    
    daughter2_mask = (seg_notrack[daughter_frame,...] == this_anno['Daughter2']).astype(int)
    daughter2_ = pd.DataFrame(measure.regionprops_table(daughter2_mask,properties=['centroid']))
    daughter2_['SpotID'] = str(SPOT_ID)
    SPOT_ID += 1
    
    daughters_ = pd.concat((daughter1_,daughter2_),ignore_index=True)
    daughters_['basalID'] = ID
    daughters_['LineageID'] = ID
    daughters_['Generation'] = 2
    daughters_['Frame'] = daughter_frame
    combined_ = pd.concat((mother_,daughters_),ignore_index=True)
    
    combined_ = combined_.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X'})
    df = pd.concat((df,combined_),ignore_index=True)
    
df['Dropped'] = False

#%% Construct graph 1) Isolate each single cells and construct a generation branch

cellIDs = df['basalID'].unique()

trees = {}
for ID in tqdm(cellIDs):
    
    this_cell = df[df['basalID'] == ID]
    
    # Find all 
    tree = nx.DiGraph()
    c = this_cell[this_cell['Generation'] == 1].sort_values('Frame')
    m = this_cell[this_cell['Generation'] == 0].iloc[0]
    d1 = this_cell[this_cell['Generation'] == 2].iloc[0]
    d2 = this_cell[this_cell['Generation'] == 2].iloc[1]
    
    for idx,row in this_cell.iterrows():
        tree.add_node(row['SpotID'])
    
    
    # Link the main cell's track
    successive_frame_pairs = np.stack((c.index[:-1],c.index[1:])).T
    for pair in successive_frame_pairs:
        tree.add_edge(c.loc[pair[0]]['SpotID'],c.loc[pair[1]]['SpotID'])
    
    first_frame = c.iloc[0]['SpotID']
    tree.add_edge(m['SpotID'],first_frame)
    
    last_frame = c.iloc[-1]['SpotID']
    # Add the two daughters
    tree.add_edge(last_frame,d1['SpotID'])
    tree.add_edge(last_frame,d2['SpotID'])
    trees[ID] = tree

# nx.draw_networkx(tree)

# for DEBUG
naive_df = df.copy()
naive_trees = trees.copy()

#%% Link sisters that share the same mother coordinate

from scipy.spatial import distance_matrix
df = naive_df.copy()
trees = naive_trees.copy()

for t in range(TT):
    mothers_in_frame = df[(df['Generation'] == 0) & (df['Frame'] == t)]
    if len(mothers_in_frame)>1:
        coords = mothers_in_frame[['Z','Y','X']].values
        D = distance_matrix(coords,coords)
        D[np.tril_indices(len(mothers_in_frame))] = np.nan
        # D[np.eye(len(mothers_in_frame)) == 1] = np.nan
        I,J = np.where(D < 1)
        sister_pairs = zip(I,J)
        
        for sister in sister_pairs:
            
            sisterA_basalID = mothers_in_frame.iloc[sister[0]]['basalID']
            sisterB_basalID = mothers_in_frame.iloc[sister[1]]['basalID']
            
            sisterA = df[df['basalID'] == sisterA_basalID]
            sisterB = df[df['basalID'] == sisterB_basalID]
            
            # Edit the 'lineageID'
            new_linID = min(sisterA_basalID,sisterB_basalID)
            df.loc[df['basalID'] == sisterA_basalID,'LineageID'] = new_linID
            df.loc[df['basalID'] == sisterB_basalID,'LineageID'] = new_linID
            
            # Edit sisterB's mother to be sisterA's mother as well
            # In table:
            unified_mother = sisterA[sisterA['Generation'] == 0]
            mother2update = sisterB[sisterB['Generation'] == 0]
            df.loc[mother2update.index,'SpotID'] = unified_mother['SpotID']
            
            # In the trees:
            treeB = trees[sisterB_basalID]
            treeB = nx.relabel_nodes(treeB,{mother2update['SpotID'].iloc[0]:
                                            unified_mother['SpotID'].iloc[0]})
            trees[sisterB_basalID] = treeB
    

#% Link across generations

cellIDs = df['basalID'].unique()
for ID in tqdm(cellIDs):

    
    this_cell = df[df['basalID'] == ID]
    c = this_cell[this_cell['Generation'] % 2 == 1].sort_values('Frame')
    # m = this_cell[this_cell['Generation'] == 0].iloc[0]
    
    
    last_frame = c.iloc[-1]
    
    # Only look at mothers within this frame
    daughters_born_next_frame = df[(df['Generation'] == 0) & (df['Frame'] == last_frame.Frame)]
    if len(daughters_born_next_frame) > 0: 
        
        D = euclidean_distance(daughters_born_next_frame,last_frame)
        # Last division cells is among the mother list
        candidate_daughters = np.where(D < 0.1)[0]
        
        for candidate in candidate_daughters:
            
            this_tree = trees[ID]
            
            # Daughter found
            daughter_basalID = daughters_born_next_frame.iloc[candidate]['basalID']
            
            daughter_cell = df[df['basalID'] == daughter_basalID]
            daughter_tree = trees[daughter_basalID]
            
            
            # 1) Edit the candidate cell's entries
            # Retrieve daughter cell's generation entry and +; amend the LineageID
            df.loc[df['basalID'] == daughter_basalID,'Generation'] = df.loc[df['basalID'] == daughter_basalID,'Generation']+ 2
            df.loc[df['basalID'] == daughter_basalID,'LineageID'] = this_cell.iloc[0]['LineageID']

            # Edit the tree annotation in daughter so that its mother same SpotID as the current dividing cell
            old_mother_spotID = daughters_born_next_frame.iloc[D.argmin()]['SpotID']
            daughter_tree = nx.relabel_nodes(daughter_tree,{old_mother_spotID:last_frame['SpotID']})
            # Update draughter tree in list (not automatically updated from dict)
            trees[daughter_basalID] = daughter_tree
            
            # 2) Edit the current cell annotation so that the daughter annotation corresponds to the actual tracked basalID
            # Which daughter is the correct one?
            d_ = this_cell[this_cell['Generation'] == 2]
            real_daughter_first_frame = daughter_cell[daughter_cell['Generation']==1].sort_values('Frame').iloc[0]
            daughter2edit = d_.iloc[np.where(euclidean_distance(real_daughter_first_frame,d_) == 0)]['SpotID']
            
            # Flag this daughter for drop from dataframe
            df.loc[daughter2edit.index,'Dropped'] = True
            # Edit the current tree annotation so the right daughter has the SpotID of the real daughter first frame
            this_tree = nx.relabel_nodes(this_tree,{daughter2edit.iloc[0]:real_daughter_first_frame['SpotID']})
            trees[ID] = this_tree
            
#%% Merge trees with the same lineageID

basalIDs2merge = set(df['basalID']) - set(df['LineageID'])

# find corresponding lineageIDs
lineageIDs2merge = set([df[df['basalID'] == ID].iloc[0]['LineageID'] for ID in basalIDs2merge])

for linID in lineageIDs2merge:

    # See what other basalIDs are contained within lineageID
    otherIDs = df[df['LineageID'] == linID]['basalID'].unique()
    combined_tree = trees[otherIDs[0]]
    for otherID in otherIDs:
        other_tree = trees[otherID]
        combined_tree = nx.compose(combined_tree,other_tree)
    
    ID2keep = otherIDs.min()
    IDs2delete = otherIDs[otherIDs != ID2keep]
    # Update the tree as the smallest basalID
    trees[ID2keep] = combined_tree
    for others in IDs2delete:
        del trees[others]

## Save as pkl intermediate
with open(path.join(dirname,'manual_basal_tracking_lineage/lineage_trees.pkl'),'wb') as f:
    pkl.dump((trees,df),f)

#%%
        
            




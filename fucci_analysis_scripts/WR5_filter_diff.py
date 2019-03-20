#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:24:39 2019

@author: xies
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from skimage import io
import os

#filename = '/data/Skin/W-R5/data.csv'
filename = '/Users/mimi/Box Sync/Mouse/Skin/W-R5/data.csv'
columns = ['Timeframe','CloneID','ParentID','CellID','PositionX',
           'PositionY','VoronoiArea','G1MarkerInVoronoiArea',
           'ActinSegmentationArea','G1MarkerInActinSegmentationArea']
df = pd.read_csv(filename,names = columns)

# 1. Find cells that have a 'parent'... i.e. cells that are guarenteed to have divided
has_parent = df['ParentID'] != 0

# Filter for cells that also have 'children' i.e cells were born in the movie
parentIDu = np.unique(df['ParentID'])
I = np.array(ismember(df['CellID'],parentIDu))
has_children = I > 0

# Also find cells that have no children (i.e. cells that leave the epithelium)
childless = I == 0
childlessID = np.unique(df[childless]['CellID'])

# Cells now are born and divide within movie
filtered = df[has_parent & has_children]

cellIDu = np.unique(filtered['CellID'])
# Collate each cell into its own sub-frame
collated = []
Ncells = len(cellIDu)
for c in cellIDu:
    this_cell = filtered[filtered['CellID'] == c]
    collated.append(this_cell)

# Collate each childless into its own sub-frame
# Make sure they're around for more than 1 frame
childless = []
for c in childlessID:
    this_cell = df[df['CellID'] == c]
    if len(this_cell) > 1:
        childless.append(this_cell)
Nchildless = len(childless)

# Reconstruct a 'cell-centric' dataframe with non-time-series data
columns = ['CellID','ParentID','Bframe','Divframe',
           'Bx','By','Dx','Dy','Barea','Divarea','G1maxframe']
X = np.zeros((Ncells,11))
for (i,c) in enumerate(collated):
    X[i,0] = c['CellID'].iloc[0]
    X[i,1] = c['ParentID'].iloc[0]
    X[i,2] = c['Timeframe'].iloc[0]
    X[i,3] = c['Timeframe'].iloc[-1]
    X[i,4] = c['PositionX'].iloc[0]
    X[i,5] = c['PositionY'].iloc[0]
    X[i,6] = c['PositionX'].iloc[-1]
    X[i,7] = c['PositionY'].iloc[-1]
    X[i,8] = c['ActinSegmentationArea'].iloc[0]
    X[i,9] = c['ActinSegmentationArea'].iloc[-1]
    X[i,10] = X[i,2] + np.array(c['G1MarkerInActinSegmentationArea']).argmax()
celldf = pd.DataFrame(data=X,columns=columns)
    
# Same for childless cells
columns = ['CellID','ParentID','Bframe','Diffframe',
           'Bx','By','Diffx','Diffy','Barea','Diffarea','G1maxframe']
X = np.zeros((Nchildless,11))
for (i,c) in enumerate(childless):
    X[i,0] = c['CellID'].iloc[0]
    X[i,1] = c['ParentID'].iloc[0]
    X[i,2] = c['Timeframe'].iloc[0]
    X[i,3] = c['Timeframe'].iloc[-1]
    X[i,4] = c['PositionX'].iloc[0]
    X[i,5] = c['PositionY'].iloc[0]
    X[i,6] = c['PositionX'].iloc[-1]
    X[i,7] = c['PositionY'].iloc[-1]
    X[i,8] = c['ActinSegmentationArea'].iloc[0]
    X[i,9] = c['ActinSegmentationArea'].iloc[-1]
    X[i,10] = X[i,2] + np.array(c['G1MarkerInActinSegmentationArea']).argmax()
childlessdf = pd.DataFrame(data=X,columns=columns)

# Go through dividing cells and see if there are differentiating cells within a window
has_neighboring_diff = np.zeros(Ncells)
windowS = 40
windowT = 2
for i in range(Ncells):
    c = celldf.iloc[i]
    t0 = c['Bframe']
    x0 = c['Bx']
    y0 = c['By']
    
    neighbor_diff = find_dividing_cells_within_neighborhood(childlessdf,[t0,x0,y0],
                                                            35,2)
    has_neighboring_diff[i] = len(neighbor_diff) > 0
celldf['HasNeighborDiff'] = has_neighboring_diff > 0

# Further filter collated cell
filtered = []
for (i,c) in enumerate(collated):
    if has_neighboring_diff[i] > 0:
        filtered.append(c)

# Export
for f in filtered:
    f['Region'] = 5
f5 = filtered

####################################

# Put everything into a padded array for heatmap visualization
max_time = max( [len(c) for c in filtered] )

A = np.zeros((Ncells,max_time))
G1 = np.zeros((Ncells,max_time))

for (i,c) in enumerate(filtered):
    a = c['ActinSegmentationArea']
    A[i,0:len(a)] = a    
    g1 = c['G1MarkerInVoronoiArea']
    G1[i,0:len(a)] = g1

plt.subplot(1,2,1); plt.pcolor(A)
plt.subplot(1,2,2); plt.pcolor(G1),plt.colorbar()

# Plot birth size and T cell cycle
Tcycle = [len(c)* 0.5 for c in filtered]
Bsize = [c['ActinSegmentationArea'].iloc[0] for c in filtered]
plt.figure()
sb.regplot(np.array(Bsize),np.array(Tcycle))


############################################################################

def find_dividing_cells_within_neighborhood(childlessdf,center,windowS,windowT):
    """
    Find cells that have differentiated windowT frames before 'center' and
    within windowS-distance of 'center'
    
    Input: cell-centric dataframe of differentiating cells, coordinate of
    'center cell' (c = [t0,x0,y0]), windows in space & time
    
    """
    
    t0 = center[0]
    x0 = center[1]
    y0 = center[2]
    
    # Temporal filter
    T = childlessdf['Diffframe'] >= (t0 - windowT)
    T = T & (childlessdf['Diffframe'] <= t0)
    
    # Spatial filter
    D = np.sqrt( (childlessdf['Diffx'] - x0)**2 +
                (childlessdf['Diffy'] - y0)**2)
    S = D < windowS
    
    cellIDs = np.unique(childlessdf[T&S]['CellID'])
    
    return cellIDs


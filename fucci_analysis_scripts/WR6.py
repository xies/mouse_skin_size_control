#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 20:02:13 2019

@author: xies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from skimage import io
import os

#filename = '/data/Skin/W-R6/data.csv'
filename = '/Users/mimi/Box Sync/Mouse/Skin/W-R6/data.csv'
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

# Cells now are born and divide within movie
filtered = df[has_parent & has_children]


cellIDu = np.unique(filtered['CellID'])
# Collate each cell into its own sub-frame
collated = []
Ncells = len(cellIDu)
for c in cellIDu:
    this_cell = filtered[filtered['CellID'] == c]
    collated.append(this_cell)

# Now all cells are aligned by birth
for c in collated:
    t = np.array(c['Timeframe'])
    t = (t - t[0]) * .5
#    plt.plot(t,c['ActinSegmentationArea'])
    plt.plot(t,c['G1MarkerInVoronoiArea'])

# Put everything into a padded array for heatmap visualization
max_time = max( [len(c) for c in collated] )

A = np.zeros((Ncells,max_time))
G1 = np.zeros((Ncells,max_time))

for (i,c) in enumerate(collated):
    a = c['ActinSegmentationArea']
    A[i,0:len(a)] = a/a.max()
    g1 = c['G1MarkerInVoronoiArea']
    G1[i,0:len(a)] = g1/g1.max()

plt.subplot(1,2,1); plt.pcolor(A)
plt.subplot(1,2,2); plt.pcolor(G1),plt.colorbar()

# Plot birth size and T cell cycle
Tcycle = [len(c)* 0.5 for c in collated]
Bsize = [c['ActinSegmentationArea'].tolist()[0] for c in collated]
plt.scatter(Bsize,Tcycle)

# Export
for c in collated:
    c['Region'] = 6
c6 = collated

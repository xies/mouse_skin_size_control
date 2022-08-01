#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:48:28 2021

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io,draw

import pickle as pkl
import csv
from os import path

dx = 0.25
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/cropped/'

#%% Load ROIs

# Load collated list of time series
with open(path.join(dirname,'collated_manual.pkl'),'rb') as f:
    c2 = pkl.load(f)
collated = c2


# Load px, py, framne, zpos as exported by export_ROIs.py
px = []
with open(path.join(dirname,'ROIs/polygon_x.csv')) as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for x in reader:
        x = np.array([int(a) for a in x])
        px.append(x)
        
py = []
with open(path.join(dirname,'ROIs/polygon_y.csv')) as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for y in reader:
        y = np.array([int(a) for a in y])
        py.append(y)

frame = np.genfromtxt(path.join(dirname,'ROIs/frame.csv'), delimiter=',',dtype=np.int)
zpos = np.genfromtxt(path.join(dirname,'ROIs/zpos.csv'), delimiter=',',dtype=np.int) - 1
cellIDs = np.genfromtxt(path.join(dirname,'ROIs/cellIDs.csv'), delimiter=',',dtype=np.int)


#%% Load thresholded nuclear mask

nuc_mask = io.imread(path.join(dirname,'h2b_mask.tif'))

nuc_ts = []
for j,c in enumerate(collated):
    c['Nuclear volume'] = np.nan
    Inondaughters = c.Daughter == 'None'
    c_ = c[Inondaughters]
    cell2lookfor = c_.iloc[0]['CellID']
    Nframes = len(c_)
    
    nuc_volume = []
    for t in c_.Frame.values:
        
        I = cellIDs == cell2lookfor
        I = I & (frame == t)
        
        idx = np.where(I)[0]
        
        this_nuc_volume = []
        for i in idx:
            z = zpos[i]
            this_im = nuc_mask[t,z,:,:]
            RR,CC = draw.polygon(py[i],px[i]) # remember coord rotation
            this_nuc = this_im[RR,CC]
            this_nuc_volume.append(this_nuc.sum())
        
        nuc_volume.append(np.array(this_nuc_volume).sum() * dx ** 2)
        
    c.at[Inondaughters,'Nuclear volume'] = nuc_volume
    collated[j] = c
    
#%% Save collated data

f = open(path.join(dirname,'tracked_cells/collated_manual_nuc.pkl'),'w')
pkl.dump(collated,f)
f.close()

#%% Cell-centric meta growth data
df = pd.read_pickle(path.join(dirname,'tracked_cells/dataframe.pkl'))

for c in collated:
    this_entry_idx = df[df.CellID == c.iloc[0].CellID].index
    if len(this_entry_idx) > 0:
        
        g1_exit_idx = np.where(c.Phase == 'G1')[0][-1]
        div_idx = np.where(c.Daughter == 'None')[0][-1]
        
        df.at[this_entry_idx,'Birth nuclear volume'] = c.iloc[0]['Nuclear volume']
        df.at[this_entry_idx,'G1 nuclear volume'] = c.iloc[g1_exit_idx]['Nuclear volume']
        df.at[this_entry_idx,'Division nuclear volume'] = c.iloc[div_idx]['Nuclear volume']
        
        
df['G1 nuclear growth'] = df['G1 nuclear volume'] - df['Birth nuclear volume']
df['Division nuclear growth'] = df['Division nuclear volume'] - df['Birth nuclear volume']
    
df.to_pickle(path.join(dirname,'tracked_cells/dataframe_nuc.pkl'))


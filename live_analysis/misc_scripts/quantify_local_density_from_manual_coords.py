#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:33:16 2023

@author: xies
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pylab as plt
from glob import glob
from natsort import natsorted

from re import search
from twophotonUtils import parse_XML_timestamps, parse_voxel_resolution_from_XML

from os import path

dirnames = {}
# dirnames['Ablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'
dirnames['Ablation_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R1'
dirnames['Ablation_R5'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R1'
# dirnames['Ablation_R6'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R2'

def counter_neighbor_cells_within_threshold(center_coords,timestamps,other_coords,th=10):
    
    Ncenters = len(center_coords)
    
    neighbor_counts = np.zeros((Ncenters,len(timestamps)))
    
    for i, t in enumerate(timestamps):
        
        # Generate euc dist matrix, ignore Z for now
        D = cdist(center_coords[['X','Y']]*dx, other_coords[other_coords['T'] == t][['X','Y']]*dx)
        
        # Go through each center point and find the num of neighbors within distance threshold
        neighbor_counts[:,i] = (D < th).sum(axis=1)
        
    return neighbor_counts



#%%

delta_ablation = {}
delta_nonablation = {}
timestamps = {}

for name,dirname in dirnames.items():
    dx,dz = parse_voxel_resolution_from_XML(dirname)
    dx = dx / 1.5
    
    sorted_files = natsorted(glob(path.join(dirname,'local_density/t*.csv')))
    t = list(parse_XML_timestamps(dirname, beginning= int(search('t([0-9]).csv',sorted_files[0]).group(1))).values())
    t = np.array([(x-t[0]).total_seconds()/3600 for x in t])
    assert(len(sorted_files) == len(t))
    timestamps[name] = t
    
    local_coords = []
    for i,f in enumerate(sorted_files):
        _df = pd.read_csv(f,index_col=0)
        #Overwrite the time axis
        _df['axis-0'] = t[i]
        
        local_coords.append(_df)
        
    local_coords = pd.concat(local_coords)
    local_coords = local_coords.rename(columns={'axis-0':'T','axis-1':'Z','axis-2':'Y','axis-3':'X'})
    
    ablation_coords = pd.read_csv(path.join(dirname,'manual_tracking/ablation_xyz.csv'),index_col = 0)
    ablation_coords = ablation_coords.rename(columns={'axis-0':'T','axis-1':'Z','axis-2':'Y','axis-3':'X'})
    
    nonablation_coords = pd.read_csv(path.join(dirname,'manual_tracking/nonablation_xyz.csv'),index_col = 0)
    nonablation_coords = nonablation_coords.rename(columns={'axis-0':'T','axis-1':'Z','axis-2':'Y','axis-3':'X'})

    
    ablation_neighbor_counts = counter_neighbor_cells_within_threshold(ablation_coords,t,local_coords)
    nonablation_neighbor_counts = counter_neighbor_cells_within_threshold(nonablation_coords,t,local_coords)
    
    
    delta_ablation[name] = ablation_neighbor_counts - ablation_neighbor_counts[:,0][:,None]
    delta_nonablation[name] = nonablation_neighbor_counts - nonablation_neighbor_counts[:,0][:,None]

#%%
for name in dirnames.keys():
    plt.plot(timestamps[name],delta_ablation[name].mean(axis=0),'b-')
    plt.plot(timestamps[name],delta_nonablation[name].mean(axis=0),'r-')
    
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:22:31 2021

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, util
from os import path
from tqdm import tqdm

import pickle as pkl

from measureSemiauto import measure_track_timeseries_from_segmentations,cell_cycle_annotate,collate_timeseries_into_cell_centric_table

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


dirnames = {}
# dirnames['WT_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'
# dirnames['WT_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R2'
# dirnames['WT_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R1'
# dirnames['WT_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R2'

# dirnames['RBKO_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
# dirnames['RBKO_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'
# dirnames['RBKO_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R1'
# dirnames['RBKO_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R2'

dirnames['RBKO_p107het_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/05-04-2023 RBKO p107het pair/F8 RBKO p107 het/R2'

dx = {}
dx['WT_R1'] = 0.206814922817744/1.5
dx['WT_R2'] = 0.206814922817744/1.5
dx['WT_R3'] = 0.165243202683616/1.5
dx['WT_R4'] = 0.165243202683616/1.5
dx['RBKO_R1'] = 0.206814922817744/1.5
dx['RBKO_R2'] = 0.206814922817744/1.5
dx['RBKO_R3'] = 0.165243202683616/1.5
dx['RBKO_R4'] = 0.165243202683616/1.5
dx['RBKO_p107het_R2'] = 0.165243202683616/1.5


mouse = {'WT_R1':'WT_M1','WT_R2':'WT_M1','RBKO_R1':'RBKO_M2','RBKO_R2':'RBKO_M2'
         ,'WT_R3':'WT_M3','WT_R4':'WT_M3','RBKO_R3':'RBKO_M4','RBKO_R4':'RBKO_M4'
         ,'RBKO_p107het_R2':'RBKO_p107het'}

pairs = {'WT_M1':'Pair 1','RBKO_M2':'Pair 1','WT_M3':'Pair 2','RBKO_M4':'Pair 2','RBKO_p107het':'Pair 3'}


RECALCULATE = True

#%% Load and collate manual track+segmentations
# Dictionary of manual segmentation (there should be no first or last time point)

for name,dirname in dirnames.items():
    
    for mode in ['curated']:

        print(f'---- Working on {name} {mode} ----')
        if name == 'WT_R4' and mode == 'manual':
            continue
        
        genotype = name.split('_')[0]
        
        # Construct pathnames
        pathdict = {}
        pathdict['Segmentation'] = path.join(dirname,f'manual_tracking/{mode}_clahe.tif')
        pathdict['H2B'] = path.join(dirname,'master_stack/G.tif')
        pathdict['FUCCI'] = path.join(dirname,'master_stack/R.tif')
        pathdict['Frame averages'] = path.join(dirname,'high_fucci_avg_size.csv')
        pathdict['Cell cycle annotations'] = path.join(dirname,f'{name}_cell_cycle_annotations.xlsx')
        
        # Construct metadata
        metadata = {}
        metadata['um_per_px'] = dx[name]
        metadata['Region'] = name
        metadata['Mouse'] = mouse[name]
        metadata['Pair'] = pairs[mouse[name]]
        metadata['Genotype'] = genotype
        metadata['Mode'] = mode
        metadata['Dirname'] = dirname
        
        
        #% Re-construct tracks with manually fixed tracking/segmentation
        # if RECALCULATE:
        tracks = measure_track_timeseries_from_segmentations(name,pathdict,metadata)
        tracks = cell_cycle_annotate(tracks,pathdict,metadata)
        
        # Save to the manual tracking folder
        with open(path.join(dirname,'manual_tracking',f'{name}_complete_cycles_fixed_{mode}.pkl'),'wb') as file:
            pkl.dump(tracks,file)
            
        # Construct the cell-centric metadata dataframe
        df,tracks = collate_timeseries_into_cell_centric_table(tracks,metadata)
        
        df.to_csv(path.join(dirname,f'manual_tracking/{name}_dataframe_{mode}.csv'))
        # Save to the manual tracking folder    
        with open(path.join(dirname,'manual_tracking',f'{name}_complete_cycles_fixed_{mode}.pkl'),'wb') as file:
            pkl.dump(tracks,file)




  
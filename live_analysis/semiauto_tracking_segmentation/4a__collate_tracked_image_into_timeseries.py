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
dirnames['Ablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'
dirnames['Nonablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'

dx = {}
dx['Ablation_R1'] = 0.14599609375*1.5
dx['Nonablation_R1'] = 0.14599609375*1.5

mouse = {'Ablation_R1':'WT_F1','Nonablation_R1':'WT_F1'}
subdir_str = {'Ablation_R1':'ablation','Nonablation_R1':'nonablation'}

pairs = {'WT_F1':np.nan,'WT_F1':np.nan}

RECALCULATE = True

timestamps = {'Ablation_R1':np.array([0,2,4,7,11,23,36])
              ,'Nonablation_R1':np.array([0,2,4,7,11,23,36])}

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
        pathdict['Segmentation'] = path.join(dirname,'manual_tracking',subdir_str[name],f'{mode}_clahe.tiff')
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
        metadata['Time stamps'] = timestamps[name]
        
        #% Re-construct tracks with manually fixed tracking/segmentation
        # if RECALCULATE:
        tracks = measure_track_timeseries_from_segmentations(name,pathdict,metadata)
        tracks = cell_cycle_annotate(tracks,pathdict,metadata)
        
        # Save to the manual tracking folder
        with open(path.join(dirname,'manual_tracking',f'{name}_dense_{mode}.pkl'),'wb') as file:
            pkl.dump(tracks,file)
            
        # Construct the cell-centric metadata dataframe
        df,tracks = collate_timeseries_into_cell_centric_table(tracks,metadata)
        
        df.to_csv(path.join(dirname,f'manual_tracking/{name}_dataframe_{mode}.csv'))
        # Save to the manual tracking folder    
        with open(path.join(dirname,'manual_tracking',f'{name}_dense_{mode}.pkl'),'wb') as file:
            pkl.dump(tracks,file)




  
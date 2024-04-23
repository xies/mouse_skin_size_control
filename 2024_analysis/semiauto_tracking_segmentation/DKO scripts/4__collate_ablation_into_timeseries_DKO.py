#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 13:21:43 2023

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
from twophotonUtils import parse_XML_timestamps

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


dirnames = {}

dirnames['DKO_R1'] = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Left ear/Post tam/R1/'
dirnames['WT_R1'] = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Right ear/Post Ethanol/R1/'

dx = {}
dx['DKO_R1'] = 0.2919921875
dx['WT_R1'] = 0.2919921875
dz = {}
dz['DKO_R1'] = 0.7
dz['WT_R1'] = 0.7

mouse = {'DKO_R1':'DKOM1','WT_R1':'DKOM1'}

pairs = {'DKOM1':'Pair 1'}
beginning = {'DKO_R1':10,'WT_R1':10}

RECALCULATE = True

#%% Load and collate manual track+segmentations
# Dictionary of manual segmentation (there should be no first or last time point)

for name,dirname in dirnames.items():
    
    for mode in ['curated']:

        timestamps = list(parse_XML_timestamps(dirname, beginning=beginning[name]).values())
        timestamps = np.array([(x-timestamps[0]).total_seconds()/3600 for x in timestamps])

        print(f'---- Working on {name} {mode} ----')
        if name == 'WT_R4' and mode == 'manual':
            continue
        
        genotype = name.split('_')[0]
        
        # Construct pathnames
        pathdict = {}
        pathdict['Segmentation'] = path.join(dirname,f'manual_tracking/{mode}_clahe.tif')
        pathdict['H2B'] = path.join(dirname,'master_stack/G.tif')
        pathdict['FUCCI'] = path.join(dirname,'master_stack/R.tif')
        pathdict['Frame averages'] = ''
        pathdict['Cell cycle annotations'] = path.join(dirname,f'{name}_cell_cycle_annotations.xlsx')
        
        # Construct metadata
        metadata = {}
        metadata['um_per_px'] = dx[name]
        metadata['um_per_slice'] = dz[name]
        metadata['Region'] = name
        metadata['Mouse'] = mouse[name]
        metadata['Pair'] = pairs[mouse[name]]
        metadata['Genotype'] = genotype
        metadata['Mode'] = mode
        metadata['Dirname'] = dirname
        metadata['Time stamps'] = timestamps
        
        
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




  
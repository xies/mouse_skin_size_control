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

from measureSemiauto import measure_track_timeseries_from_segmentations, \
    cell_cycle_annotate,collate_timeseries_into_cell_centric_table,annotate_ablation_distance

from twophotonUtils import parse_XML_timestamps

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

dirnames = {}
# dirnames['Ablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'
# dirnames['Ablation_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-23-2023 R26CreER Rb-fl no tam ablation/R2/'
# dirnames['Ablation_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R1'
# dirnames['Ablation_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2'
# dirnames['Ablation_R5'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R1'
# dirnames['Ablation_R6'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R2'

# Mouse 2
# dirnames['Ablation_R11'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/08-14-2023 R26CreER Rb-fl no tam ablation 24hr/M5 white/R3'
# dirnames['Ablation_R12'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/08-23-2023 R26CreER Rb-fl no tam ablation 16h/M5 White DOB 4-25-2023/R1/'
# dirnames['Ablation_R13'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/08-23-2023 R26CreER Rb-fl no tam ablation 16h/M5 White DOB 4-25-2023/R2'
# dirnames['Ablation_R14'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/09-27-2023 R26CreER Rb-fl no tam ablation M5/M5 white DOB 4-25-23/R1'
# dirnames['Ablation_R16'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/10-04-2023 R26CreER Rb-fl no tam ablation M5/M5 white DOB 4-25-23/R1'

# dirnames['Ablation_R18'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M1 M2 K14 Rbfl DOB DOB 06-01-2023/01-13-2024 Ablation K14Cre H2B FUCCI/Black unclipped less leaky DOB 06-30-2023/R2/'
dirnames['Ablation_R20'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M1 M2 K14 Rbfl DOB DOB 06-01-2023/01-13-2024 Ablation K14Cre H2B FUCCI/Black right clipped DOB 06-30-2023/R1'

dx = {}
dx['Ablation_R1'] = 0.14599609375
dx['Ablation_R2'] = 0.14599609375
dx['Ablation_R3'] = 0.194661458333333
dx['Ablation_R4'] = 0.194661458333333
dx['Ablation_R5'] = 0.194661458333333
dx['Ablation_R6'] = 0.194661458333333
dx['Ablation_R11'] = 0.194661458333333
dx['Ablation_R12'] = 0.194661458333333
dx['Ablation_R13'] = 0.194661458333333
dx['Ablation_R14'] = 0.194661458333333
dx['Ablation_R16'] = 0.194661458333333
dx['Ablation_R18'] = 0.2919922
dx['Ablation_R20'] = 0.2919922

dz = {}
dz['Ablation_R1'] = 1
dz['Ablation_R2'] = 1
dz['Ablation_R3'] = 1
dz['Ablation_R4'] = 1
dz['Ablation_R5'] = 1
dz['Ablation_R6'] = 1

dz['Ablation_R11'] = .7
dz['Ablation_R12'] = .7
dz['Ablation_R13'] = .7
dz['Ablation_R14'] = .7
dz['Ablation_R16'] = 1
dz['Ablation_R18'] = .7
dz['Ablation_R20'] = .7

mouse = {'Ablation_R1':'WT_F1'
         ,'Ablation_R2':'WT_F1'
         ,'Ablation_R3':'WT_F1'
         ,'Ablation_R4':'WT_F1'
         ,'Ablation_R5':'WT_F1'
         ,'Ablation_R6':'WT_F1'
         ,'Ablation_R11':'WT_M5'
         ,'Ablation_R12':'WT_M5'
         ,'Ablation_R13':'WT_M5'
         ,'Ablation_R14':'WT_M5'
         ,'Ablation_R16':'WT_M5'
         ,'Ablation_R18':'WT_Mnonclip'
         ,'Ablation_R20':'WT_Mclip'}


pairs = {'WT_F1':np.nan,'WT_M5':np.nan,'WT_Mnonclip':np.nan,'WT_Mclip':np.nan}

RECALCULATE = True

# timestamps = {'Ablation_R1':np.array([0,2,4,7,11,23,36])
#               ,'Ablation_R3':np.array([0,12,16,20,24,36])
#               ,'Ablation_R4':np.array([0,12,16,20,24,36])
#               ,'Ablation_R5':np.array([0,8,12,16,19])
#               ,'Ablation_R6':np.array([0,8,12,16,19])
#               ,'Ablation_R11':np.array([0,0.1,22,25,29,33,37])
#               ,'Ablation_R12':np.array([0,0.1,16,20,21,27,32,46])
#               ,'Ablation_R13':np.array([0,0.1,16,20,21,27,32,46])
#               ,'Ablation_R14':np.array([0,0.1,16,22,28,33,48])}

beginning = {'Ablation_R1':1,
             'Ablation_R2':1,
             'Ablation_R3':1,
             'Ablation_R4':1,
             'Ablation_R5':1,
             'Ablation_R6':1,
             'Ablation_R11':0,
             'Ablation_R12':0,
             'Ablation_R13':0,
             'Ablation_R14':0,
             'Ablation_R16':0,
             'Ablation_R18':0,
             'Ablation_R20':0
             }

#%% Load and collate manual track+segmentations
# Dictionary of manual segmentation (there should be no first or last time point)

for name,dirname in dirnames.items():
    
    timestamps = list(parse_XML_timestamps(dirname, subdir_str='*.*', beginning=beginning[name]).values())
    timestamps = np.array([(x-timestamps[0]).total_seconds()/3600 for x in timestamps])
    
    for mode in ['Nonablation']:

        print(f'---- Working on {name} {mode} ----')
        
        # Load ablation coordinates        
        ablation_coords = pd.read_csv(path.join(dirnames[name],'manual_tracking/ablation_xyz.csv')
                              ,index_col=0,names=['T','Z','Y','X'],header=0)
        
        # Construct pathnames
        pathdict = {}
        pathdict['Segmentation'] = path.join(dirname,'manual_tracking',f'{name}_{mode}.tif')
        pathdict['H2B'] = path.join(dirname,'master_stack/G.tif')
        pathdict['FUCCI'] = path.join(dirname,'master_stack/R.tif')
        pathdict['Frame averages'] = path.join(dirname,'high_fucci_avg_size.csv')
        pathdict['Cell cycle annotations'] = path.join(dirname,f'{name}_cell_cycle_annotations.xlsx')
        
        # Construct metadata
        metadata = {}
        metadata['um_per_px'] = dx[name]
        metadata['um_per_slice'] = dz[name]
        metadata['Region'] = name
        metadata['Mouse'] = mouse[name]
        metadata['Pair'] = pairs[mouse[name]]
        metadata['Genotype'] = 'WT'
        metadata['Mode'] = mode
        metadata['Dirname'] = dirname
        metadata['Time stamps'] = timestamps
        metadata['Ablated cell coords'] = ablation_coords
        
        #% Re-construct tracks with manually fixed tracking/segmentation
        tracks = measure_track_timeseries_from_segmentations(name,pathdict,metadata)
        tracks = annotate_ablation_distance(tracks,metadata)
        tracks = cell_cycle_annotate(tracks,pathdict,metadata)
        
        # Construct the cell-centric metadata dataframe
        df,tracks = collate_timeseries_into_cell_centric_table(tracks,metadata)
        
        df.to_csv(path.join(dirname,f'manual_tracking/{name}_{mode}_dataframe.csv'))
        # Save to the manual tracking folder    
        with open(path.join(dirname,'manual_tracking',f'{name}_{mode}_dense.pkl'),'wb') as file:
            pkl.dump(tracks,file)
                
#%%

for t in tracks[80:90]:
    
    print(t.CellID)
    print(t.Volume)
    plt.plot(t.Age,t['Volume normal'])


  
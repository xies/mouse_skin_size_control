#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:06:53 2023

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, measure
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl

from basicUtils import *

dirnames = {}

dirnames['Ablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'
dirnames['Nonablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'

#%%

all_tracks = {}
ts_all = {}
regions = {}
for name,dirname in dirnames.items():
    for mode in ['curated']:
        if name == 'WT_R4' and mode == 'manual':
            continue
        
        with open(path.join(dirname,'manual_tracking',f'{name}_dense_{mode}.pkl'),'rb') as file:
            tracks = pkl.load(file)
        
        for t in tracks:
            t['Time to G1/S'] = t['Frame'] - t['S phase entry frame']
            # t['Volume interp'] = smooth_growth_curve
            
        all_tracks[name+'_'+mode] = tracks
        
        ts_all[name+'_'+mode] = pd.concat(tracks)
        
        df = pd.read_csv(path.join(dirname,f'manual_tracking/{name}_dataframe_{mode}.csv'),index_col=0)
        df['Division size'] = df['Birth size'] + df['Total growth']
        df['S entry size'] = df['Birth size'] + df['G1 growth']
        df['Log birth size'] = np.log(df['Birth size']) 
        df['Fold grown'] = df['Division size'] / df['Birth size']
        df['SG2 growth'] = df['Total growth'] - df['G1 growth']
        regions[name+'_'+mode] = df

df_all = pd.concat(regions,ignore_index=True)
ts_all = pd.concat(ts_all,ignore_index=True)


for t in all_tracks['Nonablation_R1_curated']:
    plt.plot(t.Age, t['Volume interp'],'b-')
    

for t in all_tracks['Ablation_R1_curated']:
    plt.plot(t.Age, t['Volume interp'],'r-')
    
#%%



spl.derivative(1)([0,2,4,7])


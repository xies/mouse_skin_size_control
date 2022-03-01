#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:38:58 2021

@author: xies
"""

import numpy as np
import pandas as pd

import pickle as pkl

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M1 WT/R1/'

#%%

with open(path.join(dirname,'complete_cycles_seg.pkl'),'rb') as file:
    tracks = pkl.load(file)

#%% Process birth time for each time series



for track in tracks:
    
    frames = track['Frame']
    first_frame = frames.min()
    track['t'] = (track['Frame'] - first_frame) / 2.0 # half-day
    
ts = pd.concat(tracks)

#%% 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:05:27 2022

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


dirnames = {}
dirnames['WT1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'
dirnames['WT2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022 RB-KO pair/M1 WT/R1'

dirnames['RBKO1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
dirnames['RBKO2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022 RB-KO pair/M6 RBKO/M1 R1'

#%%

# all_tracks = []

regions = []
with name,dirname in dirnames.items():
    
    # with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'rb') as file:
    #     tracks = pkl.load(file)
    df = 
    regions.append(
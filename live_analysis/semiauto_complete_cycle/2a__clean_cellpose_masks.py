#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:33:51 2022

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import measure
from tqdm import tqdm

dirnames = []
dirnames.append('/Users/xies/OneDrive - Stanford/Skin/Confocal/07-19-2022 Skin/DS 06-25-22 H2B-Cerulean FUCCI2 Phall647')
dirnames.append('/Users/xies/OneDrive - Stanford/Skin/Confocal/07-19-2022 Skin/DS 06-25-22 H2BCerulean FUCCI2 Phall647 second')

#%%
# Load prediction by cellpose (.npy) and resave as tif
# see: https://github.com/MouseLand/cellpose/blob/main/docs/outputs.rst

filenames = []
filenames = filenames + glob(path.join(dirnames[0],'*/*.npy'))
# filenames= filenames + glob(path.join(dirnames[1],'*/*.npy'))

for f in filenames:ß
    out_name = path.splitext(f)[0] + '_seg.tif'
    if path.exists(out_name):
        continue
    data = np.load(f,allow_pickle=True).item()
    seg = data['masks']
    io.imsave(out_name, seg)
    io.imsave(path.splitext(f)[0] + '_prob.tif',data['flows'][3])
    
#%%

# Filters
smallest = 2000 #pixels
z_cutoff = 50 #from top


def find_matches_in_array(arr, vals2match):
    
    shape = arr.shape
    arr_fl = arr.flatten()
    I = np.in1d(arr_fl,vals2match)
    
    I = np.reshape(I,shape)
    return I

filenames = glob(path.join(dirname,f'im_seq/*_seg.tif'))

for f in filenames:
    
    seg_raw = io.imread(f)
    
    df = pd.DataFrame(measure.regionprops_table(seg_raw,properties=['label','area','centroid']))

    bad_labels = df[(df['area'] < smallest) |
                     (df['centroid-0'] < z_cutoff )]['label']
    
    I = find_matches_in_array(seg_raw, bad_labels.values)
    seg_raw[I] = 0
    
    io.imsave(path.splitext(f)[0] + '_clean.tif', seg_raw)
    
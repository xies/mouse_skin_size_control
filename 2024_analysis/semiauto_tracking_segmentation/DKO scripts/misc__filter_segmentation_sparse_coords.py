#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:32:43 2024

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, morphology, util, measure
from scipy.ndimage import convolve
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl

# dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Right ear/Post Ethanol/R1/'
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Left ear/Post tam/R1/'

# dx = 0.2920097
dx = 1

#%% Load parsed tracks, previous manual segtrack, additional segonly

MANUAL = False

# Load preliminary tracks
coords2filter = pd.read_csv(path.join(dirname,'Cell cycle/births-Spot.csv'),index_col=0,
                             header=1)
coords2filter = coords2filter.rename(columns={'Unnamed: 1':'SpotID'})
coords2filter = coords2filter.iloc[1:].astype(float)

#%% Load a bunch of big files

segonly = []
for t in range(18):
    segonly.append(io.imread(path.join(dirname,f'cellpose_B_clahe/t{t}_3d_nuc/t{t}_masks.tif')))
segonly = np.stack(segonly)

[T,Z,Y,X] = segonly.shape

# Load dfield files
dfields = {}
for t in range(1,15):
    if t == 9:
        continue
    dfield_file = glob(path.join(dirname,f'{t+10}. Day */dfield.tif'))
    
    assert(len(dfield_file) == 1)
    dfields[t] = io.imread(path.join(dfield_file[0]))
    
#%% Use tracks and extract segmentation; generate a filtered segmentation image
# where only tracked spots are shown + put 3D markers on un-segmented spots

radius = 5

[T,Z,X,Y] = segonly.shape

# Filter segmentation based on complete tracks

filtered = np.zeros_like(segonly)

for idx,spot in tqdm(coords2filter.iterrows()):
    if np.isnan(spot.FRAME):
        continue
    if t == 9:
        continue
    if t > 9:
        t = t-1
    
    x = int(spot['X']/dx)
    y = int(spot['Y']/dx)
    z = int(np.round(spot['Z']))
    t = int(spot['FRAME'])
    
    if t in dfields.keys():
        dfield = dfields[t]
        warped_coords = np.array([z,y,x]).astype(int)
        unwarped_coords = np.round(warped_coords + dfield[tuple(warped_coords)][::-1])
        
        z = int(unwarped_coords[0])
        y = int(unwarped_coords[1])
        x = int(unwarped_coords[2])
    
    this_seg = segonly[t,...]
    label = this_seg[z,y,x]
    
    if label > 0:
        # filter¸segmentation image to only include tracked spots
        filtered[t,this_seg == label] = label
    else:
        # Create a 'cube' around spots that are missing segmentations
        # print(f'Time {t} -- {i}: {ID}')
        y_low = max(0,y - radius); y_high = min(Y,y + radius)
        x_low = max(0,x - radius); x_high = min(X,x + radius)
        z_low = max(0,z - radius); z_high = min(Z,z + radius)
        filtered[t,z_low:z_high, y_low:y_high, x_low:x_high] = label
        print(label)
        if t == 4:
            stop

io.imsave(path.join('/Users/xies/Desktop/births.tif'),
  filtered.astype(np.uint16))





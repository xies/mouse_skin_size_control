#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:47:01 2024

@author: xies
"""

from skimage import io, segmentation, measure
import numpy as np
import pandas as pd
from os import path
import pickle as pkl
from tqdm import tqdm

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Right ear/Post Ethanol/R1/'

filename = path.join(dirname,'master_stack/B_clahe.tif')
B = io.imread(filename)
filename = path.join(dirname,'master_stack/R.tif')
R = io.imread(filename)

labels = io.imread(path.join(dirname,'manual_tracking/DKOM1_WT_R1.tif'))
summary = pd.read_csv(path.join(dirname,'manual_tracking/DKOM1_WT_R1_dataframe_curated.csv'))

df = []
for t,im in enumerate(labels):
    _df = pd.DataFrame(measure.regionprops_table(im,properties=['area','centroid','label']))
    _df['Frame'] = t
    df.append(_df)
df = pd.concat(df,ignore_index=True)
df = df.rename(columns={'centroid-0':'Z',
                        'centroid-1':'Y',
                        'centroid-2':'X',})

cells = {cellID: cell for cellID,cell in df.groupby('label')}

# Build the display stack
contours = np.zeros_like(labels)
for t,im in tqdm(enumerate(labels)):
    for z,this_slice in enumerate(im):
        contours[t,z,...] = segmentation.expand_labels(this_slice,distance=1)

contours =- labels

#%%

XYborder = 50
Zborder = 15

cellOI = 31

cell = cells[cellOI]

X = cell['X'].values.astype(int)
Y = cell['Y'].values.astype(int)
Z = cell['Z'].values.astype(int)
T = cell['Frame'].values.astype(int)

patch = np.zeros((3,len(T),2*Zborder,2*XYborder,2*XYborder))
for i,t in enumerate(T):
    
    patch[0,i,...] = R[t,Z[i]-Zborder:Z[i]+Zborder,
                         Y[i]-XYborder:Y[i]+XYborder,
                         X[i]-XYborder:X[i]+XYborder,]
    patch[1,i,...] = B[t,Z[i]-Zborder:Z[i]+Zborder,
                         Y[i]-XYborder:Y[i]+XYborder,
                         X[i]-XYborder:X[i]+XYborder,]
    patch[2,i,...] = contours[t,Z[i]-Zborder:Z[i]+Zborder,
                         Y[i]-XYborder:Y[i]+XYborder,
                         X[i]-XYborder:X[i]+XYborder,]
    
io.imsave('/Users/xies/Desktop/patch.tif',patch)
    
    
    
    
    
    
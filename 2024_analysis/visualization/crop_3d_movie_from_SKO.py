#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:26:19 2025

@author: xies
"""


from skimage import io, measure, morphology
import numpy as np
import pandas as pd
from os import path
import pickle as pkl
from tqdm import tqdm
import pickle as pkl

from mamutUtils import load_mamut_densely, construct_data_frame_dense


dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Right ear/Post Ethanol/R1/'
# dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Left ear/Post tam/R1'

filename = path.join(dirname,'master_stack/B_clahe.tif')
B = io.imread(filename)
filename = path.join(dirname,'master_stack/R.tif')
R = io.imread(filename)

labels = io.imread(path.join(dirname,'manual_tracking/curated_clahe.tif'))


_tracks, _links, _spots = load_mamut_densely(dirname,subdir_str='MaMuT')
tracks = construct_data_frame_dense(_tracks, _links, _spots)

#%% Example Cell 1

idx = 1

trackOI = tracks[idx]

motherID = trackOI.iloc[0].Mother
daughter_a_ID = trackOI.iloc[0].Left # Always pick left?

# mother_idx  = np.where([ np.any(t.ID == motherID) for t in tracks])[0][0]
motherOI = tracks[0]

# daughter_idx  = np.where([ np.any(t.ID == daughter_a_ID) for t in tracks])[0][0]
daughterOI = tracks[7]

tree = pd.concat((motherOI , trackOI ),ignore_index=True)

XYborder = 100
Zborder_min = 15
Zborder_max= 20

X = tree['X'].values.astype(int)
Y = tree['Y'].values.astype(int)
Z = tree['Z'].values.astype(int)
T = tree['Frame'].values.astype(int)

# Manually adjust as needed
X[4] += 30
Y[4] += 40
X[5] += 20
X[6] += 20

patch = np.zeros((3,len(T),Zborder_min+ Zborder_max,2*XYborder,2*XYborder))
for i,t in enumerate(T):

    patch[0,i,...] = R[t,Z[i]-Zborder_min:Z[i]+Zborder_max,
                         Y[i]-XYborder:Y[i]+XYborder,
                         X[i]-XYborder:X[i]+XYborder,]
    patch[1,i,...] = B[t,Z[i]-Zborder_min:Z[i]+Zborder_max,
                         Y[i]-XYborder:Y[i]+XYborder,
                         X[i]-XYborder:X[i]+XYborder,]
    patch[2,i,...] = labels[t,Z[i]-Zborder_min:Z[i]+Zborder_max,
                         Y[i]-XYborder:Y[i]+XYborder,
                         X[i]-XYborder:X[i]+XYborder,]

io.imsave('/Users/xies/Desktop/patch0.tif',patch[0,...])
io.imsave('/Users/xies/Desktop/patch1.tif',patch[1,...])

io.imsave('/Users/xies/Desktop/patch2.tif',patch[2,...])    

#%% Example Cell 2

idx = 326
Zborder_min = 15
Zborder_max= 12

trackOI = tracks[idx]

motherID = trackOI.iloc[0].Mother
daughter_a_ID = trackOI.iloc[0].Left # Always pick left?

mother_idx  = np.where([ np.any(t.ID == motherID) for t in tracks])[0][0]
motherOI = tracks[320]

daughter_idx  = np.where([ np.any(t.ID == daughter_a_ID) for t in tracks])[0][0]
daughterOI = tracks[327]

tree = pd.concat((motherOI , trackOI , daughterOI),ignore_index=True)

X = tree['X'].values.astype(int)
Y = tree['Y'].values.astype(int)
Z = tree['Z'].values.astype(int)
T = tree['Frame'].values.astype(int)

# Manually adjust as needed
Z[0] += 2
Z[2] += 2
Z[3] += 2
Y[3] -= 10
Z[4] += 6
X[4] -= 20
X[5] -= 20
Y[6] -= 20
Z[7] += 5
Y[7] -= 10
Z[8] -= 10
X[8] -= 15

patch = np.zeros((3,len(T),Zborder_min+ Zborder_max,2*XYborder,2*XYborder))
for i,t in enumerate(T):
    
    patch[0,i,...] = R[t,Z[i]-Zborder_min:Z[i]+Zborder_max,
                         Y[i]-XYborder:Y[i]+XYborder,
                         X[i]-XYborder:X[i]+XYborder,]
    patch[1,i,...] = B[t,Z[i]-Zborder_min:Z[i]+Zborder_max,
                         Y[i]-XYborder:Y[i]+XYborder,
                         X[i]-XYborder:X[i]+XYborder,]
    patch[2,i,...] = labels[t,Z[i]-Zborder_min:Z[i]+Zborder_max,
                         Y[i]-XYborder:Y[i]+XYborder,
                         X[i]-XYborder:X[i]+XYborder,]

io.imsave('/Users/xies/Desktop/patch0.tif',patch[0,...])
io.imsave('/Users/xies/Desktop/patch1.tif',patch[1,...])

io.imsave('/Users/xies/Desktop/patch2.tif',patch[2,...])    

    
    
    
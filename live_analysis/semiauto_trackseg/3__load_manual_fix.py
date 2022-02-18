#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:22:31 2021

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io
import seaborn as sb
from os import path
from glob import glob

import pickle as pkl

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/10-20-2021/WT/R1'

# dx = 0.2920097
dx = 1

#%% Load and collate manual track+segmentations ()

# Dictionary of manual segmentation (there should be no first or last time point)
manual_segs = io.imread(path.join(dirname,'manual_tracking/manual_tracking.tif'))

# Resave as 16-bit
# io.imsave(path.join(dirname,'manual_tracking/manual_tracking_8bit.tif'),manual_segs.astype(np.int8))

filename = glob(path.join(dirname,'reg/*.tif'))
imstack = list(map(io.imread,filename))

#%% Re-construct tracks with manually fixed tracking/segmentation

trackIDs = np.unique(manual_segs)

tracks = []
for trackID in trackIDs[1:]:
    
    track = pd.DataFrame()
    
    mask = manual_segs == trackID
    frames_with_this_track = np.where(np.any(np.any(np.any(mask,axis=1),axis=1), axis=1))[0]
    
    
    
    for frame in frames_with_this_track:
        
        # Measurements from segmentation/ label iamge
        this_frame = mask[frame,...]
        
        props = measure.regionprops(this_frame*1)
        Z,X,Y = props[0]['Centroid']
        volume = props[0]['Area']
        
        # Measurement from intensity image(s)
        fucci_this_frame = imstack[frame][:,:,:,0]
        props = measure.regionprops(this_frame*1, intensity_image = fucci_this_frame)
        fucci_mean = props[0]['mean_intensity']
        
        track = track.append(pd.Series({'Frame':frame,'X':X,'Y':Y,'Z':Z,'Volume':volume,'FUCCI':fucci_mean}),
                             ignore_index=True)
        
    track['CorrID'] = trackID
    track['Age'] = (track['Frame'] - track.iloc[0]['Frame'])*12
    
    tracks.append(track)
    print(f'Done with corrID {trackID}')



# Save to the manual folder    
with open(path.join(dirname,'manual_tracking/complete_cycles_fixed.pkl'),'wb') as file:
    pkl.dump(tracks,file)




ts = pd.concat(tracks)
  
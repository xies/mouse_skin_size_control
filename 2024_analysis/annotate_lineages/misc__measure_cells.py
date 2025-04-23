#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 13:45:04 2025

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from skimage import io, measure

import seaborn as sb
from os import path
from tqdm import tqdm
import pickle as pkl

from mamutUtils import load_mamut_xml_densely, construct_data_frame_dense

dirname ='/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/'

track_seg = io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif'))

#%%

all_labels = np.unique(track_seg)[1:]

tracks = []

for t in tqdm(range(15)):
    labels = segmentation.clear_border(track_seg[t,:])
    props = pd.DataFrame(measure.regionprops_table(labels,properties=['area','label']))
    props['Frame'] = t
    
    tracks.append(props)

tracks = pd.concat(tracks,ignore_index=True)
tracks = [t for _,t in tracks.groupby('label')]

#%%

plt.close('all')
trackID = 140

for t in tracks[trackID:trackID+10]:
    
    plt.plot(t.Frame - t.iloc[0]['Frame'],t['area'])
    plt.show()
    
    
    
    
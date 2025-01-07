#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:58:19 2021

Parses .csv output of Mamut and prunes out complete cell cycles

Exports (pickle) as a list of dataframes, each corresponding to a complete cycle (from birth to division).
Exported fields:
    SpotID  X    Y    Z    T     Left child      Right child    Division(flag)     Terminus (flag)

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import seaborn as sb
from os import path

import pickle as pkl

from mamutUtils import load_mamut_densely, construct_data_frame_dense

#%% Export the coordinates of the completed cell cycles (as pickle)

dirnames = []
# dirnames.append('/Volumes/T7/01-13-2023 Ablation K14Cre H2B FUCCI/Black right clipped DOB 06-30-2023/R1/')
# dirnames.append('/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Left ear/Post tam/R1/')
dirnames.append('/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Left ear/Post tam/R1')

all_tracks = []
for dirname in dirnames:
    _tracks, _links, _spots = load_mamut_densely(dirname,subdir_str='MaMuT')
    tracks = construct_data_frame_dense(_tracks, _links, _spots)
    tracks = construct_data_frame_dense(_tracks, _links, _spots)

    with open(path.join(dirname,'dense_tracks.pkl'),'wb') as file:
        pkl.dump(tracks,file)

    all_tracks.append(tracks)
    
#%%

# plt.boxplot([wtlength,rbkolength],labels=['WT','RB-KO'])
# plt.ylabel('Cell cycle length (h)')

# plt.figure()

# plt.hist(wtlength,12,histtype='step');plt.hist(rbkolength,12,histtype='step')
# plt.legend(['WT','RB-KO'])

# plt.xlabel('Cell cycle length (h)')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:58:19 2021

Parses .csv output of Mamut and pru"nes out complete cell cycles

Exports (pickle) as a list of dataframes, each corresponding to a complete cycle (from birth to division).
Exported fields:
    SpotID  X    Y    Z    T     Left child      Right child    Division(flag)     Terminus (flag)

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from xml.etree import ElementTree as et

import seaborn as sb
from os import path

import pickle as pkl

from mamutUtils import load_mamut_xml_prune_for_complete_cycles, construct_data_frame_complete_cycles

#%% Export the coordinates of the completed cell cycles (as pickle)

dirnames = []
dirnames.append('/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/Old mice/04-30-2024 16month old mice/M3 DOB 12-27-2022/R1/')

all_tracks = []
for dirname in dirnames:
    cycling_tracks,cycling_spots = load_mamut_xml_prune_for_complete_cycles(dirname)
    # cycling_tracks, cycling_links, cycling_spots = load_mamut_and_prune_for_complete_cycles(dirname)
    tracks = construct_data_frame_complete_cycles(cycling_tracks, cycling_spots)

    with open(path.join(dirname,'R1-mamut.pkl'),'wb') as file:
        pkl.dump(tracks,file)

    all_tracks.append(tracks)
    
#%% XML Parser

filename = path.join(dirname,'MaMuT/R1-mamut.xml')
tree = et.parse(filename)
root = tree.getroot()
model = root.find('Model')

if model is None:
    print('Model not found.')
    
spots = [spot for spot in model.iter('Spot')]
edges = [edge for edge in model.iter('Edge')]
tracks = [track for track in model.iter('Track')]

#%%

rbko = all_tracks[0]
# wt = all_tracks[1]

# wtlength  = (np.array([len(t) for t in wt])* 12)
rbkolength  = (np.array([len(t) for t in rbko])* 12)

# plt.boxplot([wtlength,rbkolength],labels=['WT','RB-KO'])
# plt.ylabel('Cell cycle length (h)')

# plt.figure()

# plt.hist(wtlength,12,histtype='step');plt.hist(rbkolength,12,histtype='step')
# plt.legend(['WT','RB-KO'])

# plt.xlabel('Cell cycle length (h)')


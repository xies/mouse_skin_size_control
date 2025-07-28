#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:27:01 2025

@author: xies
"""

# Core libraries
import numpy as np
# from skimage import io
import pandas as pd
import matplotlib.pylab as plt
from scipy import spatial

# General utils
from tqdm import tqdm
from os import path
# from basicUtils import nonans

dx = 0.25
dz = 1

# Filenames
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

#%% Normalize by frame and then re-measure

all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated.csv'),index_col=0)

delam = all_df[ (all_df['Delaminate next frame']) & ~(all_df['Border'])].reset_index()
distances_delaminating = spatial.distance_matrix(delam[['X','Y']],delam[['X','Y']])
distances_delaminating[np.eye(len(distances_delaminating)) > 0] = np.nan

divide = all_df[ (all_df['Divide next frame']) & ~(all_df['Border'])].reset_index()
distances_dividing = spatial.distance_matrix(divide[['X','Y']],divide[['X','Y']])
distances_dividing[np.eye(len(distances_dividing)) > 0] = np.nan

Niter = 100
random_distances = []
for i in range(Niter):
    
    x = all_df[ (~all_df['Border']) & (all_df['Cell type'] == 'Basal') 
               ].sample(len(distances_delaminating)).reset_index()
    D = spatial.distance_matrix(x[['X','Y']],x[['X','Y']])
    D[np.eye(len(D)) > 0] = np.nan
    random_distances.append(D)

mean_random_distances = np.stack(random_distances).mean(axis=0)

plt.hist(distances_delaminating.flatten(),bins=25,histtype='step')
plt.hist(distances_dividing.flatten(),bins=25,histtype='step')
plt.hist(random_distances[50].flatten(),bins=25,histtype='step')

plt.legend(['Delaminating','Dividing','Random cells'])
plt.xlabel('Distance between cells/events (um)')
plt.ylabel('Count')


#%%

plt.scatter(delam['X'],delam['Y'],color='b')
plt.scatter(divide['X'],divide['Y'],color='r')
plt.legend(['Delaminating','Dividing'])



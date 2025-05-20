#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 12:20:25 2025

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
import napari
from skimage import io
from cellpose import models
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'))

dz = 1; dx = 0.25

tracks = [t for _,t in all_df.groupby('TrackID')]
model = models.Cellpose(model_type='nuclei')

#%%

track = tracks[3]
track = track[track['Cell type'] == 'Basal']

coords = track[['Frame','Z','Y-pixels','X-pixels']].astype(int)
xy_border = 20; z_border = 10

im_seq = np.array([io.imread(path.join(dirname,f'h2b_sequence/t{t}.tif')) for t in range(15)])
 
im_cropped = im_seq[coords['Frame'].iloc[0]: coords['Frame'].iloc[-1],
                coords['Z'].min() - z_border: coords['Z'].max() + z_border,
                coords['Y-pixels'].min() - xy_border: coords['Y-pixels'].max() + xy_border,
                coords['X-pixels'].min() - xy_border: coords['X-pixels'].max() + xy_border]

masks = {}
flows = {}
for cellprob_threshold in tqdm(np.linspace(0.1,.8,5)):
    masks[cellprob_threshold] = \
        np.array([model.eval(im_cropped[t,...],diameter=None, 
                             do_3D=True,cellprob_threshold=cellprob_threshold, anisotropy=4)[0]
        for t in range(6)])

#%%

viewer = napari.Viewer()
viewer.add_image(im_cropped)
for prob,labels in masks.items():
    viewer.add_labels(labels,name = f'prob = {prob}',blending='additive')

#%%

new_coords = coords.copy()
new_coords['Z'] = coords['Z'] + z_border - coords['Z'].min()
new_coords['Y-pixels'] = coords['Y-pixels'] + xy_border - coords['Y-pixels'].min()
new_coords['X-pixels'] = coords['X-pixels'] + xy_border - coords['X-pixels'].min()

vol_by_th = {}
for prob,labels in masks.items():

    vols = np.array([ (labels == labels[t,new_coords['Z'].iloc[t],
                    new_coords['Y-pixels'].iloc[t],
                    new_coords['X-pixels'].iloc[t]]).sum() for t in range(6)])
    vol_by_th[prob] = vols

vols = pd.DataFrame(vol_by_th).T
vols.index.name = 'Threshold'

import seaborn as sb
sb.lineplot(vols.T)

#%%


    
    
    
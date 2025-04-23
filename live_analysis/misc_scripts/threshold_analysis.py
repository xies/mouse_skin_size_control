#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:39:35 2025

@author: xies
"""

from tqdm import tqdm
import pandas as pd
from skimage import io, measure, segmentation, morphology

labels = io.imread('/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/3d_nuc_seg_supra/cellpose_manual/t8.tif')
im = io.imread('/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/Cropped_images/B.tif')[8,...]


#%%

df = []
for d in tqdm(range(-10,10)):
    if d < 0:
        _labels = morphology.erosion(labels,morphology.ball(-d))
    elif d > 0:
        _labels = segmentation.expand_labels(labels,d)
    elif d == 0:
        _labels = labels
    _df = pd.DataFrame(measure.regionprops_table(_labels, intensity_image = im,
                              properties=['label','area','mean_intensity']))
    _df['Dilation'] = d
    _df['Total intensity'] = _df['area']*_df['mean_intensity']
    df.append(_df)
df = pd.concat(df,ignore_index=True)

#%%

pivot = pd.pivot_table(df,values='mean_intensity',index=['label','Dilation'])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:10:17 2023

@author: xies
"""


import numpy as np
import pandas as pd
from skimage import io,morphology, filters, measure, util
from imageUtils import colorize_segmentation

from os import path
from glob import glob
from tqdm import tqdm

import seaborn as sb

from matplotlib.path import Path
from SelectFromCollection import SelectFromCollection

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

#%%

alpha_threshold = 1

by_frame = []
for t in tqdm(range(15)):
    this_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    im = io.imread(path.join(dirname,f'im_seq/t{t}.tif'))
    R = im[...,0]
    # df = pd.DataFrame(measure.regionprops_table(this_seg,intensity_image=R,properties=['label','area','intensity_mean']))
    Nlabels = len(np.unique(this_seg))-1
    labels = np.zeros(Nlabels)
    volumes = np.zeros(Nlabels)
    mean_fucci = np.zeros(Nlabels)
    bg_mean_fucci = np.zeros(Nlabels)
    bg_std_fucci = np.zeros(Nlabels)
    fucci_bg_sub = np.zeros(Nlabels)
    for i,p in enumerate(measure.regionprops(this_seg,intensity_image=R)):
        labels[i] = p['label']
        volumes[i] = p['area']
        mean_fucci[i] = p['intensity_mean']
        
        bbox = p['bbox']
        local_fucci = R[bbox[0]:bbox[3],bbox[1]:bbox[4],bbox[2]:bbox[5]]
        local_seg = p['image']
        footprint = morphology.cube(5)
        local_seg = morphology.dilation(local_seg,footprint=footprint)
        
        bg_mean = local_fucci[~local_seg].mean()
        bg_std = local_fucci[~local_seg].std()
        bg_mean_fucci[i] = bg_mean
        bg_std_fucci[i] = bg_std
        fucci_bg_sub[i] = mean_fucci[i] - bg_mean

    df = pd.DataFrame()
    df['CellposeID'] = labels
    df['FUCCI intensity'] = mean_fucci
    df['FUCCI background mean'] = bg_mean_fucci
    df['FUCCI background std'] = bg_std_fucci
    df['FUCCI bg sub'] = fucci_bg_sub
    df['Frame'] = t
    
    
    by_frame.append(df)
    
df = pd.concat(by_frame,ignore_index=True)

df['FUCCI thresholded'] = 'Low'
I = df['FUCCI intensity'] > df['FUCCI background mean'] + alpha_threshold * df['FUCCI background std']
df.loc[I,'FUCCI thresholded'] = 'High'
    
 #%% Polygon gating? (not sure how useful)

# th = filters.threshold_otsu(df['FUCCI intensity'].values)
# plt.hist(df['FUCCI intensity'],100,log=True)
# plt.vlines(x=th,ymin=0,ymax=1000,colors='r')

# df['FUCCI thresholded'] = 'Low'
# df.loc[df['FUCCI intensity'] > th,'FUCCI thresholded'] = 'High'

# ax = plt.scatter(df['Volume'],df['FUCCI bg sub'],alpha=0.05)
# plt.ylabel('FUCCI intensity')
# plt.xlabel('Cell size (px)')
# plt.xlim([0,25000])
# gate = roipoly()

# NB: select G2 cells!
# selector = SelectFromCollection(plt.gca(), ax)

# verts = np.array(selector.poly.verts)
# x = verts[:,0]
# y = verts[:,1]

# p_ = Path(np.array([x,y]).T)
# I = np.array([p_.contains_point([x,y]) for x,y in zip(df['Volume'],df['FUCCI bg sub'])])

# df['Phase'] = 'Low FUCCI'
# df.loc[I,'Phase'] = 'High FUCCI'

#%% Merge with the tissue annotations from step 5

ts = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col=0)
ts = ts.merge(df,on=['CellposeID','Frame'])
ts.to_csv(path.join(dirname,'tissue_dataframe.csv'))


#%% Generate images

for t in tqdm(range(15)):

    this_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    colored = colorize_segmentation(this_seg,
                                    {row['CellposeID']:row['FUCCI thresholded'] == 'High' for i,row in df[df['Frame'] == t].iterrows()} )
    
    io.imsave(path.join(dirname,f'Misc visualizations/High FUCCI/t{t}.tif'),
                        util.img_as_ubyte(colored))



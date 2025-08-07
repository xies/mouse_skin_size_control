#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 22:27:55 2025

@author: xies
"""

import numpy as np
from imageUtils import most_likely_label, fill_in_cube
from os import path
from skimage import io, measure
import pandas as pd
from glob import glob
from natsort import natsorted
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

filelist = natsorted(glob(path.join(dirname,'3d_cyto_seg/3d_cyto_manual/t*_cleaned.tif')))
previous_segs = np.stack([io.imread(f) for f in filelist])

filelist = natsorted(glob(path.join(dirname,'3d_cyto_seg/3d_cyto_raw/*/t*_masks.tif')))
raw_cytos = np.stack([io.imread(f) for f in filelist])

nuclear_segs = io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif'))


cyto_segs = np.zeros_like(nuclear_segs)
for t,im in tqdm(enumerate(nuclear_segs)):
    
    _df = pd.DataFrame(measure.regionprops_table(im, intensity_image = previous_segs[t,...],
                                                properties =['label','centroid'], extra_properties=[most_likely_label]))
    df = _df.rename(columns={'most_likely_label':'PreviousID','centroid-0':'Z','centroid-1':'Y','centroid-2':'X'})
    _df = pd.DataFrame(measure.regionprops_table(im, intensity_image = raw_cytos[t,...],
                                                properties =['label'], extra_properties=[most_likely_label]))
    _df = _df.rename(columns={'most_likely_label':'RawID'})
    
    df = df.merge(right=_df,left_on='label',right_on='label')
    df = df.replace(0,np.nan)
    df['CytoID'] = range(len(df))
    df['CytoID'] += 1
    
    # Construct filtered cyto_seg, starting with any curated segs first
    for idx,row in df.iterrows():
        
        if not np.isnan(row['PreviousID']):
            mask = measure.label((previous_segs[t,...] == row['PreviousID']).astype(int))
            _df =pd.DataFrame( measure.regionprops_table(mask,properties=['area','label']))
            mask = mask == _df.sort_values('area',ascending=False).iloc[0]['label']
            
            cyto_segs[t,mask] = row['CytoID']
        elif not np.isnan(row['RawID']):    
            mask = measure.label((raw_cytos[t,...] == row['RawID']).astype(int))
            _df =pd.DataFrame( measure.regionprops_table(mask,properties=['area','label']))
            mask = mask == _df.sort_values('area',ascending=False).iloc[0]['label']
            cyto_segs[t,mask] = row['CytoID']

        else:
            cyto_segs[t,...] = fill_in_cube(cyto_segs[t,...],
                                            row[['Z','Y','X']].astype(int),
                                            label = row['CytoID'].astype(int))
            
    io.imsave(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual_combined/t{t}.tif'), cyto_segs[t,...].astype(np.uint16))
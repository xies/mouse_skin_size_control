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

filelist = natsorted(glob(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/t*basal.tif')))
basal_segs = np.stack([io.imread(f) for f in filelist])

filelist = natsorted(glob(path.join(dirname,'im_seq/t*_3d_nuc/t*.tif')))
all_segs = np.stack([io.imread(f) for f in filelist])


suprabasal_segs = np.zeros_like(basal_segs)
for t,im in tqdm(enumerate(all_segs)):
    
    _df = pd.DataFrame(measure.regionprops_table(im, intensity_image = basal_segs[t,...],
                                                properties =['label','centroid'], extra_properties=[most_likely_label]))
    df = _df.rename(columns={'most_likely_label':'basalID','centroid-0':'Z','centroid-1':'Y','centroid-2':'X'})
  
    
    df = df.merge(right=_df,left_on='label',right_on='label')
    df = df[df['basalID'] == 0]
    
    for idx,row in df.iterrows():
        mask = basal_segs[t,...] == row['basalID']
        suprabasal_segs[t,mask] = row['label']
            
    io.imsave(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_suprabasal/t{t}_suprabasal.tif'),
              suprabasal_segs[t,...].astype(np.uint16))
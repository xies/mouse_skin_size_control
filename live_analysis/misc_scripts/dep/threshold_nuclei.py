#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:08:54 2022

@author: xies
"""

from skimage import io, filters
from skimage.filters.thresholding import _cross_entropy
from glob import glob
from os import path
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1/'

#%%

flist = glob(path.join(dirname,'*. Day*/ZSeries-*/G_reg_reg.tif'))

# im = io.imread(filename)

for f in tqdm(flist):
    im = io.imread(f)
    
    mask = np.zeros_like(im)
    # for i,im_ in enumerate(im):
    #     th = filters.threshold_local(im_,block_size=225)
    #     mask[i,...] = im_ > th
    th = filters.threshold_li(im)
    mask = im > th
    
    output_name = path.dirname(flist[0]) + '/G_mask.tif'
    io.imsave(output_name,mask.astype(np.int16),check_contrast=False)

#%%

io.imshow(im[44,...])
plt.figure()
io.imshow(mask[44,...])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 9

@author: xies
"""

import numpy as np
from cellpose import models
from cellpose import io

from os import path
from shutil import move
from os import mkdir
from time import time
from tqdm import tqdm

model = models.Cellpose(model_type='cyto2')

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/'

diameter = 37 #OK for 1.5x BE basal cells at 1.4 zoomin
anisotropy = 1.0
cellprob_threshold = -0.1

# Load the raw image (RGB,Z,X,Y)

OVERWRITE = False

B = io.imread(path.join(dirname,'Cropped_images/B.tif'))
G = io.imread(path.join(dirname,'Cropped_images/G.tif'))

for t,im in tqdm(enumerate(G)):

    im_nuc = B[t,...]
    d = dirname
    output_dir = path.join(d,f'3d_cyto_seg_supra/3d_cyto_supra_raw/t{t}_seg')
    if path.exists( output_dir ) and not OVERWRITE:
        print(f'Skipping {t}')
        continue
    else:
        mkdir(output_dir)

    tic = time()
    print(f'\n--- Predicting on {t} ---')
    npy_savepath = path.join(output_dir,f't{t}') + '_seg.npy'
    mask_savepath = path.join(output_dir,f't{t}_masks.tif')
    print(f'Saving model to: {npy_savepath}')

    masks,flows,styles,diams = model.eval([im,im_nuc],diameter=diameter,channels=[1,0], do_3D=True,
                                          cellprob_threshold=cellprob_threshold,
                                          anisotropy=anisotropy)
    #    io.masks_flows_to_seg(im, masks,flows,diams,npy_savepath)
	
	# Also resave the mask as standalone .tif for convenience
    io.imsave(mask_savepath,masks)

    toc = time()
    print(f'Saved to {output_dir}')
    print(f'Processed in {toc- tic:0.4f} seconds')

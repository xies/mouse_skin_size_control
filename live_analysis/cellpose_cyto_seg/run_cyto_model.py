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

model = models.Cellpose(model_type='cyto')

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2/'

diameter = 26 #OK for 1.5x BE basal cells at 1.4 zoomin
anisotropy = 1.0
cellprob_threshold = -0.1

# Load the raw image (RGB,Z,X,Y)

OVERWRITE = False

B = io.imread(path.join(dirname,'master_stack/B_blur_sobel.tif'))

for t,im in tqdm(enumerate(B)):

    f = path.join(dirname,f'cellpose_B_blur_sobel/t{t}.tif')
    d = path.dirname(f)
    
    basename = path.splitext(path.basename(f))[0] # i.e. 't9'
    output_dir = path.join(d,basename + '_3d_cyto')
    if path.exists( output_dir ) and not OVERWRITE:
        print(f'Skipping {f}')
        continue
    else:
        mkdir(output_dir)

    tic = time()
    print(f'Predicting on {f}')
    
    masks,flows,styles,diams = model.eval(im,diameter=None, do_3D=True,
                                          cellprob_threshold=cellprob_threshold, anisotropy=anisotropy)
    io.masks_flows_to_seg(im, masks,flows,diams,f)
	# annoyingly, need to manually move
    move(path.join(d, basename + '_seg.npy'), path.join(output_dir,basename + '_seg.npy'))
	
	# Also resave the mask as standalone .tif for convenience
    io.imsave(path.join(output_dir, basename + '_masks.tif'),masks)

    toc = time()
    print(f'Saved to {output_dir}')
    print(f'Processed in {toc- tic:0.4f} seconds')

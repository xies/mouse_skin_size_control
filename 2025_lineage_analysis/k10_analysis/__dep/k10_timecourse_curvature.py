#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:17:02 2026

@author: xies
"""


import numpy as np
import pandas as pd

from os import path
from skimage import io, util
from glob import glob
from tqdm import tqdm

times = [0,12,36,48]

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/Shared/K10 paw/Time course'
R = io.imread(path.join(dirname,'R.tif'))
G = io.imread(path.join(dirname,'G.tif'))
B = io.imread(path.join(dirname,'B.tif'))

#%%

from collections import defaultdict

dz = 0.5; dx= 0.3
sigmas = [5/dz,5/dx,5/dx]

from measurements import get_bm_image, get_mesh_from_bm_image, get_tissue_curvature_over_grid

manual = defaultdict(dict)
manual[2] = {'method':'maximum'}

height_images = []
curvatures = []
# heightmaps = []
for t in range(4):
    
    heightmaps,height_image = get_bm_image(B[t,94:],sigmas = sigmas, gradient_sign=+1,
                                           **manual[t])
    height_images.append(height_image)
    io.imsave(path.join(dirname,f'height_images/t{t}.tif'), height_image)
    
    print('Curvature')
    mesh = get_mesh_from_bm_image(height_images[t],spacing=[dz,dx,dx])
    mean_curvature,_ = get_tissue_curvature_over_grid(mesh, height_images[t].shape,
                                                      spacing=[dz,dx,dx])
    curvatures.append(mean_curvature)
    io.imsave(path.join(dirname,f'curvatures/t{t}.tif'), mean_curvature)
    
    
    
    
    
#%%


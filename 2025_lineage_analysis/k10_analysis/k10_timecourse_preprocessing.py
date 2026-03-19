#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:00:31 2026

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

ragged_R = []
ragged_G = []
ragged_B = []
for i,t in enumerate(times):
    im = io.imread(path.join(dirname,f'20181009_K10rtTA_pTRE_mG_H2BmCh_male1_P19_paw_{t}h_area1.tif'))
    ragged_R.append(im[...,0])
    # io.imsave(path.join(dirname,f'Raw channels/R/t{i}.tif'),im[...,0])
    ragged_G.append(im[...,1])
    # io.imsave(path.join(dirname,f'Raw channels/G/t{i}.tif'),im[...,1])
    ragged_B.append(im[...,2])
    # io.imsave(path.join(dirname,f'Raw channels/B/t{i}.tif'),im[...,2])

#%%

from measurements import get_bm_image

dz = 0.5; dx= 0.3
sigmas = [5/dz,5/dx,5/dx]

heightmaps = []
height_images = []

for t,B in tqdm(enumerate(ragged_B)):

    heightmap, height_image,blur = get_bm_image(B,sigmas=sigmas,
                                           gradient_sign = +1,
                                           return_gradient=True)
    heightmaps.append(heightmap)
    height_images.append(height_image)
    # io.imsave(path.join(dirname,f'Raw images/heightmaps/t{t}.tif'),heightmap)
    # io.imsave(path.join(dirname,f'Raw images/height_images/t{t}.tif'),height_image)
    # io.imsave(path.join(dirname,f't{t}_blur.tif'),blur)

#%%

from measurements import get_mesh_from_bm_image, get_tissue_curvature_over_grid

curvatures = []
for t in tqdm(range(4)):

    mesh = get_mesh_from_bm_image(height_images[t],spacing=[dz,dx,dx])
    mean_curvature,_ = get_tissue_curvature_over_grid(mesh, height_images[t].shape,
                                                      spacing=[dz,dx,dx])
    curvatures.append(mean_curvature)
    io.imsave(path.join(dirname,f'Raw images/curvatures/t{t}.tif'), mean_curvature)

#%% Generate basal collagen images

from skimage import transform, exposure


Ytrim = slice(300,800)
Xtrim = slice(300,800)

mean_heights = np.array([h[Ytrim,Xtrim].mean().astype(int) for h in heightmaps])
mean_heights[2] = 74

transformation_mats = {}
ref = curvatures[0][Ytrim,Xtrim]
curvature_transformed = [curvatures[0]]

ragged_transformed_B = [exposure.rescale_intensity(ragged_B[0][mean_heights[0],...],
                                                   out_range=(0,1))]

basal_collagen = np.zeros((4,*heightmap.shape))
basal_collagen[0,...] = exposure.rescale_intensity(ragged_B[0][mean_heights[0],...],
                                                   out_range=(0,1))

for t in tqdm(range(1,4)):
    basal_collagen[t,...] = exposure.rescale_intensity(ragged_B[t][mean_heights[t],...],
        out_range=(0,1))

io.imsave(path.join(dirname,'Raw images/basal_collagen.tif'),basal_collagen)

#%%

from imageUtils import z_align_ragged_timecourse

coarse_Tmats = np.load(path.join(dirname,'coarse_Tmats.npy'))

ragged_transformed_R = []

# manuals = {
#     0:transform.EuclideanTransform(translation=[-40,-40],rotation=np.deg2rad(-5)),
#     2:transform.EuclideanTransform(translation=[0,-60],rotation=np.deg2rad(2)),
#     3:transform.EuclideanTransform(rotation=np.deg2rad(10),
#                                           translation=[120,-150])}

for t in range(4):

    R = ragged_R[t].astype(float).copy()
    R_transformed = np.zeros_like(R)
    T = coarse_Tmats[t,...]
    # if t in manuals.keys():
    #     T = T + manuals[t]
    for z,im in enumerate(R):
        R_transformed[z,...] = transform.warp( im,T )
    ragged_transformed_R.append(R_transformed)

R = z_align_ragged_timecourse(ragged_transformed_R,mean_heights)
# io.imsave(path.join(dirname,'R.tif'),R)

for t in range(4):
    io.imsave(path.join(dirname,f'Raw channels/R_coarse/t{t}.tif'),R[t,...])

#%% Load coarse Tmats, pad, and transform

coarse_Tmats = np.load(path.join(dirname,'coarse_Tmats.npy'))
rotations = np.load(path.join(dirname,'refinement_Tmats.npz'))['rotation']
translations = np.load(path.join(dirname,'refinement_Tmats.npz'))['translation']

ragged_transformed_R = []
ragged_transformed_G = []
ragged_transformed_B = []

for t in range(4):

    R = ragged_R[t].astype(float).copy()
    R_transformed = np.zeros_like(R)
    G = ragged_G[t].astype(float).copy()
    G_transformed = np.zeros_like(G)
    B = ragged_B[t].astype(float).copy()
    B_transformed = np.zeros_like(B)

    Tc = transform.EuclideanTransform(coarse_Tmats[t,...])
    Tt = transform.EuclideanTransform(translation = -translations[t,[2,1]])
    Tr = transform.EuclideanTransform(rotation = np.arccos(rotations[t,1,1]))
    print(f't = {t}:')
    print(f'Translation = {translations[t,[2,1]]}')
    print(f'Translation = {np.degrees(np.arccos(rotations[t,1,1]))}')

    T = Tc + Tt + Tr

    for z,im in enumerate(R):
        R_transformed[z,...] = transform.warp( im, T)
    ragged_transformed_R.append(R_transformed)


    for z,im in enumerate(G):
        G_transformed[z,...] = transform.warp( im, T)
    ragged_transformed_G.append(G_transformed)


    for z,im in enumerate(B):
        B_transformed[z,...] = transform.warp( im, T)
    ragged_transformed_B.append(B_transformed)


print('Z-padding...')
R = z_align_ragged_timecourse(ragged_transformed_R,mean_heights)
G = z_align_ragged_timecourse(ragged_transformed_G,mean_heights)
B = z_align_ragged_timecourse(ragged_transformed_B,mean_heights)

print('Saving...')
io.imsave(path.join(dirname,'R.tif'),R)
io.imsave(path.join(dirname,'G.tif'),G)
io.imsave(path.join(dirname,'B.tif'),B)


# # print('Saving...')
# # for t in range(4):
# io.imsave(path.join(dirname,f't1.tif'),R[1,...])

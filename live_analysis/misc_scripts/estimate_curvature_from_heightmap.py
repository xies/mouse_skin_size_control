#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:32:30 2022

@author: xies
"""

import numpy as np
from skimage import io, filters
from os import path
from glob import glob

from tqdm import tqdm
from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
from scipy.spatial import Delaunay

dirname = dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

#%%

t = 0

sigma = 20 # should be roughly 1/2-1/4 of the diameter of a cell

im = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))

# Gaussian smooth
im_blur = filters.gaussian(im.astype(float), sigma = sigma)
im_blur = np.round(im_blur).astype(int)
io.imshow(im_blur)

#%%

XX,YY = np.meshgrid(range(460),range(460))

coords3d = np.array([XX.flatten(),YY.flatten()]).T
 
tri_dense = Delaunay(coords3d)

mesh = Trimesh(vertices = np.array([XX.flatten(),YY.flatten(),im_blur.flatten()]).T, faces=tri_dense.simplices)
mean_curve = discrete_mean_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
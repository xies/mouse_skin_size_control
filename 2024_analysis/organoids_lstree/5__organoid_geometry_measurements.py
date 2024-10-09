#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:45:00 2024

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, measure
from os import path
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sb
import pyvista as pv
import trimesh as tm


dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)

dx = 0.26
dz = 2

T = 65

#%% Calculate cell position WRT spherical coordinates of organoid mesh

# Decimate dataframe into frames
df_by_frame = {k:v for k,v in df.groupby('Frame')}

for t in tqdm(range(T-1)):
    
    # Load organoid mesh
    mesh = pv.read(path.join(dirname,f'harmonic_mesh/shmesh_lmax5_t{t+1:04d}.vtk'))
    
    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
    tmesh = tm.Trimesh(mesh.points, faces_as_array)
    
    # calculate organoid curvature at each cell coordinate
    
    cell_points = df_by_frame[t][['Z','Y','X']]
    curvatures = tm.discrete_mean_curvature_measure(mesh,cell_points,radius=0)
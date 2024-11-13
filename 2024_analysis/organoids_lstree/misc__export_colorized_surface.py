#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:16:56 2024

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
from skimage import io
import trimesh as tm
import pickle as pkl
import pyvista as pv

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

dx = 0.26
dz = 2
ymax = 105.04
xmax = 102.96000000000001
zmax = 84

t = 64
# Surface file
mesh = pv.read(path.join(dirname,f'harmonic_mesh/shmesh_lmax5_t{t+1:04d}.vtk'))
vertices = np.asarray(mesh.points)
faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
tmesh = tm.Trimesh(mesh.points, faces_as_array)

#%%

kappa_radius = 15

curvatures = tm.curvature.discrete_mean_curvature_measure(tmesh,vertices,radius = kappa_radius)/kappa_radius
faces = np.asarray(mesh.faces).reshape((-1,4))[:,1:]
values = curvatures

np.savez(path.join(dirname,f'visualization_on_mesh/curvature/t{t+1:04d}.npz'),
          vertices,faces,values)

#%% Extract geodesic neighbors

from scipy.spatial import distance

neighborhood_distance = 20

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)
df_by_frame = {k:v for k,v in df.groupby('Frame')}
cell_points = df_by_frame[t][['Z','Y','X']]

DistMat_cells = distance.squareform(distance.pdist(cell_points))
DistMat_cells[np.eye(len(vertices)).astype(bool)] = np.nan

# Define 'neighborhood'
neighborhood_distance = 10 #um

cell_neighbors = {}
cell_neighbors_idx = {}
for i,pt in enumerate(cell_points.values):

    # Find the df entries of all cells within distance
    neighbors = df.loc[cell_points.iloc[np.where(DistMat_cells[i,:] < neighborhood_distance)[0]].index]
    cell_neighbors_idx[cell_points.iloc[i].name] = neighbors.index
    cell_neighbors[df_by_frame[t].iloc[i]['cellID']] = neighbors['cellID']


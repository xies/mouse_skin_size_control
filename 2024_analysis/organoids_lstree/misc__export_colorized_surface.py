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
import matplotlib.pyplot as plt

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

dx = 0.26
dz = 2
ymax = 105.04
xmax = 102.96000000000001
zmax = 84


for t in range(64):
    
    # Surface file
    mesh = pv.read(path.join(dirname,f'harmonic_mesh/shmesh_lmax5_t{t+1:04d}.vtk'))
    vertices = np.asarray(mesh.points)
    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
    tmesh = tm.Trimesh(mesh.points, faces_as_array)
    
    kappa_radius = 15
    
    curvatures = tm.curvature.discrete_mean_curvature_measure(tmesh,vertices,radius = kappa_radius)/kappa_radius
    faces = np.asarray(mesh.faces).reshape((-1,4))[:,1:]
    values = curvatures
    
    np.savez(path.join(dirname,f'visualization_on_mesh/curvature/t{t+1:04d}.npz'),
              vertices,faces,values)

#%% Extract from dataframe

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'),index_col=0)
collated = {k:v for k,v in df.groupby('trackID')}
data = collated[80]

plt.figure()
plt.plot(data['Local cell density'])
plt.plot(data['Change in local cell density'])

plt.figure()
plt.plot(data['Mean neighbor Gem intensity'])

plt.figure()
plt.plot(data['Mean curvature'])

#%% Extract neighborhood mesh

t = 1

trackID = 1
cellID = collated[trackID][collated[trackID].Frame == t]['cellID'].iloc[0]

mesh = pv.read(path.join(dirname,f'harmonic_mesh/shmesh_lmax5_t{t+1:04d}.vtk'))

with open(path.join(dirname,f'geometric_neighbors/geometric_neighbors_T{t+1:04d}.pkl'),'rb') as f:
    cell_neighbors = pkl.load(f)

neighborIDs = cell_neighbors[cellID]

with open(path.join(dirname,f'manual_seg_mesh/individual_mesh_by_cellID_T{t+1:04d}.pkl'),'rb') as f:
    cell_meshes = pkl.load(f)

pl = pv.Plotter()
pl.add_mesh(mesh)
center = cell_meshes[cellID]
pl.add_mesh(center,color='r')

for neighborID in neighborIDs.values:
    pl.add_mesh(cell_meshes[neighborID],color='b')
    
pl.show()



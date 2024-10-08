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
import trimesh as tm
import networkx as nx

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)

dx = 0.26
dz = 2

T = 65

def find_nearest_vertex(tmesh,query_pts,face_idx):
    assert(len(query_pts) == len(face_idx))
    vert_idx = np.zeros_like(face_idx)
    for i,pt in enumerate(query_pts):
        candidate_verts = tmesh.vertices[tmesh.faces[face_idx[i]]]
        D = ((pt - candidate_verts)**2).sum(axis=0)
        vert_idx[i] = tmesh.faces[face_idx[i]][D.argmin()]
    return vert_idx

#%% Calculate cell position WRT spherical coordinates of organoid mesh

kappa_radius = 2

# Decimate dataframe into frames
df_by_frame = {k:v for k,v in df.groupby('Frame')}

# for t in tqdm(range(T-1)):
t = 0

import pyvista as pv
# Load organoid mesh
mesh = pv.read(path.join(dirname,f'harmonic_mesh/shmesh_lmax5_t{t+1:04d}.vtk'))

faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
tmesh = tm.Trimesh(mesh.points, faces_as_array)

# calculate organoid curvature at each cell coordinate
cell_points = df_by_frame[t][['Z','Y','X']]

# # Query for nearest point and then calculate curvature on organoid
query_on_surface,_,face_idx = tmesh.nearest.on_surface(cell_points)
vert_idx = find_nearest_vertex(tmesh,cell_points.values,face_idx)

curvatures = tm.curvature.discrete_mean_curvature_measure(tmesh,query_on_surface,radius = kappa_radius)/kappa_radius

#Define geodesic distance
DistMat_cells = np.ones((len(vert_idx),len(vert_idx))) * np.nan
for i,pt in enumerate(vert_idx):
    for j,other_pt in enumerate(vert_idx):
        if i > j:
            path = mesh.geodesic(pt,other_pt)
            DistMat_cells[i,j] = path.length

# Define 'neighborhood'
neighborhood_distance = 20 #um

cell_neighbors = {}
for i,pt in enumerate(cell_points.values):
    
    # Find the df entries of all cells within distance
    neighbors = df.loc[cell_points.iloc[np.where(DistMat_cells[i,:] < neighborhood_distance)[0]].index]
    cell_neighbors[cell_points.iloc[i].name] = neighbors.index
    
    
    
    


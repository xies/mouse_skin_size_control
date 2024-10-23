#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:45:00 2024

@author: xies
"""

import numpy as np
import pandas as pd
# from skimage import io, measure
from os import path
import matplotlib.pyplot as plt
from tqdm import tqdm
# import seaborn as sb
import trimesh as tm
import pyvista as pv
import pickle as pkl

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 2_2um/'

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)

dx = 0.26
dz = 2

T = 45

def find_nearest_vertex(tmesh,query_pts,face_idx):
    assert(len(query_pts) == len(face_idx))
    vert_idx = np.zeros_like(face_idx)
    for i,pt in enumerate(query_pts):
        candidate_verts = tmesh.vertices[tmesh.faces[face_idx[i]]]
        D = ((pt - candidate_verts)**2).sum(axis=0)
        vert_idx[i] = tmesh.faces[face_idx[i]][D.argmin()]
    return vert_idx

#%% Calculate cell position WRT spherical coordinates of organoid mesh

RECALCULATE_NEIGHBORHOOD = False
kappa_radius = 15

# Decimate dataframe into frames
df_by_frame = {k:v for k,v in df.groupby('Frame')}

for t in tqdm(range(T)):

    # Load organoid shape mesh
    mesh = pv.read(path.join(dirname,f'harmonic_mesh/shmesh_lmax5_t{t+1:04d}.vtk'))
    df_by_frame[t]['Organoid volume'] = mesh.volume
    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
    tmesh = tm.Trimesh(mesh.points, faces_as_array)
    
    # Categorize cells that are entirely interior ro the organoid
    with open(path.join(dirname,f'manual_seg_mesh/individual_mesh_by_cellID_T{t+1:04d}.pkl'),'rb') as f:
        cell_meshes = pkl.load(f)
        
    for cellID,cell_mesh in cell_meshes.items():
        df_by_frame[t].loc[df_by_frame[t]['cellID'] == cellID,'Organoid interiority'] = \
            -tm.proximity.signed_distance(tmesh,cell_mesh.points).sum()
            
    # calculate organoid curvature at each cell coordinate
    cell_points = df_by_frame[t][['Z','Y','X']]
    # # Query for nearest point and then calculate curvature on organoid
    query_on_surface,_,face_idx = tmesh.nearest.on_surface(cell_points)
    vert_idx = find_nearest_vertex(tmesh,cell_points.values,face_idx)
    
    curvatures = tm.curvature.discrete_mean_curvature_measure(tmesh,query_on_surface,radius = kappa_radius)/kappa_radius
    df_by_frame[t]['Mean curvature'] = curvatures
    
    if path.exists(path.join(dirname,f'geodesic_neighbors/geodesic_distmat_T{t+1:04d}.pkl')) and not RECALCULATE_NEIGHBORHOOD:
        with open(path.join(dirname,f'geodesic_neighbors/geodesic_distmat_T{t+1:04d}.pkl'),'rb') as f:
            DistMat_cells = pkl.load(f)
    else:
        #Define geodesic distance
        DistMat_cells = np.ones((len(vert_idx),len(vert_idx))) * np.nan
        for i,pt in enumerate(vert_idx):
            for j,other_pt in enumerate(vert_idx):
                if i > j:
                    try:
                        geod = mesh.geodesic(pt,other_pt)
                        l = geod.length
                    except:
                        l = 1000 # placeholder
                    DistMat_cells[i,j] = l
        # Save distmat
        with open(path.join(dirname,f'geodesic_neighbors/geodesic_distmat_T{t+1:04d}.pkl'),'wb') as f:
            pkl.dump(DistMat_cells,f)
    
    # Define surface normal of surface
    normals = tmesh.vertex_normals[tmesh.faces[face_idx]]
    # find 'centroid' of triangle face
    bary = tm.triangles.points_to_barycentric(triangles=tmesh.triangles[face_idx], points=query_on_surface)
    norm_vecs = tm.unitize((normals * bary.reshape((-1, 3, 1))).sum(axis=1))
    # Export surface normals
    pd.DataFrame(np.hstack((query_on_surface,norm_vecs)),
                 columns=['Z','Y','X','Normal-0','Normal-1','Normal-2']).to_csv(path.join(dirname,f'harmonic_mesh/surface_normals_T{t+1:04d}.csv'))
    # Cell orientation wrt to surface normal
    orientations = np.zeros(len(norm_vecs))
    for i,n in enumerate(norm_vecs):
        orientations[i] = np.dot(n,df_by_frame[t].iloc[i][['Principal axis-0','Principal axis-1','Principal axis-2']])
    df_by_frame[t]['Orientation'] = orientations
    
    # Define 'neighborhood'
    neighborhood_distance = 20 #um
    
    cell_neighbors = {}
    cell_neighbors_idx = {}
    for i,pt in enumerate(cell_points.values):
    
        # Find the df entries of all cells within distance
        neighbors = df.loc[cell_points.iloc[np.where(DistMat_cells[i,:] < neighborhood_distance)[0]].index]
        cell_neighbors_idx[cell_points.iloc[i].name] = neighbors.index
        cell_neighbors[df_by_frame[t].iloc[i]['cellID']] = neighbors['cellID']
    
    # Propagate information on the neighborhood
    for center_idx,neighbors_idx in cell_neighbors_idx.items():
    
        # center_cell = df.loc[center_cell]
        neighbors = df.loc[neighbors_idx]
        if len(neighbors) > 0:
            # Mean/std neighbor volume
            df_by_frame[t].at[center_idx,'Mean neighbor volume'] = neighbors['Mesh volume'].mean()
            df_by_frame[t].at[center_idx,'Std neighbor volume'] = neighbors['Mesh volume'].std()
            df_by_frame[t].at[center_idx,'Mean neighbor H2B intensity'] = neighbors['Normalized H2B intensity'].mean()
            df_by_frame[t].at[center_idx,'Mean neighbor Cdt1 intensity'] = neighbors['Normalized Cdt1 intensity'].mean()
            df_by_frame[t].at[center_idx,'Mean neighbor Gem intensity'] = neighbors['Normalized Gem intensity'].mean()
            df_by_frame[t].at[center_idx,'Local cell density'] = len(neighbors)
    
    # Save neighborhood information
    with open(path.join(dirname,f'geodesic_neighbors/geodesic_neighbors_T{t+1:04d}.pkl'),'wb') as f:
        pkl.dump(cell_neighbors,f)
    with open(path.join(dirname,f'geodesic_neighbors/geodesic_neighbors_dfindex_T{t+1:04d}.pkl'),'wb') as f:
        pkl.dump(cell_neighbors_idx,f)
    
# Recombine into dataframe
df_combined = pd.concat(df_by_frame,ignore_index=True)
df_combined = df_combined.drop(columns='index')
                      
df_combined.to_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features.csv'))




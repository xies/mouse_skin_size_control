#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:25:38 2024

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, measure
from os import path
from scipy.spatial import cKDTree as KDTree
import meshFMI

from tqdm import tqdm

from aicsshparam import shtools, shparam
import pyvista as pv

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

dx = 0.26
dz = 2

#%%

for t in tqdm(np.arange(1,65)):

    labels = io.imread(path.join(dirname,f'manual_segmentation/man_Channel0-T{t:04d}.tif'))
    
    # Extract the coords
    df = pd.DataFrame(measure.regionprops_table(labels,properties=['area','label','centroid']))
    df = df.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X','area':'Nuclear volume'})
    df['X'] *= dx
    df['Y'] *= dx
    df['Z'] *= dz
    
    # Center the organoid to origin
    organoid_coords = df[['Z','Y','X']].values
    centroid = organoid_coords.mean(axis=0)
    organoid_coords = organoid_coords - centroid
    
    # Raw mesh
    ptCloud = pv.PolyData((organoid_coords))
    mesh = ptCloud.reconstruct_surface()
    
    # Smoothed using spherical harmonics
    (coeffs, _), _ = shparam.get_shcoeffs(image=organoid_coords, lmax=5, POINT_CLOUD=True)
    coeffs_mat = shtools.convert_coeffs_dict_to_matrix(coeffs, lmax=5)
    smooth_mesh,grid = shtools.get_even_reconstruction_from_coeffs(coeffs_mat, npoints=8192)
    smooth_mesh = pv.PolyData(smooth_mesh)
    
    # Re-translate to original centroid
    smooth_mesh.translate(centroid, inplace=True)
    smooth_mesh.triangulate()
    mesh.translate(centroid, inplace=True)
    ptCloud.translate(centroid, inplace=True)
    
    # p = pv.Plotter()
    # p.add_mesh(smooth_mesh)
    # p.add_points(ptCloud, color='red')
    
    # p.show()
    
    # Save VTK
    smooth_mesh.save(path.join(dirname,f'harmonic_mesh/shmesh_lmax5_t{t:04d}.vtk'))
    
    # Export vert, face, value tuple for napari usage
    vertices = np.asarray(smooth_mesh.points)
    faces = np.asarray(smooth_mesh.faces).reshape((-1,4))[:,1:]
    normals = np.asarray(smooth_mesh.point_normals)
    values = np.dot(normals,[1,-1,1])
    np.savez(path.join(dirname,f'harmonic_mesh/shmesh_lmax5_t{t:04d}.npz'),
              vertices,faces,values)
    
    # Render all cells using mesh
    
    # Export pretty mesh per time point
    pretty_mesh = meshFMI.labels_to_mesh(labels,[dz,dx,dx],show_progress=False)
    pretty_mesh = pv.PolyData( pretty_mesh )
    pretty_mesh.save(path.join(dirname,f'manual_seg_mesh/pretty_mesh_t{t:04d}.vtk'))
    
    # Export vert, face, value tuple for napari usage
    v = np.asarray(pretty_mesh.points)
    v = np.swapaxes(v,0,1)
    f = np.asarray(pretty_mesh.faces).reshape((-1,4))[:,1:]
    val = np.asarray(pretty_mesh['label_id'])
    np.savez(path.join(dirname,f'manual_seg_mesh/pretty_mesh_t{t:04d}.npz'),
              v,f,val)


#%%

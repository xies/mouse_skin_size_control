#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:00:15 2025

@author: xies
"""

# Core libraries
import numpy as np
from skimage import transform, measure, io
import matplotlib.pylab as plt
# import seaborn as sb

# Specific utils
from aicsshparam import shtools,shparam
from trimesh import Trimesh
import pyvista as pv

# General utils
from tqdm import tqdm
from os import path,makedirs
import pickle as pkl

from measurements import estimate_sh_coefficients

dt = 12 #hrs
dx = 0.25
dz = 1
Z_SHIFT = 10
KAPPA = 5 # microns

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

# Load segmentations
tracked_nuc = io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif'))
tracked_cyto = io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif'))
tracked_manual_cyto = io.imread(path.join(dirname,'Mastodon/tracked_manual_cyto.tif'))

#%%

def reconstruct_mesh(coeffs,lmax=5):
    coeffs = coeffs.to_dict()
    coeffs = {'_'.join(k.split('_')[1:3]):v for k,v in coeffs.items()}
    # Convert to matrix
    mat = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float32)
    for L in range(lmax):
        for M in range(L + 1):
            for cid, C in enumerate(["C", "S"]):
                key = f"shcoeffs_L{L}M{M}{C}"
                if key in coeffs.keys():
                    mat[cid, L, M] = coeffs[key]
                else:
                    mat[cid,L,M] = 0
    mesh = shtools.get_even_reconstruction_from_coeffs(mat)
    return mesh



#%% 
# p = measure.regionprops_table(mask.astype(int),properties=['centroid'])
# verts[:,0] -= p['centroid-2']
# verts[:,1] -= p['centroid-1']
# verts[:,2] -= p['centroid-0']

# np.savez('/Users/xies/Desktop/example_mesh.npz',
#          {'verts':verts,'faces':faces,'normals':normals,'values':values})

#%%




#%%


mask =  transform.resize(mask, mask.shape*np.array([dz/dx,1,1]))
aligned_masks = {1: np.squeeze(shtools.align_image_2d(mask)[0])}

verts,faces,normals,values = measure.marching_cubes(aligned_masks[1])
verts = verts[:,[2,1,0]]
verts = verts - verts.mean(axis=0)
original_mesh = trimesh.Trimesh(verts,faces)

# Parametrize with SH coefficients and record
(coeffs,_),_ = shparam.get_shcoeffs(image=aligned_masks[1], lmax=5)
M = shtools.convert_coeffs_dict_to_matrix(coeffs,lmax=5)
mesh = shtools.get_even_reconstruction_from_coeffs(M)[0]

#%%

pl = pv.Plotter()
pl.add_mesh(mesh,opacity=0.5)
pl.add_mesh(pv.wrap(original_mesh),color='r',opacity=0.1)
pl.show()

#%%
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)


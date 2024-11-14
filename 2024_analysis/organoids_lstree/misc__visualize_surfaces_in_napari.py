#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:25:38 2024

@author: xies
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from os import path
from skimage import io
import pickle as pkl

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

dx = 0.26
dz = 2
ymax = 105.04
xmax = 102.96000000000001
zmax = 84

@dataclass
class Surface:
    vertices: np.array
    faces: np.array
    values: np.array

def load_surface_from_npz(filename,transpose=False):
    arr = np.load(filename)
    vertices = arr['arr_0']
    faces = arr['arr_1']
    values = arr['arr_2']
    if transpose:
        # VTK is in YXZ
        # vertices[:,1] = -vertices[:,1] + ymax
        # vertices[:,0] = -vertices[:,0] + zmax
        vertices = vertices[:,[2,1,0]]
        # values = np.random.permutation(values)
        # vertices[:,2] = -vertices[:,2] + xmax
    surf = Surface(vertices,faces,values)
    return surf

t = 0

# filename = path.join(dirname,f'harmonic_mesh/shmesh_lmax5_T{t+1:04d}.npz')
# organoid_surface = load_surface_from_npz(filename)
# viewer.add_surface((organoid_surface.vertices,organoid_surface.faces,organoid_surface.values)
#         ,name='organoid')

filename = path.join(dirname,f'visualization_on_mesh/curvature/T{t+1:04d}.npz')
organoid_surface = load_surface_from_npz(filename)
viewer.add_surface((organoid_surface.vertices,organoid_surface.faces,organoid_surface.values)
        ,name='organoid',colormap='twilight_shifted')
#
# filename = path.join(dirname,f'manual_seg_mesh/pretty_mesh_T{t+1:04d}.npz')
# rot = load_surface_from_npz(filename,transpose=False)
# all_segs = viewer.add_surface((rot.vertices,rot.faces,rot.values)
#         ,name='all_segmentations',colormap='magma')

# im = io.imread(path.join(dirname,f'manual_segmentation/man_Channel0-T{t+1:04d}.tif'))
# viewer.add_labels(im,name='labels', scale=[2,.26,.26])

# Visualize vectors
# vectors = pd.read_csv(path.join(dirname,f'manual_seg_mesh/principal_vector_cellID_T{t+1:04d}.csv'))
# pos = np.zeros((len(vectors),2,3))
# pos[:,0,:] = vectors[['Z','Y','X']].values
# pos[:,1,:] = vectors[['Principal axis-0','Principal axis-1','Principal axis-2']].values
# viewer.add_vectors(pos, edge_width=1, length=10,name='Cell axes')

# vectors = pd.read_csv(path.join(dirname,f'harmonic_mesh/surface_normals_T{t+1:04d}.csv'))
# pos = np.zeros((len(vectors),2,3))
# pos[:,0,:] = vectors[['Z','Y','X']].values
# pos[:,1,:] = vectors[['Normal-0','Normal-1','Normal-2']].values
# viewer.add_vectors(pos, edge_width=1, length=10,name='Surface normals')

im = io.imread(path.join(dirname,f'Channel0-Deconv/Channel0-T{t+1:04d}.tif'))
raw_image = viewer.add_image(im,name='image', scale=[2,.26,.26],rendering='attenuated_mip',blending='additive'
    ,contrast_limits=[0,30000],attenuation=1)

# animate

from napari_animation import Animation

# animation = Animation(viewer)

# Rocking animations with raw image only
viewer.dims.ndisplay=3
viewer.camera.angles = (0,0,90)
raw_image.visible=True
all_segs.visible=False
viewer.camera.angles = (0,180,90)

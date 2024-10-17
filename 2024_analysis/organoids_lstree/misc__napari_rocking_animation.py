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

t = 63

# Use segmentations
filename = path.join(dirname,f'manual_seg_mesh/pretty_mesh_T{t+1:04d}.npz')
rot = load_surface_from_npz(filename,transpose=False)
segs = viewer.add_surface((rot.vertices,rot.faces,rot.values)
        ,name='all_segmentations',colormap='magma')

# Use organoid surface model
filename = path.join(dirname,f'harmonic_mesh/shmesh_lmax5_T{t+1:04d}.npz')
organoid_surface = load_surface_from_npz(filename)
organoid = viewer.add_surface((organoid_surface.vertices,organoid_surface.faces,organoid_surface.values)
        ,name='organoid', colormap='orange', opacity=0.6)

im = io.imread(path.join(dirname,f'Channel0-Deconv/Channel0-T{t+1:04d}.tif'))
raw_image = viewer.add_image(im,name='image', scale=[2,.26,.26],rendering='attenuated_mip',blending='additive'
    ,contrast_limits=[0,30000],attenuation=1)

# animate

from napari_animation import Animation

animation = Animation(viewer)

rotation_frames = 30

# Rocking animations with raw image only
viewer.dims.ndisplay=3
viewer.camera.angles = (0,0,90)
viewer.camera.zoom = 6

raw_image.visible=True
segs.visible = False
organoid.visible = False
animation.capture_keyframe()

# 180
viewer.camera.angles = (0,180,90)
animation.capture_keyframe(steps=rotation_frames)

#360
viewer.camera.angles = (0,359,90)
animation.capture_keyframe(steps=rotation_frames)

# Pause
viewer.camera.angles = (0,0,90)
animation.capture_keyframe(steps=10)

segs.visible = True
animation.capture_keyframe(steps=10)
# 180
viewer.camera.angles = (0,180,90)
animation.capture_keyframe(steps=rotation_frames)
#360
viewer.camera.angles = (0,359,90)
animation.capture_keyframe(steps=rotation_frames)

# Pause
viewer.camera.angles = (0,0,90)
animation.capture_keyframe(steps=10)
segs.visible = False
organoid.visible = True
animation.capture_keyframe(steps=10)
# 180
viewer.camera.angles = (0,180,90)
animation.capture_keyframe(steps=rotation_frames)
#360
viewer.camera.angles = (0,359,90)
animation.capture_keyframe(steps=rotation_frames)

# Pause
viewer.camera.angles = (0,0,90)
animation.capture_keyframe(steps=10)

animation.animate('/Users/xies/Desktop/rocking', canvas_only=True)
animation.animate('/Users/xies/Desktop/rocking.mov', canvas_only=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:14:40 2024

@author: xies

based on: https://github.com/napari/napari-animation/blob/main/examples/layer_planes.py

"""

from napari_animation import Animation
import napari
import skimage

from os import path

viewer = napari.Viewer(ndisplay=3)

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

max_size = 250

cortex_data = skimage.io.imread(path.join(dirname,'Cropped_images/G.tif'))[0,:,:max_size,:max_size]
nuc_data = skimage.io.imread(path.join(dirname,'Cropped_images/B.tif'))[0,:,:max_size,:max_size]
labels_data = skimage.io.imread(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/t0.tif'))[:,:max_size,:max_size]

nuc_layer = viewer.add_image(nuc_data,name = 'nuc',blending = 'translucent', depiction = 'plane', colormap='blue')
# cortex_layer = viewer.add_image(cortex_data,name = 'cortex',blending = 'additive', depiction='plane')
labels_layer = viewer.add_labels(labels_data,name = 'labels',blending = 'translucent')

animation = Animation(viewer)

viewer.camera.angles = (-18.23797054423494, 41.97404742075617, 141.96173085742896)
viewer.camera.zoom *= 0.3

for l in viewer.layers:
    l.scale = [4,1,1]

# Tie the truncation point to the slice position of nuc_layer
def replace_labels_data():
    z_cutoff = int(nuc_layer.plane.position[0])
    new_labels_data = labels_data.copy()
    new_labels_data[z_cutoff : ,...] = 0
    labels_layer.data = new_labels_data

# Go up and down the stack once
labels_layer.visible = False
nuc_layer.visible = True
# cortex_layer.visible = True
nuc_layer.plane.position = (0, 0, 0)
# cortex_layer.plane.position = (0, 0, 0)
animation.capture_keyframe(steps=30)

nuc_layer.plane.position = (71, 0, 0)
# cortex_layer.plane.position = (71, 0, 0)
animation.capture_keyframe(steps=30)

nuc_layer.plane.position = (0, 0, 0)
# cortex_layer.plane.position = (0, 0, 0)

animation.capture_keyframe(steps=30)

# Now clip the segmentation until the current plane

nuc_layer.plane.events.position.connect(replace_labels_data)
labels_layer.visible = True
labels_layer.experimental_clipping_planes = [{
    "position": (0, 0, 0),
    "normal": (-1, 0, 0),  # point up in z (i.e: show stuff above plane)
}]

nuc_layer.plane.position = (71, 0, 0)
# cortex_layer.plane.position = (71, 0, 0)
# access first plane, since it's a list
labels_layer.experimental_clipping_planes[0].position = (71, 0, 0)
animation.capture_keyframe(steps=30)

nuc_layer.plane.position = (0, 0, 0)
# cortex_layer.plane.position = (0, 0, 0)
animation.capture_keyframe(steps=30)

animation.animate("layer_planes.mp4", canvas_only=True)
nuc_layer.plane.position = (0, 0, 0)



